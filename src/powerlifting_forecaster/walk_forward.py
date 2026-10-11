"""Evaluate annual forecasts with model selection restricted to each fold's past."""

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from powerlifting_forecaster.baselines import DriftBaselines, evaluate_baselines
from powerlifting_forecaster.evaluation import score_predictions
from powerlifting_forecaster.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from powerlifting_forecaster.splits import VALIDATION_START
from powerlifting_forecaster.train import prepare_features
from powerlifting_forecaster.tune import tune_regressor


EVALUATION_YEARS = (2020, 2021, 2022)


def split_year(frame: pd.DataFrame, year: int) -> dict[str, pd.DataFrame]:
    """Reserve the preceding year for selection and the requested year for evaluation."""
    if not isinstance(year, int) or not 1901 <= year < pd.Timestamp(VALIDATION_START).year:
        raise ValueError("Evaluation years must precede the main validation period.")
    dates = pd.to_datetime(frame["target_date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any() or not dates.lt(pd.Timestamp(VALIDATION_START)).all():
        raise ValueError("Walk-forward inputs must contain only dated rows from the original training period.")
    if not frame.index.is_unique:
        raise ValueError("Source row indices must be unique.")
    masks = {
        "train": dates.lt(pd.Timestamp(year - 1, 1, 1)),
        "validation": dates.between(pd.Timestamp(year - 1, 1, 1), pd.Timestamp(year, 1, 1), inclusive="left"),
        "evaluation": dates.between(pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1), inclusive="left"),
    }
    parts = {name: frame.loc[mask].copy() for name, mask in masks.items()}
    if any(part.empty for part in parts.values()):
        raise ValueError("Each fold must have nonempty training, selection, and evaluation periods.")
    return parts


def evaluate_year(frame: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select on the preceding year, refit on all earlier rows, then score the next year."""
    parts = split_year(frame, year)
    _, selection = tune_regressor(
        parts["train"], parts["validation"],
        validation_start=f"{year - 1}-01-01", test_start=f"{year}-01-01",
    )
    refit = pd.concat([parts["train"], parts["validation"]])
    evaluation = parts["evaluation"]
    baseline_report = evaluate_baselines(
        refit, evaluation, validation_start=f"{year}-01-01", test_start=f"{year + 1}-01-01",
    )
    baselines = DriftBaselines(**baseline_report["fitted_parameters"])
    features = prepare_features(refit)
    categories = {column: features[column].cat.categories.tolist() for column in CATEGORICAL_FEATURES}
    model = lgb.LGBMRegressor(**{
        **selection["model_parameters"], "n_estimators": selection["best_iteration"],
    })
    # Tree count is fixed before refitting; evaluation outcomes never reach fit().
    model.fit(features, refit["delta_total"], categorical_feature=CATEGORICAL_FEATURES)
    predictions = baselines.predict(evaluation[["previous_total", "prior_meets"]])
    predictions["lightgbm"] = evaluation["previous_total"] + model.predict(
        prepare_features(evaluation, categories), validate_features=True,
    )
    predictions["source_row"] = evaluation.index
    predictions["evaluation_year"] = year
    predictions["target_date"] = evaluation["target_date"]
    predictions["actual_total"] = evaluation["target_total"]
    return predictions, {
        "evaluation_year": year,
        "selection": selection,
        "refit_rows": len(refit), "evaluation_rows": len(evaluation),
        "refit_end_exclusive": f"{year}-01-01",
        "evaluation_end_exclusive": f"{year + 1}-01-01",
        "refit_parameters": model.get_params(),
        "refit_tree_count": model.booster_.current_iteration(),
        "baseline_parameters": baseline_report["fitted_parameters"],
        "evaluation_metrics": score_predictions(predictions),
    }


def evaluate_walk_forward(
    frame: pd.DataFrame, years: tuple[int, ...] = EVALUATION_YEARS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate increasing calendar years and pool their disjoint prediction rows."""
    if not years or list(years) != sorted(set(years)):
        raise ValueError("Evaluation years must be nonempty, unique, and increasing.")
    predictions, folds = [], []
    for year in years:
        print(f"Evaluating year: {year}", flush=True)
        year_predictions, fold = evaluate_year(frame, year)
        predictions.append(year_predictions)
        folds.append(fold)
        metrics = fold["evaluation_metrics"]["lightgbm"]
        print(f"{year}: MAE {metrics['mae_kg']:.4f} kg", flush=True)
    combined = pd.concat(predictions, ignore_index=True)
    return combined, {
        "evaluation_years": list(years), "evaluation_rows": len(combined),
        "feature_columns": FEATURE_COLUMNS,
        "selection_policy": "Repeat the same four candidates within each fold using only the preceding validation year; refit the selected tree count on all earlier data.",
        "history_policy": "Predict each meet using information known before that meet; earlier results in an evaluation year remain available to later meets. Lifter overlap is allowed.",
        "holdout_policy": "Use only the original pre-2023 training file; the main validation and final test files are not loaded.",
        "pooled_metric_policy": "Score concatenated predictions; each meet has equal weight rather than each year.",
        "folds": folds, "pooled_metrics": score_predictions(combined),
        "years_improved_vs_experience_drift": sum(
            fold["evaluation_metrics"]["lightgbm"]["mae_kg"]
            < fold["evaluation_metrics"]["experience_drift"]["mae_kg"] for fold in folds
        ),
    }


def main() -> None:
    """Run the fixed annual evaluation and save its report and local predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/splits/train.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/walk_forward.json"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/walk_forward_predictions.csv"))
    args = parser.parse_args()
    with args.input.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    columns = ["target_date", "target_total", "delta_total", *FEATURE_COLUMNS]
    dtypes = {
        column: "string" if column in ["target_date", *CATEGORICAL_FEATURES] else "float64"
        for column in columns
    }
    frame = pd.read_csv(args.input, usecols=columns, dtype=dtypes, keep_default_na=False, na_values=[""])
    predictions, report = evaluate_walk_forward(frame)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions, index=False)
    with args.predictions.open("rb") as source:
        prediction_checksum = hashlib.file_digest(source, "sha256").hexdigest()
    report = {
        "input_file": args.input.name, "input_sha256": checksum, **report,
        "prediction_file": args.predictions.name, "prediction_sha256": prediction_checksum,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
