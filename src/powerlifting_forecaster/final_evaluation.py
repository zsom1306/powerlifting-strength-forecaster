"""Refit the validation-selected model and evaluate the reserved final test period."""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from powerlifting_forecaster.baselines import DriftBaselines
from powerlifting_forecaster.evaluation import score_predictions
from powerlifting_forecaster.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from powerlifting_forecaster.splits import TEST_START, VALIDATION_START
from powerlifting_forecaster.train import load_training_splits, prepare_features


def frozen_parameters(selection: dict[str, object]) -> dict[str, object]:
    """Recover the feature contract and fixed tree count from chronological selection."""
    if (selection["feature_columns"] != FEATURE_COLUMNS
            or selection["categorical_features"] != CATEGORICAL_FEATURES
            or selection["validation_start"] != VALIDATION_START
            or selection["test_start"] != TEST_START):
        raise ValueError("The selection report must match the current features and split boundaries.")
    trees = selection["best_iteration"]
    if type(trees) is not int or not 1 <= trees <= selection["model_parameters"]["n_estimators"]:
        raise ValueError("The selected tree count must be a positive integer within the tuning budget.")
    return {**selection["model_parameters"], "n_estimators": trees}


def validate_rows(frame: pd.DataFrame, *, test: bool) -> pd.Series:
    """Require valid targets, unique row indices, and the correct side of the time boundary."""
    dates = pd.to_datetime(frame["target_date"], format="%Y-%m-%d", errors="raise")
    in_period = dates.ge(pd.Timestamp(TEST_START)) if test else dates.lt(pd.Timestamp(TEST_START))
    if frame.empty or not frame.index.is_unique or dates.isna().any() or not in_period.all():
        raise ValueError("Rows must be nonempty, uniquely indexed, and confined to the expected period.")
    values = frame[["previous_total", "target_total", "delta_total"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or not (values[:, :2] > 0).all()
            or not np.allclose(values[:, 1] - values[:, 0], values[:, 2], rtol=0, atol=1e-8)):
        raise ValueError("Totals must be positive and finite, with consistent finite changes.")
    return dates


def fit_final_model(
    refit: pd.DataFrame, selection: dict[str, object],
) -> tuple[lgb.LGBMRegressor, DriftBaselines, dict[str, object]]:
    """Fit the selected tree count and baseline drifts using only pre-test observations."""
    parameters = frozen_parameters(selection)
    dates = validate_rows(refit, test=False)
    features = prepare_features(refit)
    baselines = DriftBaselines.fit(refit[["prior_meets", "delta_total"]])
    model = lgb.LGBMRegressor(**parameters)
    # No evaluation set or early stopping: tree count was fixed before opening test data.
    model.fit(features, refit["delta_total"], categorical_feature=CATEGORICAL_FEATURES)
    return model, baselines, {
        "selected_candidate": selection["selected_candidate"],
        "feature_columns": FEATURE_COLUMNS, "categorical_features": CATEGORICAL_FEATURES,
        "refit_rows": len(refit), "refit_end_exclusive": TEST_START,
        "refit_first_target_date": dates.min().strftime("%Y-%m-%d"),
        "refit_last_target_date": dates.max().strftime("%Y-%m-%d"),
        "refit_parameters": model.get_params(),
        "refit_tree_count": model.booster_.current_iteration(),
        "baseline_parameters": asdict(baselines),
        "refit_category_counts": {
            column: len(features[column].cat.categories) for column in CATEGORICAL_FEATURES
        },
    }


def evaluate_test(
    model: lgb.LGBMRegressor, baselines: DriftBaselines, test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score frozen predictors on every test row without fitting or selecting anything."""
    dates = validate_rows(test, test=True)
    categories = dict(zip(CATEGORICAL_FEATURES, model.booster_.pandas_categorical, strict=True))
    features = prepare_features(test, categories)
    predictions = baselines.predict(test[["previous_total", "prior_meets"]])
    predictions["predicted_delta"] = model.predict(features, validate_features=True)
    predictions["lightgbm"] = test["previous_total"] + predictions["predicted_delta"]
    predictions["source_row"] = test.index
    predictions["target_date"] = test["target_date"]
    predictions["previous_total"] = test["previous_total"]
    predictions["actual_total"] = test["target_total"]
    predictions["actual_delta"] = test["delta_total"]
    return predictions, {
        "test_rows": len(test), "test_start": TEST_START,
        "test_first_target_date": dates.min().strftime("%Y-%m-%d"),
        "test_last_target_date": dates.max().strftime("%Y-%m-%d"),
        "test_unseen_category_rows": {
            column: int((test[column].notna() & features[column].isna()).sum())
            for column in CATEGORICAL_FEATURES
        },
        "test_metrics": score_predictions(predictions),
    }


def file_sha256(path: Path) -> str:
    """Hash a file without loading its contents into memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    """Verify selection inputs, refit before loading test data, and save final artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--selection", type=Path, default=Path("reports/lightgbm_tuning.json"))
    parser.add_argument("--model", type=Path, default=Path("models/lightgbm_final.txt"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/test_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/final_test.json"))
    args = parser.parse_args()
    outputs = [args.model, args.predictions, args.report]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise FileExistsError("Final output paths must be distinct and must not already exist.")

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selection_checksum = file_sha256(args.selection)
    frozen_parameters(selection)
    frames, inputs = load_training_splits(args.data_dir)
    if inputs != selection["inputs"]:
        raise ValueError("Train/validation files differ from the inputs used for model selection.")
    refit = pd.concat([frames["train"], frames["validation"]], ignore_index=True)
    print(f"Refitting {selection['best_iteration']} fixed trees on {len(refit):,} pre-test rows.", flush=True)
    model, baselines, refit_report = fit_final_model(refit, selection)
    del frames, refit

    test_path = args.data_dir / "test.csv"
    inputs["test"] = {"file": test_path.name, "sha256": file_sha256(test_path)}
    columns = ["target_date", "target_total", "delta_total", *FEATURE_COLUMNS]
    dtypes = {
        column: "string" if column in ["target_date", *CATEGORICAL_FEATURES] else "float64"
        for column in columns
    }
    test = pd.read_csv(test_path, usecols=columns, dtype=dtypes, keep_default_na=False, na_values=[""])
    predictions, test_report = evaluate_test(model, baselines, test)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model), num_iteration=-1)
    predictions.to_csv(args.predictions, index=False)
    report = {
        "inputs": inputs, "selection_report": args.selection.name,
        "selection_sha256": selection_checksum, **refit_report, **test_report,
        "target": "delta_total = target_total - previous_total",
        "prediction": "predicted_total = previous_total + predicted_delta",
        "primary_comparison": "MAE reduction versus experience-adjusted drift, chosen on validation.",
        "selection_policy": "Freeze features, hyperparameters, and tree count from chronological validation; refit on train plus validation with no test-driven selection or early stopping.",
        "history_policy": "Predict each meet using information known before that meet; earlier test-period results remain available to later meets. Lifter overlap is allowed.",
        "category_policy": "Levels come from the pre-test refit data only; unseen categories become missing.",
        "metric_policy": "MAE and RMSE are in kilograms; R-squared is computed on reconstructed totals. Each meet has equal weight.",
        "model_file": args.model.name, "model_sha256": file_sha256(args.model),
        "prediction_file": args.predictions.name, "prediction_sha256": file_sha256(args.predictions),
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ["lightgbm", "numpy", "pandas", "scikit-learn"]
        },
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, metrics in report["test_metrics"].items():
        print(f"{name}: MAE {metrics['mae_kg']:.4f} kg; RMSE {metrics['rmse_kg']:.4f} kg", flush=True)
    print(f"Saved final test report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
