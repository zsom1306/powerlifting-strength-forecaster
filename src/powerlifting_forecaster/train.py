"""Train a LightGBM change-in-total regressor using chronological validation."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from powerlifting_forecaster.baselines import evaluate_baselines
from powerlifting_forecaster.evaluation import regression_metrics
from powerlifting_forecaster.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS


MODEL_PARAMETERS = {
    "objective": "regression_l1", "metric": "l1",
    "n_estimators": 1000, "learning_rate": 0.05,
    "num_leaves": 31, "min_child_samples": 100, "reg_lambda": 1.0,
    "random_state": 42, "n_jobs": 4,
    "deterministic": True, "force_col_wise": True, "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 50


def prepare_features(
    frame: pd.DataFrame, categories: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Select predictors and encode categories using training-only levels."""
    features = frame[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL_FEATURES:
            values = features[column].astype("string")
            levels = sorted(values.dropna().unique()) if categories is None else categories[column]
            features[column] = pd.Categorical(values.where(values.isin(levels)), categories=levels)
        else:
            values = pd.to_numeric(features[column], errors="raise").astype("float64")
            if np.isinf(values).any():
                raise ValueError(f"Feature {column} contains infinite values.")
            features[column] = values
    return features


def train_regressor(
    train: pd.DataFrame, validation: pd.DataFrame,
) -> tuple[lgb.LGBMRegressor, dict[str, object]]:
    """Fit delta targets and score reconstructed totals against identical baseline rows."""
    # Reuse the chronological and target-consistency checks from baseline evaluation.
    baseline_report = evaluate_baselines(train, validation)
    train_features = prepare_features(train)
    categories = {
        column: train_features[column].cat.categories.tolist() for column in CATEGORICAL_FEATURES
    }
    validation_features = prepare_features(validation, categories)
    model = lgb.LGBMRegressor(**MODEL_PARAMETERS)
    model.fit(
        train_features, train["delta_total"],
        eval_X=validation_features, eval_y=validation["delta_total"],
        eval_names=["validation"], categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)],
    )
    predicted_delta = pd.Series(
        model.predict(validation_features, num_iteration=model.best_iteration_, validate_features=True),
        index=validation.index,
    )
    predicted_total = validation["previous_total"] + predicted_delta
    metrics = regression_metrics(validation["target_total"], predicted_total)
    for reference in ["persistence", "experience_drift"]:
        reference_mae = baseline_report["validation_metrics"][reference]["mae_kg"]
        metrics[f"mae_reduction_vs_{reference}_pct"] = (
            100 * (1 - metrics["mae_kg"] / reference_mae) if reference_mae > 0 else None
        )
    return model, {
        **baseline_report,
        "target": "delta_total = target_total - previous_total",
        "prediction": "predicted_total = previous_total + predicted_delta",
        "feature_columns": FEATURE_COLUMNS, "categorical_features": CATEGORICAL_FEATURES,
        "category_policy": "Levels come from training only; unseen categories become missing.",
        "training_category_counts": {column: len(levels) for column, levels in categories.items()},
        "validation_unseen_category_rows": {
            column: int((validation[column].notna() & validation_features[column].isna()).sum())
            for column in CATEGORICAL_FEATURES
        },
        "model_parameters": model.get_params(),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": model.best_iteration_,
        "evaluated_iterations": len(model.evals_result_["validation"]["l1"]),
        "selection_policy": "Select tree count by validation MAE; final test is not loaded.",
        "validation_metrics": {**baseline_report["validation_metrics"], "lightgbm": metrics},
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ["lightgbm", "numpy", "pandas", "scikit-learn"]
        },
    }


def main() -> None:
    """Read train/validation splits and save a local model plus a reproducible report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--model", type=Path, default=Path("models/lightgbm_delta.txt"))
    parser.add_argument("--report", type=Path, default=Path("reports/lightgbm_validation.json"))
    args = parser.parse_args()
    columns = ["target_date", "target_total", "delta_total", *FEATURE_COLUMNS]
    dtypes = {
        column: "string" if column in ["target_date", *CATEGORICAL_FEATURES] else "float64"
        for column in columns
    }
    frames, inputs = {}, {}
    for name in ["train", "validation"]:
        path = args.data_dir / f"{name}.csv"
        with path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        frames[name] = pd.read_csv(
            path, usecols=columns, dtype=dtypes, keep_default_na=False, na_values=[""],
        )
        inputs[name] = {"file": path.name, "sha256": checksum}
    model, report = train_regressor(frames["train"], frames["validation"])
    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model), num_iteration=model.best_iteration_)
    with args.model.open("rb") as source:
        model_checksum = hashlib.file_digest(source, "sha256").hexdigest()
    report = {
        "inputs": inputs, **report,
        "model_file": args.model.name, "model_sha256": model_checksum,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, metrics in report["validation_metrics"].items():
        print(f"{name}: MAE {metrics['mae_kg']:.2f} kg; RMSE {metrics['rmse_kg']:.2f} kg")
    print(f"Selected {report['best_iteration']} trees; saved model: {args.model.resolve()}")
    print(f"Saved report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
