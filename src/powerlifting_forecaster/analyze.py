"""Describe frozen test errors, training progression, and model feature reliance."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from powerlifting_forecaster.baselines import experience_groups
from powerlifting_forecaster.evaluation import MODEL_NAMES, score_predictions
from powerlifting_forecaster.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from powerlifting_forecaster.final_evaluation import file_sha256, validate_rows
from powerlifting_forecaster.splits import VALIDATION_START
from powerlifting_forecaster.train import prepare_features


GAP_BINS = [0, 90, 180, 365, 730, float("inf")]
GAP_LABELS = ["1-90", "91-180", "181-365", "366-730", "731+"]


def gap_groups(days: pd.Series) -> pd.Series:
    """Assign prespecified gap bands in days, including all positive integer gaps."""
    if not (np.isfinite(days) & days.gt(0) & days.mod(1).eq(0)).all():
        raise ValueError("Competition gaps must be finite positive integers.")
    return pd.cut(days, bins=GAP_BINS, labels=GAP_LABELS)


def align_predictions(test: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Reject mismatched prediction rows before attaching diagnostic features."""
    validate_rows(test, test=True)
    if (len(test) != len(predictions)
            or not np.array_equal(predictions["source_row"], test.index)
            or not np.array_equal(predictions["target_date"], test["target_date"])):
        raise ValueError("Predictions must match test row identifiers, order, and dates.")
    for saved, source in [("actual_total", "target_total"), ("actual_delta", "delta_total"),
                          ("previous_total", "previous_total")]:
        if not np.allclose(predictions[saved], test[source], rtol=0, atol=1e-8):
            raise ValueError(f"Saved {saved} does not match the test file.")
    if (not np.isfinite(predictions[[*MODEL_NAMES, "predicted_delta"]].to_numpy()).all()
            or not np.allclose(predictions["lightgbm"],
                               predictions["previous_total"] + predictions["predicted_delta"],
                               rtol=0, atol=1e-8)):
        raise ValueError("Predictions must be finite and consistent with reconstructed totals.")
    result = test[["prior_meets", "days_since_previous_meet"]].copy()
    result["error"] = predictions["lightgbm"].to_numpy() - test["target_total"].to_numpy()
    result["baseline_error"] = predictions["experience_drift"].to_numpy() - test["target_total"].to_numpy()
    return result


def error_summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Summarize absolute errors and signed bias, weighting each meet equally."""
    if frame.empty or not np.isfinite(frame[["error", "baseline_error"]].to_numpy()).all():
        raise ValueError("Error summaries require nonempty finite errors.")
    absolute = frame["error"].abs()
    mae, baseline_mae = float(absolute.mean()), float(frame["baseline_error"].abs().mean())
    return {
        "rows": len(frame), "lightgbm_mae_kg": mae, "baseline_mae_kg": baseline_mae,
        "lightgbm_rmse_kg": float(np.sqrt(frame["error"].pow(2).mean())),
        "bias_kg": float(frame["error"].mean()),
        "median_absolute_error_kg": float(absolute.median()),
        "p90_absolute_error_kg": float(absolute.quantile(0.90)),
        "p95_absolute_error_kg": float(absolute.quantile(0.95)),
        "within_25_kg_pct": float(100 * absolute.le(25).mean()),
        "mae_reduction_vs_baseline_pct": 100 * (1 - mae / baseline_mae) if baseline_mae > 0 else None,
    }


def grouped_errors(frame: pd.DataFrame, groups: pd.Series) -> list[dict[str, object]]:
    """Report every observed band with its denominator; never silently drop missing groups."""
    if not groups.index.equals(frame.index) or groups.isna().any() or frame.empty:
        raise ValueError("Groups must cover every diagnostic row with matching indices.")
    return [
        {"group": str(label), **error_summary(part)}
        for label, part in frame.groupby(groups, observed=True, sort=True)
    ]


def training_progression(train: pd.DataFrame) -> list[dict[str, object]]:
    """Describe changes by prior-meet count using only the original training period."""
    dates = pd.to_datetime(train["target_date"], format="%Y-%m-%d", errors="raise")
    if train.empty or not dates.lt(pd.Timestamp(VALIDATION_START)).all() or not np.isfinite(train["delta_total"]).all():
        raise ValueError("Progression analysis requires finite changes from pre-validation training rows.")
    return [
        {"group": str(label), "rows": len(part), "mean_delta_kg": float(part["delta_total"].mean()),
         "median_delta_kg": float(part["delta_total"].median())}
        for label, part in train.groupby(experience_groups(train["prior_meets"]), observed=True, sort=True)
    ]


def permutation_importance(
    model: lgb.Booster, test: pd.DataFrame, saved_delta: pd.Series,
    *, sample_size: int = 20000, repeats: int = 3, seed: int = 42,
) -> dict[str, object]:
    """Measure delta-MAE increases from individual feature shuffles with the model fixed."""
    if (test.empty or not saved_delta.index.equals(test.index)
            or sample_size < 1 or repeats < 2):
        raise ValueError("Use aligned nonempty inputs, a positive sample size, and at least two repeats.")
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(test), size=min(sample_size, len(test)), replace=False))
    sample = test.iloc[positions]
    categories = dict(zip(CATEGORICAL_FEATURES, model.pandas_categorical, strict=True))
    features = prepare_features(sample, categories)
    predicted = model.predict(features, num_threads=4, validate_features=True)
    if not np.allclose(predicted, saved_delta.iloc[positions], rtol=0, atol=1e-8):
        raise ValueError("The loaded model must reproduce the saved sample predictions.")
    actual = sample["delta_total"].to_numpy()
    original_mae = float(np.abs(actual - predicted).mean())
    orders = [rng.permutation(len(sample)) for _ in range(repeats)]
    importances = []
    for column in FEATURE_COLUMNS:
        increases = []
        for order in orders:
            shuffled = features.copy()
            # Array assignment permutes values without aligning them back by row index.
            shuffled[column] = features[column].iloc[order].array
            delta = model.predict(shuffled, num_threads=4, validate_features=True)
            increases.append(float(np.abs(actual - delta).mean() - original_mae))
        importances.append({"feature": column, "mean_mae_increase_kg": float(np.mean(increases)),
                            "std_mae_increase_kg": float(np.std(increases)), "repeat_increases_kg": increases})
        print(f"Permutation complete: {column}", flush=True)
    return {
        "sample_rows": len(sample), "repeats": repeats, "seed": seed,
        "sampling_policy": "Uniform rows without replacement using NumPy default_rng; sort selected positions, then reuse the same permutations for every feature.",
        "sample_original_mae_kg": original_mae,
        "method": "Shuffle one predictor at a time and measure the increase in delta MAE; the original previous-total reconstruction offset stays fixed.",
        "limitations": "Descriptive model reliance, not causation or additive contributions. Correlated predictors can mask importance; shuffling can create implausible feature combinations. Standard deviations describe repeat variability, not confidence intervals.",
        "features": sorted(importances, key=lambda item: item["mean_mae_increase_kg"], reverse=True),
    }


def main() -> None:
    """Verify frozen artifacts and save descriptive analyses without fitting any models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--evaluation", type=Path, default=Path("reports/final_test.json"))
    parser.add_argument("--model", type=Path, default=Path("models/lightgbm_final.txt"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/test_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/analysis.json"))
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    sources = {
        "train": (args.data_dir / "train.csv", evaluation["inputs"]["train"]["sha256"]),
        "test": (args.data_dir / "test.csv", evaluation["inputs"]["test"]["sha256"]),
        "model": (args.model, evaluation["model_sha256"]),
        "predictions": (args.predictions, evaluation["prediction_sha256"]),
        "evaluation": (args.evaluation, file_sha256(args.evaluation)),
    }
    if args.report.resolve() in [path.resolve() for path, _ in sources.values()]:
        raise ValueError("Analysis output must not replace a source artifact.")
    inputs = {}
    for name, (path, expected) in sources.items():
        checksum = file_sha256(path)
        if checksum != expected:
            raise ValueError(f"The {name} artifact differs from the frozen final evaluation.")
        inputs[name] = {"file": path.name, "sha256": checksum}
    columns = ["target_date", "target_total", "delta_total", *FEATURE_COLUMNS]
    dtypes = {column: "string" if column in ["target_date", *CATEGORICAL_FEATURES] else "float64"
              for column in columns}
    test = pd.read_csv(sources["test"][0], usecols=columns, dtype=dtypes, keep_default_na=False, na_values=[""])
    predictions = pd.read_csv(args.predictions)
    diagnostic = align_predictions(test, predictions)
    metrics = score_predictions(predictions)
    for name, scores in metrics.items():
        for metric, value in scores.items():
            expected = evaluation["test_metrics"][name][metric]
            if value is None and expected is None:
                continue
            if value is None or expected is None or not np.isclose(value, expected, rtol=0, atol=1e-8):
                raise ValueError("Saved predictions do not reproduce the final evaluation metrics.")
    train = pd.read_csv(sources["train"][0], usecols=["target_date", "prior_meets", "delta_total"])
    model = lgb.Booster(model_file=str(args.model))
    importance = permutation_importance(model, test, predictions["predicted_delta"])
    report = {
        "inputs": inputs, "test_rows": len(test),
        "test_first_target_date": evaluation["test_first_target_date"],
        "test_last_target_date": evaluation["test_last_target_date"],
        "analysis_policy": "Post-evaluation descriptive analysis of the frozen model; no feature selection, tuning, or refitting. Each meet has equal weight; repeated lifters are not independent samples.",
        "bias_definition": "Predicted minus actual total; positive bias means overprediction.",
        "group_policy": "Fixed experience bands and right-inclusive gap bands in days; all test rows appear once in each grouping. Associations are descriptive, with no causal adjustment.",
        "gap_labels": GAP_LABELS, "test_metrics": metrics,
        "overall_errors": error_summary(diagnostic),
        "errors_by_experience": grouped_errors(diagnostic, experience_groups(diagnostic["prior_meets"])),
        "errors_by_gap": grouped_errors(diagnostic, gap_groups(diagnostic["days_since_previous_meet"])),
        "training_progression": training_progression(train),
        "training_progression_policy": "Pre-2023 retained repeat-meet observations only; cross-sectional band summaries, not an individual athlete's expected trajectory.",
        "permutation_importance": importance,
        "package_versions": {name: version(name) for name in ["lightgbm", "numpy", "pandas", "scikit-learn"]},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved analysis: {args.report.resolve()}")


if __name__ == "__main__":
    main()
