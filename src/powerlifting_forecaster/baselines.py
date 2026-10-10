"""Fit training-only drift baselines and evaluate them on chronological validation data."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from powerlifting_forecaster.evaluation import regression_metrics
from powerlifting_forecaster.splits import TEST_START, VALIDATION_START


EXPERIENCE_BINS = [0, 1, 2, 4, 8, 16, float("inf")]
EXPERIENCE_LABELS = ["1", "2", "3-4", "5-8", "9-16", "17+"]


def experience_groups(prior_meets: pd.Series) -> pd.Series:
    """Assign fixed experience bands, pooling the sparse upper tail."""
    values = pd.to_numeric(prior_meets, errors="raise").astype("float64")
    if not (np.isfinite(values) & values.ge(1) & values.mod(1).eq(0)).all():
        raise ValueError("Prior-meet counts must be finite positive integers.")
    return pd.cut(values, bins=EXPERIENCE_BINS, labels=EXPERIENCE_LABELS)


@dataclass(frozen=True)
class DriftBaselines:
    """Fitted mean changes in kilograms and their training sample counts."""

    global_drift: float
    experience_drift: dict[str, float]
    experience_counts: dict[str, int]

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "DriftBaselines":
        """Estimate drift using only the supplied training targets."""
        delta = pd.to_numeric(train["delta_total"], errors="raise").astype("float64")
        if delta.empty or not np.isfinite(delta).all():
            raise ValueError("Training changes must be nonempty and finite.")
        groups = experience_groups(train["prior_meets"])
        summary = delta.groupby(groups, observed=True).agg(["mean", "count"])
        return cls(float(delta.mean()), summary["mean"].to_dict(), summary["count"].to_dict())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict next totals from previous totals and prior-meet counts."""
        previous = pd.to_numeric(features["previous_total"], errors="raise").astype("float64")
        if not (np.isfinite(previous) & previous.gt(0)).all():
            raise ValueError("Previous totals must be finite and positive.")
        groups = experience_groups(features["prior_meets"])
        drift = groups.astype("string").map(self.experience_drift).fillna(self.global_drift)
        return pd.DataFrame({
            "persistence": previous,
            "global_drift": previous + self.global_drift,
            "experience_drift": previous + drift,
        }, index=features.index)


def evaluate_baselines(
    train: pd.DataFrame, validation: pd.DataFrame,
    validation_start: str = VALIDATION_START, test_start: str = TEST_START,
) -> dict[str, object]:
    """Fit on the training period and score every baseline on identical validation rows."""
    validation_date, test_date = pd.to_datetime(
        [validation_start, test_start], format="%Y-%m-%d", errors="raise",
    )
    if not validation_date < test_date <= pd.Timestamp(TEST_START):
        raise ValueError("Validation bounds must be ordered and precede the reserved final test.")
    train_dates = pd.to_datetime(train["target_date"], format="%Y-%m-%d", errors="raise")
    validation_dates = pd.to_datetime(validation["target_date"], format="%Y-%m-%d", errors="raise")
    if train.empty or not train_dates.lt(validation_date).all():
        raise ValueError("Training rows must precede the validation period.")
    if validation.empty or not validation_dates.between(
        validation_date, test_date, inclusive="left",
    ).all():
        raise ValueError("Validation rows must fall within the reserved validation period.")
    for frame in [train, validation]:
        if not np.allclose(frame["target_total"] - frame["previous_total"], frame["delta_total"],
                           rtol=0, atol=1e-8):
            raise ValueError("Changes must equal target total minus previous total.")

    model = DriftBaselines.fit(train[["prior_meets", "delta_total"]])
    predictions = model.predict(validation[["previous_total", "prior_meets"]])
    scores = {name: regression_metrics(validation["target_total"], predictions[name]) for name in predictions}
    for reference in ["persistence", "experience_drift"]:
        reference_mae = scores[reference]["mae_kg"]
        for metrics in scores.values():
            metrics[f"mae_reduction_vs_{reference}_pct"] = (
                100 * (1 - metrics["mae_kg"] / reference_mae) if reference_mae > 0 else None
            )
    return {
        "training_rows": len(train), "validation_rows": len(validation),
        "validation_start": validation_start, "test_start": test_start,
        "experience_bands": EXPERIENCE_LABELS,
        "fitted_parameters": asdict(model),
        "unseen_experience_policy": "Use the training-wide mean change.",
        "validation_metrics": scores,
        "best_validation_baseline": min(scores, key=lambda name: scores[name]["mae_kg"]),
    }


def main() -> None:
    """Read training and validation files and save fitted parameters and validation metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--report", type=Path, default=Path("reports/baseline_validation.json"))
    args = parser.parse_args()
    columns = ["target_date", "previous_total", "prior_meets", "target_total", "delta_total"]
    dtypes = {column: "float64" for column in columns if column != "target_date"}
    frames, inputs = {}, {}
    for name in ["train", "validation"]:
        path = args.data_dir / f"{name}.csv"
        with path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        frames[name] = pd.read_csv(
            path, usecols=columns, dtype={**dtypes, "target_date": "string"},
            keep_default_na=False, na_values=[""],
        )
        inputs[name] = {"file": path.name, "sha256": checksum}
    report = {"inputs": inputs, **evaluate_baselines(frames["train"], frames["validation"])}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, metrics in report["validation_metrics"].items():
        print(f"{name}: MAE {metrics['mae_kg']:.2f} kg; RMSE {metrics['rmse_kg']:.2f} kg")
    print(f"Saved report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
