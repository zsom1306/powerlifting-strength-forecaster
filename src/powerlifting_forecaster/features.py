"""Construct historical strength and athlete features for a scheduled competition."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from powerlifting_forecaster.histories import FEATURE_COLUMNS as BASE_FEATURE_COLUMNS
from powerlifting_forecaster.histories import INPUT_COLUMNS as HISTORY_INPUT_COLUMNS
from powerlifting_forecaster.histories import build_histories


CONTEXT_COLUMNS = {"Sex": "sex", "Equipment": "equipment", "Federation": "federation"}
CATEGORICAL_FEATURES = list(CONTEXT_COLUMNS.values())
INPUT_COLUMNS = [*HISTORY_INPUT_COLUMNS, "Age", *CONTEXT_COLUMNS, "Tested"]
FEATURE_COLUMNS = [
    *BASE_FEATURE_COLUMNS, "historical_best_total", "recent_mean_total_3",
    "mean_prior_gap_days", "estimated_age", *CATEGORICAL_FEATURES, "tested_category",
]


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Combine prior-only histories with entry details assumed known before the meet."""
    histories, audit = build_histories(frame)
    context = frame[["Name", "Date", "Age", *CONTEXT_COLUMNS, "Tested"]].copy()
    context["Date"] = pd.to_datetime(context["Date"], format="%Y-%m-%d", errors="raise")
    context = context.sort_values(["Name", "Date"]).reset_index(drop=True)
    age = pd.to_numeric(context["Age"], errors="raise").astype("Float64")
    age = age.where(age.between(0, float("inf"), inclusive="neither"))
    age_history = pd.DataFrame({"age": age, "date": context["Date"].where(age.notna())})
    # Shift before carrying age forward; both operations stay within the same lifter.
    prior_age = age_history.groupby(context["Name"], sort=False).shift(1)
    prior_age = prior_age.groupby(context["Name"], sort=False).ffill()
    context["estimated_age"] = prior_age["age"] + (context["Date"] - prior_age["date"]).dt.days / 365.25
    if not context["Tested"].fillna("").isin(["", "Yes"]).all():
        raise ValueError("Tested must be 'Yes' or blank under the source schema.")
    context["tested_category"] = context["Tested"].eq("Yes").fillna(False).astype("int8")
    context = context.rename(columns={"Name": "lifter_name", "Date": "target_date", **CONTEXT_COLUMNS})
    context[CATEGORICAL_FEATURES] = context[CATEGORICAL_FEATURES].astype("string")
    keys = ["lifter_name", "target_date"]
    results = histories.merge(
        context[[*keys, "estimated_age", *CATEGORICAL_FEATURES, "tested_category"]],
        on=keys, how="left", validate="one_to_one",
    )

    lifters = results.groupby("lifter_name", sort=False, observed=True)
    # Previous totals are already shifted, so the current row is safe to include.
    results["historical_best_total"] = lifters["previous_total"].cummax()
    recent = pd.concat([
        results["previous_total"], lifters["previous_total"].shift(1),
        lifters["previous_total"].shift(2),
    ], axis=1)
    results["recent_mean_total_3"] = recent.mean(axis=1)
    completed_gaps = (results["prior_meets"] - 1).where(results["prior_meets"].gt(1))
    results["mean_prior_gap_days"] = (
        results["days_since_first_meet"] - results["days_since_previous_meet"]
    ) / completed_gaps
    return results, {
        **audit,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "context_assumption": "Target date and sex, equipment, federation, and tested categories are known at entry; results metadata is their proxy.",
        "age_policy": "Latest finite positive age reported strictly before the target meet, plus elapsed days / 365.25; otherwise missing.",
        "recent_mean_policy": "Mean of up to three prior retained totals.",
        "gap_policy": "Mean of completed gaps between prior retained meets; missing with only one prior meet.",
        "tested_policy": "1 for a drug-tested entry category ('Yes'), 0 for blank; not an individual test result.",
    }


def build_feature_file(input_path: Path, output_path: Path) -> dict[str, object]:
    """Write the feature table and record its source checksum and feature policies."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output must not replace the input file.")
    with input_path.open("rb") as source:
        input_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    frame = pd.read_csv(
        input_path, usecols=INPUT_COLUMNS, dtype="string",
        keep_default_na=False, na_values=[""], low_memory=False,
    )
    results, audit = build_features(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".csv.part")
    destination = partial_path.open("x", encoding="utf-8", newline="")
    try:
        with destination:
            results.to_csv(destination, index=False, date_format="%Y-%m-%d")
        partial_path.replace(output_path)
    finally:
        partial_path.unlink(missing_ok=True)
    return {
        "input_file": input_path.name, "input_sha256": input_sha256,
        "output_file": output_path.name, **audit,
    }


def main() -> None:
    """Build features from deduplicated results and save the construction audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/lifter_meets.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lifter_features.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/features.json"))
    args = parser.parse_args()
    report = build_feature_file(args.input, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {report['output_rows']:,} rows with {len(FEATURE_COLUMNS)} features: {args.output.resolve()}")
    print(f"Saved audit: {args.report.resolve()}")


if __name__ == "__main__":
    main()
