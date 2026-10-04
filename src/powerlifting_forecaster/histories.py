"""Build prior-meet predictors and total-change targets, with weights in kilograms."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PREVIOUS_COLUMNS = {
    "Date": "previous_date",
    "TotalKg": "previous_total",
    "Best3SquatKg": "previous_squat",
    "Best3BenchKg": "previous_bench",
    "Best3DeadliftKg": "previous_deadlift",
}
INPUT_COLUMNS = ["Name", *PREVIOUS_COLUMNS]
FEATURE_COLUMNS = [
    "previous_total", "previous_squat", "previous_bench", "previous_deadlift",
    "prior_meets", "days_since_previous_meet", "days_since_first_meet",
]


def build_histories(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Construct forecasts for known meet dates from unique, cleaned lifter-date rows."""
    frame = frame.loc[:, INPUT_COLUMNS].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="raise")
    if frame["Name"].fillna("").str.strip().eq("").any() or frame["Date"].isna().any():
        raise ValueError("Results must have a lifter name and meet date.")
    if frame.duplicated(["Name", "Date"]).any():
        raise ValueError("Resolve duplicate lifter-date records before building histories.")
    for column in list(PREVIOUS_COLUMNS)[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Float64")
    if not frame["TotalKg"].between(0, float("inf"), inclusive="neither").fillna(False).all():
        raise ValueError("Results must have finite positive totals.")

    frame = frame.sort_values(["Name", "Date"]).reset_index(drop=True)
    lifters = frame.groupby("Name", sort=False, observed=True)
    # Shift within each lifter so a target result cannot enter its own predictors.
    previous = lifters[list(PREVIOUS_COLUMNS)].shift(1).rename(columns=PREVIOUS_COLUMNS)
    results = frame[["Name", "Date", "TotalKg"]].rename(columns={
        "Name": "lifter_name", "Date": "target_date", "TotalKg": "target_total",
    }).join(previous)
    results["prior_meets"] = lifters.cumcount()
    results["days_since_previous_meet"] = (frame["Date"] - previous["previous_date"]).dt.days
    results["days_since_first_meet"] = (frame["Date"] - lifters["Date"].transform("first")).dt.days
    results["delta_total"] = results["target_total"] - results["previous_total"]

    # First appearances have no previous total and cannot define a progression target.
    results = results.loc[results["prior_meets"].gt(0)].reset_index(drop=True)
    results["days_since_previous_meet"] = results["days_since_previous_meet"].astype("int64")
    if not results["previous_date"].lt(results["target_date"]).all():
        raise ValueError("Every previous meet must strictly precede its target meet.")
    audit = {
        "input_rows": len(frame),
        "input_lifters": frame["Name"].nunique(),
        "excluded_first_meets": int(lifters.ngroups),
        "output_rows": len(results),
        "output_lifters": results["lifter_name"].nunique(),
        "feature_columns": FEATURE_COLUMNS,
        "target": "delta_total = target_total - previous_total (kg)",
        "date_assumption": "The target meet date is known when forecasting.",
        "history_scope": "Prior retained, valid full-power meets under the same source name.",
    }
    return results, audit


def build_history_file(input_path: Path, output_path: Path) -> dict[str, object]:
    """Build histories across the complete input file and save a provenance audit."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output must not replace the input file.")
    with input_path.open("rb") as source:
        input_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    frame = pd.read_csv(
        input_path, usecols=INPUT_COLUMNS, dtype="string",
        keep_default_na=False, na_values=[""], low_memory=False,
    )
    results, audit = build_histories(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".part")
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
    """Write chronological prediction rows and their construction audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/lifter_meets.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lifter_histories.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/histories.json"))
    args = parser.parse_args()
    report = build_history_file(args.input, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {report['output_rows']:,} prediction rows: {args.output.resolve()}")
    print(f"Saved audit: {args.report.resolve()}")


if __name__ == "__main__":
    main()
