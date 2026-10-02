"""Resolve same-day records before constructing chronological lifter histories."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from powerlifting_forecaster.clean_data import LIFT_COLUMNS, NUMERIC_COLUMNS, RESULT_COLUMNS


HISTORY_KEYS = ["Name", "Date"]
CONFLICT_COLUMNS = ["Sex", "Equipment", "MeetCountry", "TotalKg", *LIFT_COLUMNS]


def deduplicate_results(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Exclude conflicting days and select one complete original row per remaining day."""
    frame = frame.loc[:, RESULT_COLUMNS].reset_index(drop=True)
    frame = frame.assign(Date=pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="raise"))
    if frame["Name"].fillna("").str.strip().eq("").any() or frame["Date"].isna().any():
        raise ValueError("Results must have a lifter name and meet date.")
    if not frame["TotalKg"].between(0, float("inf"), inclusive="neither").fillna(False).all():
        raise ValueError("Results must have finite positive totals.")
    if not frame["Event"].eq("SBD").fillna(False).all():
        raise ValueError("Results must contain only full-power events.")

    repeated = frame.duplicated(HISTORY_KEYS, keep=False)
    duplicates = frame.loc[repeated]
    variation = duplicates.groupby(HISTORY_KEYS, observed=True)[CONFLICT_COLUMNS].nunique()
    conflicting = variation.gt(1)
    conflicting_keys = variation.index[conflicting.any(axis=1)]
    excluded = pd.MultiIndex.from_frame(duplicates[HISTORY_KEYS]).isin(conflicting_keys)

    text_columns = [column for column in RESULT_COLUMNS if column not in ["Date", *NUMERIC_COLUMNS]]
    ranking = duplicates.loc[~excluded].astype({column: "string" for column in text_columns})
    ranking["_completeness"] = ranking[RESULT_COLUMNS].notna().sum(axis=1)
    # Sort text lexically so category ordering and input order cannot decide ties.
    selected = ranking.sort_values(
        ["_completeness", *RESULT_COLUMNS],
        ascending=[False, *([True] * len(RESULT_COLUMNS))],
        na_position="last",
    ).drop_duplicates(HISTORY_KEYS).index

    # Select whole rows; aggregating each column separately could invent a result.
    results = pd.concat([frame.loc[~repeated], frame.loc[selected]], ignore_index=True)
    results = results.sort_values(HISTORY_KEYS).reset_index(drop=True)
    audit = {
        "input_rows": len(frame),
        "duplicate_groups": len(variation),
        "rows_in_duplicate_groups": len(duplicates),
        "conflicting_groups": len(conflicting_keys),
        "conflicting_groups_by_column": {
            column: int(conflicting[column].sum()) for column in CONFLICT_COLUMNS
        },
        "excluded_conflicting_rows": int(excluded.sum()),
        "collapsed_duplicate_rows": len(frame) - int(excluded.sum()) - len(results),
        "output_rows": len(results),
        "selection_policy": "Most nonmissing fields, then ascending source-column values; whole rows only.",
        "conflict_policy": "Exclude all rows for days with conflicting nonmissing context or performance values.",
    }
    return results, audit


def deduplicate_file(input_path: Path, output_path: Path) -> dict[str, object]:
    """Resolve duplicates across the complete input file and save sorted results."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output must not replace the input file.")
    with input_path.open("rb") as source:
        input_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    dtypes = {column: "category" for column in RESULT_COLUMNS}
    dtypes.update({column: "Float64" for column in NUMERIC_COLUMNS})
    dtypes.update({"Name": "string", "Date": "string"})
    frame = pd.read_csv(
        input_path, usecols=RESULT_COLUMNS, dtype=dtypes,
        keep_default_na=False, na_values=[""], low_memory=False,
    )
    results, audit = deduplicate_results(frame)

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
        "input_file": input_path.name,
        "input_sha256": input_sha256,
        "output_file": output_path.name,
        **audit,
    }


def main() -> None:
    """Write unique lifter-date results and the duplicate-resolution audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/interim/full_power_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lifter_meets.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/deduplication.json"))
    args = parser.parse_args()
    report = deduplicate_file(args.input, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {report['output_rows']:,} unique lifter-date results: {args.output.resolve()}")
    print(f"Excluded {report['conflicting_groups']:,} conflicting days.")
    print(f"Saved audit: {args.report.resolve()}")


if __name__ == "__main__":
    main()
