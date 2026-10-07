"""Partition prediction rows chronologically by target-meet date."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


VALIDATION_START = "2023-01-01"
TEST_START = "2025-01-01"


def chronological_split(
    frame: pd.DataFrame,
    validation_start: str = VALIDATION_START,
    test_start: str = TEST_START,
) -> dict[str, pd.DataFrame]:
    """Split existing histories without resetting information at period boundaries."""
    validation_date, test_date = pd.to_datetime(
        [validation_start, test_start], format="%Y-%m-%d", errors="raise",
    )
    if not validation_date < test_date:
        raise ValueError("Validation must start strictly before the final test period.")
    frame = frame.copy()
    for column in ["target_date", "previous_date"]:
        frame[column] = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="raise")
        if frame[column].isna().any():
            raise ValueError(f"{column} must not contain missing dates.")
    if frame["lifter_name"].fillna("").str.strip().eq("").any():
        raise ValueError("Prediction rows must have a lifter name.")
    if frame.duplicated(["lifter_name", "target_date"]).any():
        raise ValueError("Prediction rows must have unique lifter-target-date keys.")
    if not frame["previous_date"].lt(frame["target_date"]).all():
        raise ValueError("Previous meets must strictly precede their target meets.")

    dates = frame["target_date"]
    masks = {
        "train": dates.lt(validation_date),
        "validation": dates.ge(validation_date) & dates.lt(test_date),
        "test": dates.ge(test_date),
    }
    splits = {name: frame.loc[mask].reset_index(drop=True) for name, mask in masks.items()}
    if any(part.empty for part in splits.values()):
        raise ValueError("Each chronological split must contain prediction rows.")
    return splits


def split_history_file(
    input_path: Path,
    output_dir: Path,
    validation_start: str = VALIDATION_START,
    test_start: str = TEST_START,
) -> dict[str, object]:
    """Save separate period files and audit their dates and counts without scoring targets."""
    paths = {name: output_dir / f"{name}.csv" for name in ["train", "validation", "test"]}
    if input_path.resolve() in [path.resolve() for path in paths.values()]:
        raise ValueError("Split outputs must not replace the input file.")
    with input_path.open("rb") as source:
        input_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    frame = pd.read_csv(
        input_path, dtype="string", keep_default_na=False, na_values=[""], low_memory=False,
    )
    splits = chronological_split(frame, validation_start, test_start)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, part in splits.items():
        output_path = paths[name]
        partial_path = output_path.with_suffix(".csv.part")
        destination = partial_path.open("x", encoding="utf-8", newline="")
        try:
            with destination:
                part.to_csv(destination, index=False, date_format="%Y-%m-%d")
            partial_path.replace(output_path)
        finally:
            partial_path.unlink(missing_ok=True)
        summaries[name] = {
            "file": output_path.name,
            "rows": len(part),
            "lifters": part["lifter_name"].nunique(),
            "first_target_date": part["target_date"].min().strftime("%Y-%m-%d"),
            "last_target_date": part["target_date"].max().strftime("%Y-%m-%d"),
        }
    return {
        "input_file": input_path.name,
        "input_sha256": input_sha256,
        "input_rows": len(frame),
        "split_column": "target_date",
        "validation_start": validation_start,
        "test_start": test_start,
        "history_policy": "Prior results remain available across split boundaries; lifter overlap is allowed.",
        "test_policy": "Reserve test outcomes for final evaluation after model and feature choices are fixed.",
        "splits": summaries,
    }


def main() -> None:
    """Write chronological train, validation, and final test files with a split audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/lifter_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--report", type=Path, default=Path("reports/splits.json"))
    parser.add_argument("--validation-start", default=VALIDATION_START)
    parser.add_argument("--test-start", default=TEST_START)
    args = parser.parse_args()
    report = split_history_file(args.input, args.output_dir, args.validation_start, args.test_start)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, summary in report["splits"].items():
        print(f"Saved {summary['rows']:,} {name} rows: {(args.output_dir / summary['file']).resolve()}")
    print(f"Saved audit: {args.report.resolve()}")


if __name__ == "__main__":
    main()
