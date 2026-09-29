"""Select valid full-power results and normalize fields before deduplication."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


LIFT_COLUMNS = ["Best3SquatKg", "Best3BenchKg", "Best3DeadliftKg"]
NUMERIC_COLUMNS = ["Age", "BodyweightKg", *LIFT_COLUMNS, "TotalKg"]
RESULT_COLUMNS = [
    "Name", "Date", "Sex", "Event", "Equipment", "Federation", "ParentFederation",
    "MeetName", "MeetCountry", "Division", "Place", "Sanctioned", "Tested",
    *NUMERIC_COLUMNS,
]


def clean_results(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Retain full-power results with finite positive totals and usable history keys."""
    full_power = frame["Event"].eq("SBD").fillna(False)
    results = frame.loc[full_power, RESULT_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        results[column] = pd.to_numeric(results[column], errors="raise").astype("Float64")

    valid_total = results["TotalKg"].between(0, float("inf"), inclusive="neither").fillna(False)
    invalid_total_count = int((~valid_total).sum())
    results = results.loc[valid_total].copy()
    results["Date"] = pd.to_datetime(results["Date"], format="%Y-%m-%d", errors="raise")
    if results["Name"].fillna("").str.strip().eq("").any() or results["Date"].isna().any():
        raise ValueError("Retained results must have a lifter name and meet date.")

    # Nonpositive best lifts do not represent successful attempts.
    nonpositive_lifts = results[LIFT_COLUMNS].le(0)
    results[LIFT_COLUMNS] = results[LIFT_COLUMNS].mask(nonpositive_lifts)
    return results, {
        "input_rows": len(frame),
        "excluded_non_sbd": int((~full_power).sum()),
        "excluded_invalid_total": invalid_total_count,
        "nonpositive_best_lifts_to_missing": int(nonpositive_lifts.sum().sum()),
        "output_rows": len(results),
    }


def clean_archive(archive_path: Path, output_path: Path) -> dict[str, object]:
    """Stream filtered results to an intermediate CSV and return an audit summary."""
    if archive_path.resolve() == output_path.resolve():
        raise ValueError("The output must not replace the source archive.")
    with archive_path.open("rb") as source:
        archive_sha256 = hashlib.file_digest(source, "sha256").hexdigest()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".part")
    counts: Counter[str] = Counter()
    destination = partial_path.open("x", encoding="utf-8", newline="")
    try:
        with destination, ZipFile(archive_path) as archive:
            csv_members = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(csv_members) != 1:
                raise ValueError("Expected exactly one CSV file in the archive.")
            csv_member = csv_members[0]
            with archive.open(csv_member) as source:
                chunks = pd.read_csv(
                    source, usecols=RESULT_COLUMNS, dtype="string", encoding="utf-8",
                    keep_default_na=False, na_values=[""], chunksize=100_000,
                )
                write_header = True
                for chunk in chunks:
                    results, chunk_counts = clean_results(chunk)
                    counts.update(chunk_counts)
                    results.to_csv(destination, index=False, header=write_header, date_format="%Y-%m-%d")
                    write_header = False
        partial_path.replace(output_path)
    finally:
        partial_path.unlink(missing_ok=True)

    return {
        "archive_sha256": archive_sha256,
        "csv_member": csv_member,
        "output_file": output_path.name,
        "duplicates_resolved": False,
        **dict(counts),
    }


def main() -> None:
    """Write filtered results and a separate cleaning audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/openpowerlifting-latest.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/full_power_results.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/initial_cleaning.json"))
    args = parser.parse_args()
    report = clean_archive(args.archive, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Retained {report['output_rows']:,} of {report['input_rows']:,} records.")
    print(f"Saved results: {args.output.resolve()}")
    print(f"Saved audit: {args.report.resolve()}")


if __name__ == "__main__":
    main()
