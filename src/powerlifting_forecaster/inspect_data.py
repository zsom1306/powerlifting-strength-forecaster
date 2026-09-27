"""Summarize the structure and field coverage of an OpenPowerlifting archive."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


INSPECTION_COLUMNS = [
    "Name", "Date", "Sex", "Event", "Equipment", "Federation", "Tested",
    "Age", "BodyweightKg", "Best3SquatKg", "Best3BenchKg", "Best3DeadliftKg",
    "TotalKg",
]


def inspect_dataset(archive_path: Path, chunk_size: int = 100_000) -> dict[str, object]:
    """Count records, distinct names, categories, and blank fields without filtering."""
    row_count = 0
    lifter_names: set[str] = set()
    blank_counts: Counter[str] = Counter()
    categories = {column: Counter() for column in ("Event", "Equipment")}
    date_bounds: list[pd.Timestamp] = []

    with archive_path.open("rb") as source:
        archive_sha256 = hashlib.file_digest(source, "sha256").hexdigest()

    with ZipFile(archive_path) as archive:
        csv_members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError("Expected exactly one CSV file in the archive.")
        csv_member = csv_members[0]
        with archive.open(csv_member) as source:
            columns = pd.read_csv(source, nrows=0, encoding="utf-8").columns.tolist()

        with archive.open(csv_member) as source:
            chunks = pd.read_csv(
                source,
                usecols=INSPECTION_COLUMNS,
                dtype="string",
                encoding="utf-8",
                keep_default_na=False,
                na_values=[""],
                chunksize=chunk_size,
            )
            for chunk in chunks:
                row_count += len(chunk)
                lifter_names.update(chunk["Name"].dropna().unique())
                blank_counts.update(chunk.isna().sum().to_dict())
                for column, counts in categories.items():
                    counts.update(chunk[column].value_counts().to_dict())

                dates = pd.to_datetime(chunk["Date"], format="%Y-%m-%d", errors="raise")
                dates = dates.dropna()
                if not dates.empty:
                    date_bounds.extend([dates.min(), dates.max()])

    return {
        "archive_sha256": archive_sha256,
        "csv_member": csv_member,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "distinct_lifter_names": len(lifter_names),
        "earliest_date": min(date_bounds).date().isoformat() if date_bounds else None,
        "latest_date": max(date_bounds).date().isoformat() if date_bounds else None,
        "blank_counts": {column: int(blank_counts[column]) for column in INSPECTION_COLUMNS},
        "category_counts": {
            column: {value: int(count) for value, count in sorted(counts.items())}
            for column, counts in categories.items()
        },
    }


def main() -> None:
    """Write an aggregate inspection report for a local dataset archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/openpowerlifting-latest.zip"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_inspection.json"))
    args = parser.parse_args()
    report = inspect_dataset(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Inspected {report['row_count']:,} records across {report['column_count']} columns.")
    print(f"Saved report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
