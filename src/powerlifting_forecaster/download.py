"""Download and fingerprint the official OpenPowerlifting bulk archive."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile


DATASET_URL = (
    "https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip"
)


def download_dataset(output_dir: Path) -> Path:
    """Save a verified archive and source metadata without replacing a snapshot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "openpowerlifting-latest.zip"
    metadata_path = archive_path.with_suffix(".json")
    partial_path = archive_path.with_suffix(".zip.part")

    if archive_path.exists() or metadata_path.exists():
        raise FileExistsError("A snapshot already exists; choose a new output directory.")

    digest = hashlib.sha256()
    downloaded_bytes = 0
    partial_file = partial_path.open("xb")
    try:
        with partial_file, urlopen(DATASET_URL, timeout=60) as response:
            last_modified = response.headers.get("Last-Modified")
            expected_bytes = response.headers.get("Content-Length")
            while chunk := response.read(1024 * 1024):
                partial_file.write(chunk)
                digest.update(chunk)
                downloaded_bytes += len(chunk)

        if expected_bytes is not None and downloaded_bytes != int(expected_bytes):
            raise ValueError("Downloaded size does not match the server's Content-Length.")

        with ZipFile(partial_path) as archive:
            if archive.testzip() is not None:
                raise BadZipFile("Downloaded archive failed its integrity check.")

        metadata = {
            "source_url": DATASET_URL,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_modified": last_modified,
            "archive_name": archive_path.name,
            "size_bytes": downloaded_bytes,
            "sha256": digest.hexdigest(),
        }
        partial_path.rename(archive_path)
        with metadata_path.open("x", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
            metadata_file.write("\n")
    finally:
        partial_path.unlink(missing_ok=True)

    return archive_path


def main() -> None:
    """Download the dataset to the directory specified on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory, relative to the current directory by default.",
    )
    args = parser.parse_args()
    archive_path = download_dataset(args.output_dir)
    print(f"Saved archive: {archive_path.resolve()}")
    print(f"Saved metadata: {archive_path.with_suffix('.json').resolve()}")


if __name__ == "__main__":
    main()
