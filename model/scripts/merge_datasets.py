"""CLI entrypoint: combine per-month dataset exports into one training manifest.

Each export lives in its own subfolder of --dataset-root (e.g. model/dataset/2025-04,
model/dataset/2025-09), with its own manifest.csv. This writes
<dataset-root>/manifest.csv listing every sample with paths relative to
--dataset-root, so `train_model --dataset-dir <dataset-root>` trains on all of them.
Sample files are not copied or moved. Each month keeps the train/val/test split it
was exported with, so val and test contain days from every month.

Re-run it after adding or removing a month; the combined manifest is rewritten.

Example:
    python -m model.scripts.merge_datasets
"""

import argparse
import collections
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.csv"
FIELDNAMES = ["filename", "lat", "lon", "goes_time", "has_fire", "split"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("model/dataset"))
    return parser.parse_args()


def _read_month(month_dir: Path, dataset_root: Path) -> list[dict]:
    """Rows of one export's manifest, with filenames rewritten relative to
    dataset_root. Exports made on Windows store backslash paths, normalized here."""
    with (month_dir / MANIFEST_FILENAME).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    missing = 0
    for row in rows:
        path = month_dir / row["filename"].replace("\\", "/")
        if not path.exists():
            missing += 1
        row["filename"] = path.relative_to(dataset_root).as_posix()
    if missing:
        raise FileNotFoundError(f"{month_dir}: {missing} file(s) listed in its manifest are missing on disk")
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    month_dirs = sorted(p.parent for p in args.dataset_root.glob(f"*/{MANIFEST_FILENAME}"))
    if not month_dirs:
        raise SystemExit(f"No exports found: expected {args.dataset_root}/<name>/{MANIFEST_FILENAME}")

    all_rows = []
    for month_dir in month_dirs:
        rows = _read_month(month_dir, args.dataset_root)
        counts = collections.Counter(row["split"] for row in rows)
        logger.info(
            "%s: %d samples (train %d, val %d, test %d)",
            month_dir.name, len(rows), counts["train"], counts["val"], counts["test"],
        )
        all_rows.extend(rows)

    output_path = args.dataset_root / MANIFEST_FILENAME
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    counts = collections.Counter(row["split"] for row in all_rows)
    logger.info(
        "Wrote %s: %d samples from %d export(s) (train %d, val %d, test %d)",
        output_path, len(all_rows), len(month_dirs), counts["train"], counts["val"], counts["test"],
    )


if __name__ == "__main__":
    main()
