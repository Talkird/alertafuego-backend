"""CLI entrypoint: build and export the GOES-19/VIIRS training dataset.

Prerequisite (one-time, manual): run `earthengine authenticate` locally and set
EARTH_ENGINE_PROJECT_ID in .env before running this script.

Each day is written to disk as soon as it's built and recorded in
`completed_days.txt` inside --output-dir. Re-running the same command after a
crash skips the days already done and continues numbering where it left off.

Example:
    python -m model.scripts.export_dataset \\
        --start-date 2024-01-15 --end-date 2024-01-22 \\
        --train-end-date 2024-01-19 --val-end-date 2024-01-20 \\
        --limit 20
"""

import argparse
import csv
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import ee

from model.data_pipeline.config import PipelineConfig, default_config
from model.data_pipeline.dataset_builder import (
    Sample,
    append_manifest,
    assign_temporal_splits,
    build_negative_samples,
    build_positive_samples,
    save_sample,
)
from model.data_pipeline.ee_client import init_earth_engine

logger = logging.getLogger(__name__)

#: Earth Engine caps query results at 5000 elements (see matching.MAX_VIIRS_FEATURES_PER_QUERY).
#: Pulling one day at a time instead of the whole requested range keeps every query
#: well under that cap, even across a full dry season.
CHUNK_DAYS = 1

#: Earth Engine occasionally rejects a single call mid-run (seen: a 403 "Not signed
#: up for Earth Engine" after hundreds of identical successful calls in the same
#: run). A failed day is rebuilt from scratch after a growing delay instead of
#: ending a multi-hour export.
MAX_DAY_ATTEMPTS = 4
RETRY_BASE_DELAY_SECONDS = 60

PROGRESS_FILENAME = "completed_days.txt"
MANIFEST_FILENAME = "manifest.csv"


def _daterange_chunks(start: datetime, end: datetime, days: int = CHUNK_DAYS) -> Iterator[tuple[datetime, datetime]]:
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=days), end)
        yield chunk_start, chunk_end
        chunk_start = chunk_end


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=_parse_date)
    parser.add_argument("--end-date", required=True, type=_parse_date)
    parser.add_argument("--train-end-date", required=True, type=_parse_date)
    parser.add_argument("--val-end-date", required=True, type=_parse_date)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("model/dataset"))
    parser.add_argument("--limit", type=int, default=None, help="Cap total sample count, for smoke tests.")
    return parser.parse_args()


def _load_progress(output_dir: Path) -> tuple[set[date], int, int]:
    """Days already exported to output_dir, the next free sample index, and how
    many samples the manifest already holds."""
    completed_days: set[date] = set()
    progress_path = output_dir / PROGRESS_FILENAME
    if progress_path.exists():
        completed_days = {date.fromisoformat(line) for line in progress_path.read_text().split() if line}

    last_index = 0
    existing_count = 0
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_count += 1
                last_index = max(last_index, int(re.search(r"sample_(\d+)", row["filename"]).group(1)))
    return completed_days, last_index + 1, existing_count


def _mark_day_completed(output_dir: Path, day: date) -> None:
    with (output_dir / PROGRESS_FILENAME).open("a", encoding="utf-8") as f:
        f.write(f"{day.isoformat()}\n")


def _build_day(
    cfg: PipelineConfig, chunk_start: datetime, chunk_end: datetime, remaining_limit: int | None
) -> tuple[list[Sample], list[Sample]]:
    # Split the remaining --limit between positives and negatives up front (by
    # ratio) - otherwise a limit exactly matched by available detections
    # starves negatives of budget.
    positive_limit = None
    if remaining_limit is not None:
        positive_limit = max(1, round(remaining_limit / (1 + cfg.negative_to_positive_ratio)))

    chunk_positives = build_positive_samples(cfg, chunk_start, chunk_end, limit=positive_limit)

    n_negatives = round(len(chunk_positives) * cfg.negative_to_positive_ratio)
    if remaining_limit is not None:
        n_negatives = min(n_negatives, max(remaining_limit - len(chunk_positives), 0))

    positive_locations = [(sample.lat, sample.lon) for sample in chunk_positives]
    chunk_negatives = build_negative_samples(cfg, chunk_start, chunk_end, n_negatives, positive_locations)
    return chunk_positives, chunk_negatives


def _build_day_with_retries(
    cfg: PipelineConfig, chunk_start: datetime, chunk_end: datetime, remaining_limit: int | None
) -> tuple[list[Sample], list[Sample]]:
    for attempt in range(1, MAX_DAY_ATTEMPTS + 1):
        try:
            return _build_day(cfg, chunk_start, chunk_end, remaining_limit)
        except (ee.EEException, OSError) as exc:
            if attempt == MAX_DAY_ATTEMPTS:
                raise
            delay = RETRY_BASE_DELAY_SECONDS * attempt
            logger.warning(
                "Day %s failed (attempt %d/%d), retrying in %ds: %s",
                chunk_start.date(), attempt, MAX_DAY_ATTEMPTS, delay, exc,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    init_earth_engine()

    cfg = replace(
        default_config(output_dir=args.output_dir),
        patch_size_px=args.patch_size,
        negative_to_positive_ratio=args.negative_ratio,
    )
    manifest_path = cfg.output_dir / MANIFEST_FILENAME

    completed_days, next_index, existing_count = _load_progress(cfg.output_dir)
    if completed_days:
        logger.info(
            "Resuming %s: %d day(s) already exported, %d samples in manifest",
            cfg.output_dir, len(completed_days), existing_count,
        )

    remaining_limit = None if args.limit is None else args.limit - existing_count
    written = 0

    for chunk_start, chunk_end in _daterange_chunks(args.start_date, args.end_date):
        if chunk_start.date() in completed_days:
            continue
        if remaining_limit is not None and remaining_limit <= 0:
            break

        chunk_positives, chunk_negatives = _build_day_with_retries(cfg, chunk_start, chunk_end, remaining_limit)
        samples = chunk_positives + chunk_negatives
        splits = assign_temporal_splits(samples, args.train_end_date.date(), args.val_end_date.date())

        rows = []
        for sample, split in zip(samples, splits):
            path = save_sample(sample, split, cfg.output_dir, next_index)
            next_index += 1
            rows.append(
                {
                    "filename": str(path.relative_to(cfg.output_dir)),
                    "lat": sample.lat,
                    "lon": sample.lon,
                    "goes_time": sample.goes_time.isoformat(),
                    "has_fire": sample.has_fire,
                    "split": split,
                }
            )
        append_manifest(rows, manifest_path)
        _mark_day_completed(cfg.output_dir, chunk_start.date())

        written += len(samples)
        if remaining_limit is not None:
            remaining_limit -= len(samples)

        logger.info(
            "Chunk %s..%s: %d positive, %d negative (%d written this run)",
            chunk_start.date(), chunk_end.date(), len(chunk_positives), len(chunk_negatives), written,
        )

    logger.info(
        "Wrote %d samples this run to %s (%d total in manifest)", written, cfg.output_dir, existing_count + written
    )


if __name__ == "__main__":
    main()
