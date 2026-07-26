"""CLI entrypoint: build and export the GOES-19/VIIRS training dataset.

Prerequisite (one-time, manual): run `earthengine authenticate` locally and set
EARTH_ENGINE_PROJECT_ID in .env before running this script.

Example:
    python -m model.scripts.export_dataset \\
        --start-date 2024-01-15 --end-date 2024-01-22 \\
        --train-end-date 2024-01-19 --val-end-date 2024-01-20 \\
        --limit 20
"""

import argparse
import logging
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from model.data_pipeline.config import default_config
from model.data_pipeline.dataset_builder import (
    Sample,
    assign_temporal_splits,
    build_negative_samples,
    build_positive_samples,
    save_sample,
    write_manifest,
)
from model.data_pipeline.ee_client import init_earth_engine

logger = logging.getLogger(__name__)

#: Earth Engine caps query results at 5000 elements (see matching.MAX_VIIRS_FEATURES_PER_QUERY).
#: Pulling one day at a time instead of the whole requested range keeps every query
#: well under that cap, even across a full dry season.
CHUNK_DAYS = 1


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    init_earth_engine()

    cfg = replace(
        default_config(output_dir=args.output_dir),
        patch_size_px=args.patch_size,
        negative_to_positive_ratio=args.negative_ratio,
    )

    positives: list[Sample] = []
    negatives: list[Sample] = []
    remaining_limit = args.limit

    for chunk_start, chunk_end in _daterange_chunks(args.start_date, args.end_date):
        if remaining_limit is not None and remaining_limit <= 0:
            break

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

        positives.extend(chunk_positives)
        negatives.extend(chunk_negatives)
        if remaining_limit is not None:
            remaining_limit -= len(chunk_positives) + len(chunk_negatives)

        logger.info(
            "Chunk %s..%s: %d positive, %d negative (running total %d)",
            chunk_start.date(), chunk_end.date(), len(chunk_positives), len(chunk_negatives),
            len(positives) + len(negatives),
        )

    all_samples = positives + negatives
    splits = assign_temporal_splits(all_samples, args.train_end_date.date(), args.val_end_date.date())

    rows = []
    for index, (sample, split) in enumerate(zip(all_samples, splits), start=1):
        path = save_sample(sample, split, cfg.output_dir, index)
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

    write_manifest(rows, cfg.output_dir / "manifest.csv")
    logger.info("Wrote %d samples (%d positive, %d negative) to %s", len(rows), len(positives), len(negatives), cfg.output_dir)


if __name__ == "__main__":
    main()
