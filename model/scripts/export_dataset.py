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
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from model.data_pipeline.config import default_config
from model.data_pipeline.dataset_builder import (
    assign_temporal_splits,
    build_negative_samples,
    build_positive_samples,
    save_sample,
    write_manifest,
)
from model.data_pipeline.ee_client import init_earth_engine

logger = logging.getLogger(__name__)


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

    positives = build_positive_samples(cfg, args.start_date, args.end_date)
    if args.limit is not None:
        positives = positives[: args.limit]

    n_negatives = round(len(positives) * cfg.negative_to_positive_ratio)
    if args.limit is not None:
        n_negatives = min(n_negatives, max(args.limit - len(positives), 0))

    positive_locations = [(sample.lat, sample.lon) for sample in positives]
    negatives = build_negative_samples(cfg, args.start_date, args.end_date, n_negatives, positive_locations)

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
