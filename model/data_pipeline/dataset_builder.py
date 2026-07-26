"""Orchestrate positive/negative sample construction, temporal splitting and export to disk."""

import csv
import logging
import random
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import ee
import numpy as np

from model.data_pipeline.config import PipelineConfig
from model.data_pipeline.labels import build_label_image, sample_mask
from model.data_pipeline.matching import (
    FireDetection,
    fetch_viirs_detections,
    find_nearest_goes_capture,
    match_detections_to_goes,
)
from model.data_pipeline.negative_sampling import (
    load_argentina_land_geometry,
    sample_negative_locations,
    sample_negative_timestamps,
)
from model.data_pipeline.patches import compute_patch_region, extract_patch, target_projection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    patch: np.ndarray
    mask: np.ndarray
    has_fire: bool
    goes_time: datetime
    lat: float
    lon: float


def build_positive_samples(
    cfg: PipelineConfig, start: datetime, end: datetime, limit: int | None = None, seed: int = 0
) -> list[Sample]:
    """One sample per VIIRS detection matched to a GOES-19 capture.

    The detection list is subsampled (randomly, not just truncated - Earth Engine
    returns detections in scan order, so a plain [:n] slice would geographically
    bias the dataset) down to min(limit, cfg.max_detections_per_chunk) before
    matching against GOES-19 - matching does a couple of Earth Engine round-trips
    per detection, so a single high-activity day can otherwise take hours.
    """
    detections = fetch_viirs_detections(cfg.argentina_bbox, start, end, cfg.min_viirs_confidence)
    effective_limit = cfg.max_detections_per_chunk if limit is None else min(limit, cfg.max_detections_per_chunk)
    if len(detections) > effective_limit:
        logger.warning("Subsampling %d detections down to %d for %s..%s", len(detections), effective_limit, start.date(), end.date())
        detections = random.Random(seed).sample(detections, effective_limit)
    matched = match_detections_to_goes(detections, cfg.match_window_minutes)

    # Group by capture time so a patch's mask reflects every fire seen in that GOES
    # image, not just the single detection that happened to center the patch.
    detections_by_capture: dict[datetime, list[FireDetection]] = {}
    for m in matched:
        detections_by_capture.setdefault(m.goes_capture_time, []).append(m.detection)

    projection = target_projection()
    samples: list[Sample] = []
    for m in matched:
        region = compute_patch_region(m.detection.lat, m.detection.lon, cfg.patch_size_px, projection)
        try:
            patch = extract_patch(m.goes_image, region, cfg.patch_size_px)
            label_image = build_label_image(detections_by_capture[m.goes_capture_time], projection)
            mask = sample_mask(label_image, region)
        except ee.EEException as exc:
            logger.warning("Skipping detection at (%.4f, %.4f): %s", m.detection.lat, m.detection.lon, exc)
            continue
        samples.append(
            Sample(
                patch=patch,
                mask=mask,
                has_fire=True,
                goes_time=m.goes_capture_time,
                lat=m.detection.lat,
                lon=m.detection.lon,
            )
        )
    logger.info("Built %d positive samples", len(samples))
    return samples


def build_negative_samples(
    cfg: PipelineConfig,
    start: datetime,
    end: datetime,
    n: int,
    positive_locations: list[tuple[float, float]],
) -> list[Sample]:
    """n no-fire samples, spatially away from known fires and temporally real GOES captures."""
    land_geometry = load_argentina_land_geometry(cfg.argentina_bbox)
    locations = sample_negative_locations(n, land_geometry, positive_locations, cfg.min_negative_distance_km)
    timestamps = sample_negative_timestamps(start, end, len(locations))

    projection = target_projection()
    samples: list[Sample] = []
    for (lat, lon), ts in zip(locations, timestamps):
        goes_image = find_nearest_goes_capture(ts, cfg.match_window_minutes)
        if goes_image is None:
            continue
        region = compute_patch_region(lat, lon, cfg.patch_size_px, projection)
        try:
            patch = extract_patch(goes_image, region, cfg.patch_size_px)
        except ee.EEException as exc:
            logger.warning("Skipping negative at (%.4f, %.4f): %s", lat, lon, exc)
            continue
        mask = np.zeros(patch.shape[1:], dtype=np.uint8)
        samples.append(Sample(patch=patch, mask=mask, has_fire=False, goes_time=ts, lat=lat, lon=lon))
    logger.info("Built %d negative samples", len(samples))
    return samples


def assign_temporal_splits(samples: list[Sample], train_end: date, val_end: date) -> list[str]:
    """Assign each sample to train/val/test by contiguous date ranges (no shuffling)."""
    splits = []
    for sample in samples:
        sample_date = sample.goes_time.date()
        if sample_date <= train_end:
            splits.append("train")
        elif sample_date <= val_end:
            splits.append("val")
        else:
            splits.append("test")
    return splits


def save_sample(sample: Sample, split: str, output_dir: Path, index: int) -> Path:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"sample_{index:06d}.npz"
    np.savez(path, patch=sample.patch, mask=sample.mask)
    return path


def write_manifest(rows: list[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "lat", "lon", "goes_time", "has_fire", "split"]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
