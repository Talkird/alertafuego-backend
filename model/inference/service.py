"""Orchestrates a single detection run: fetch latest image, predict, threshold."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import ee
import numpy as np
import torch
from shapely.geometry.base import BaseGeometry

from model.data_pipeline.config import BBox
from model.data_pipeline.ee_client import init_earth_engine
from model.inference.config import InferenceConfig, default_config
from model.inference.land_filter import filter_to_polygon, load_argentina_polygon
from model.inference.predictor import load_model, predict_raster
from model.inference.raster_fetch import fetch_calibrated_chunks, get_latest_goes_image, image_capture_time
from model.inference.tiling import assemble_raster, pixel_to_latlon
from model.inference.visualize import render_detection_thumbnail, save_debug_image, save_true_color_image
from model.training.normalization import BandStats, load_band_stats
from model.training.strata import is_day

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceContext:
    model: torch.nn.Module
    band_stats: BandStats
    device: torch.device
    cfg: InferenceConfig
    argentina_polygon: BaseGeometry


@dataclass(frozen=True)
class DetectionResult:
    image_time: datetime
    bbox: BBox
    #: The lowest threshold applied; with a night threshold each detection carries its own.
    threshold: float
    chunk_count: int
    detections: list[tuple[float, float, float, float, bytes]]  # (lat, lon, probability, threshold, thumbnail_png)


def load_context(cfg: InferenceConfig | None = None) -> InferenceContext:
    """One-time startup routine: Earth Engine auth, normalization stats, model weights."""
    cfg = cfg if cfg is not None else default_config()
    init_earth_engine()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    band_stats = load_band_stats(cfg.checkpoint_dir / "norm_stats.json")
    model = load_model(cfg.checkpoint_dir / "model_best.pt", device)
    argentina_polygon = load_argentina_polygon(cfg.argentina_bbox)
    logger.info("Inference context loaded on device: %s", device)
    return InferenceContext(
        model=model, band_stats=band_stats, device=device, cfg=cfg, argentina_polygon=argentina_polygon
    )


def run_detection(
    ctx: InferenceContext,
    bbox: BBox,
    threshold: float,
    debug_image_dir: Path | None = None,
    goes_image: ee.Image | None = None,
    night_threshold: float | None = None,
) -> DetectionResult:
    """Detects fires in goes_image (default: the latest GOES-19 capture) over bbox,
    at threshold, or night_threshold where the sun is below the horizon if given."""
    if goes_image is None:
        goes_image = get_latest_goes_image()
    image_time = image_capture_time(goes_image)

    chunks = fetch_calibrated_chunks(goes_image, bbox, ctx.cfg.chunk_size_px)
    raster = assemble_raster(chunks, bbox)
    probabilities = predict_raster(ctx.model, ctx.band_stats, raster, ctx.device, tta=ctx.cfg.tta)

    _, height, width = raster.shape
    thresholds = np.full((height, width), threshold, dtype=np.float32)
    if night_threshold is not None:
        lat = bbox.north - (np.arange(height) + 0.5) / height * (bbox.north - bbox.south)
        lon = bbox.west + (np.arange(width) + 0.5) / width * (bbox.east - bbox.west)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        thresholds[~is_day(lat_grid, lon_grid, image_time)] = night_threshold

    detections: list[tuple[float, float, float, float, int, int]] = []  # lat, lon, probability, threshold, row, col
    for row, col in zip(*(probabilities >= thresholds).nonzero()):
        row, col = int(row), int(col)
        lat, lon = pixel_to_latlon(row, col, bbox, (height, width))
        detections.append((lat, lon, float(probabilities[row, col]), float(thresholds[row, col]), row, col))

    if debug_image_dir is not None:
        pixel_detections = [(row, col, prob) for _, _, prob, _, row, col in detections]
        save_debug_image(raster, pixel_detections, image_time, threshold, debug_image_dir)
        save_true_color_image(raster, image_time, debug_image_dir)

    detections = filter_to_polygon(detections, ctx.argentina_polygon)
    detections_with_thumbnails = [
        (lat, lon, probability, pixel_threshold, render_detection_thumbnail(raster, row, col))
        for lat, lon, probability, pixel_threshold, row, col in detections
    ]

    logger.info("Detection run: %d chunks, %d detections", len(chunks), len(detections))
    return DetectionResult(
        image_time=image_time,
        bbox=bbox,
        threshold=min(threshold, night_threshold if night_threshold is not None else threshold),
        chunk_count=len(chunks),
        detections=detections_with_thumbnails,
    )
