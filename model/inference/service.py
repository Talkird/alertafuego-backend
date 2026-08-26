"""Orchestrates a single detection run: fetch latest image, tile, predict, threshold."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from shapely.geometry.base import BaseGeometry

from model.data_pipeline.config import BBox
from model.data_pipeline.ee_client import init_earth_engine
from model.inference.config import InferenceConfig, default_config
from model.inference.land_filter import filter_to_polygon, load_argentina_polygon
from model.inference.predictor import load_model, predict_tiles
from model.inference.raster_fetch import fetch_calibrated_chunks, get_latest_goes_image
from model.inference.tiling import assemble_raster, pixel_to_latlon, tile_raster
from model.inference.visualize import render_detection_thumbnail, save_debug_image, save_true_color_image
from model.training.normalization import BandStats, load_band_stats

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
    threshold: float
    chunk_count: int
    detections: list[tuple[float, float, float, bytes]]  # (lat, lon, probability, thumbnail_png)


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
    ctx: InferenceContext, bbox: BBox, threshold: float, debug_image_dir: Path | None = None
) -> DetectionResult:
    goes_image = get_latest_goes_image()
    image_time_millis = goes_image.get("system:time_start").getInfo()
    image_time = datetime.fromtimestamp(image_time_millis / 1000, tz=timezone.utc).replace(tzinfo=None)

    chunks = fetch_calibrated_chunks(goes_image, bbox, ctx.cfg.chunk_size_px)
    raster = assemble_raster(chunks, bbox)
    tiles, offsets = tile_raster(raster, ctx.cfg.patch_size_px)
    probabilities = predict_tiles(ctx.model, ctx.band_stats, tiles, ctx.device)

    _, height, width = raster.shape
    cropped_height = (height // ctx.cfg.patch_size_px) * ctx.cfg.patch_size_px
    cropped_width = (width // ctx.cfg.patch_size_px) * ctx.cfg.patch_size_px

    detections: list[tuple[float, float, float, int, int]] = []  # lat, lon, probability, row, col
    for (row_offset, col_offset), tile_probs in zip(offsets, probabilities):
        rows, cols = (tile_probs >= threshold).nonzero()
        for row, col in zip(rows, cols):
            abs_row, abs_col = row_offset + int(row), col_offset + int(col)
            probability = float(tile_probs[row, col])
            lat, lon = pixel_to_latlon(abs_row, abs_col, bbox, (cropped_height, cropped_width))
            detections.append((lat, lon, probability, abs_row, abs_col))

    if debug_image_dir is not None:
        pixel_detections = [(row, col, prob) for _, _, prob, row, col in detections]
        save_debug_image(raster, pixel_detections, image_time, threshold, debug_image_dir)
        save_true_color_image(raster, image_time, debug_image_dir)

    detections = filter_to_polygon(detections, ctx.argentina_polygon)
    detections_with_thumbnails = [
        (lat, lon, probability, render_detection_thumbnail(raster, row, col))
        for lat, lon, probability, row, col in detections
    ]

    logger.info("Detection run: %d chunks, %d tiles, %d detections", len(chunks), len(tiles), len(detections))
    return DetectionResult(
        image_time=image_time,
        bbox=bbox,
        threshold=threshold,
        chunk_count=len(chunks),
        detections=detections_with_thumbnails,
    )
