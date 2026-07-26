"""Fetch and calibrate a full-bbox GOES-19 raster, chunked to stay under Earth
Engine's sampleRectangle pixel cap (262,144px/call - see docs/DECISIONS.md)."""

import logging
import math
from datetime import datetime, timedelta, timezone

import ee
import numpy as np

from model.data_pipeline.config import BBox, GOES_COLLECTION_ID, RAW_FILL_VALUE
from model.data_pipeline.patches import (
    apply_quality_mask,
    band_scale_offsets,
    calibrate_raw_bands,
    degrees_per_pixel,
    target_projection,
)

logger = logging.getLogger(__name__)

#: How far back to look for a recent capture. GOES-19 images every 10 minutes;
#: 3 hours gives generous slack for ingestion latency without matching a stale image.
RECENT_WINDOW_HOURS = 3

#: sampleRectangle's hard cap is 262,144px/call - stay safely under it.
MAX_SAMPLE_PIXELS = 262_144


class NoImageAvailableError(Exception):
    """Raised when no GOES-19 capture exists within the recent window."""


def get_latest_goes_image() -> ee.Image:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=RECENT_WINDOW_HOURS)
    recent = ee.ImageCollection(GOES_COLLECTION_ID).filterDate(window_start.isoformat(), now.isoformat())
    if recent.size().getInfo() == 0:
        raise NoImageAvailableError(f"No GOES-19 capture in the last {RECENT_WINDOW_HOURS}h")
    return ee.Image(recent.sort("system:time_start", False).first())


def compute_chunk_grid(bbox: BBox, chunk_size_px: int) -> list[BBox]:
    """Split bbox into a grid of sub-bboxes each ~chunk_size_px x chunk_size_px."""
    lon_deg_per_px, lat_deg_per_px = degrees_per_pixel(target_projection())
    chunk_width_deg = chunk_size_px * lon_deg_per_px
    chunk_height_deg = chunk_size_px * lat_deg_per_px

    n_cols = max(1, math.ceil((bbox.east - bbox.west) / chunk_width_deg))
    n_rows = max(1, math.ceil((bbox.north - bbox.south) / chunk_height_deg))

    chunks = []
    for row in range(n_rows):
        south = bbox.south + row * chunk_height_deg
        north = min(bbox.north, south + chunk_height_deg)
        for col in range(n_cols):
            west = bbox.west + col * chunk_width_deg
            east = min(bbox.east, west + chunk_width_deg)
            chunks.append(BBox(west=west, south=south, east=east, north=north))
    return chunks


def fetch_chunk_raster(
    goes_image: ee.Image, chunk_bbox: BBox, scale_offsets: dict[str, tuple[float, float]]
) -> np.ndarray:
    region = ee.Geometry.BBox(chunk_bbox.west, chunk_bbox.south, chunk_bbox.east, chunk_bbox.north)
    masked = apply_quality_mask(goes_image).reproject(target_projection())
    sampled = masked.sampleRectangle(region=region, defaultValue=RAW_FILL_VALUE)
    properties = sampled.getInfo()["properties"]

    raster = calibrate_raw_bands(properties, scale_offsets)
    n_pixels = raster.shape[1] * raster.shape[2]
    if n_pixels > MAX_SAMPLE_PIXELS:
        logger.warning(
            "Chunk %s returned %d pixels, over the %d cap - chunk_size_px may need lowering",
            chunk_bbox, n_pixels, MAX_SAMPLE_PIXELS,
        )
    return raster


def fetch_calibrated_chunks(
    goes_image: ee.Image, bbox: BBox, chunk_size_px: int
) -> list[tuple[BBox, np.ndarray]]:
    """Fetch bbox as one or more calibrated (16, h, w) chunks, each paired with its
    own geographic sub-bbox so tiling.assemble_raster() can place it correctly."""
    scale_offsets = band_scale_offsets(goes_image)
    chunk_bboxes = compute_chunk_grid(bbox, chunk_size_px)
    logger.info("Fetching %d chunk(s) for bbox %s", len(chunk_bboxes), bbox)
    return [(chunk_bbox, fetch_chunk_raster(goes_image, chunk_bbox, scale_offsets)) for chunk_bbox in chunk_bboxes]
