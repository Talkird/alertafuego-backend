"""Stitch fetched raster chunks into one canvas and map its pixels to coordinates."""

import numpy as np

from model.data_pipeline.config import BBox, PATCH_FILL_VALUE
from model.data_pipeline.patches import degrees_per_pixel, target_projection


def assemble_raster(chunks: list[tuple[BBox, np.ndarray]], bbox: BBox) -> np.ndarray:
    """Paste chunks into one (16, H, W) canvas by absolute geographic position -
    avoids reasoning about grid row/col concatenation order and any north/south
    flip, since each chunk is placed independently from its own known sub-bbox."""
    lon_deg_per_px, lat_deg_per_px = degrees_per_pixel(target_projection())
    n_bands = chunks[0][1].shape[0]
    width_px = round((bbox.east - bbox.west) / lon_deg_per_px)
    height_px = round((bbox.north - bbox.south) / lat_deg_per_px)
    canvas = np.full((n_bands, height_px, width_px), PATCH_FILL_VALUE, dtype=np.float32)

    for chunk_bbox, raster in chunks:
        _, chunk_height, chunk_width = raster.shape
        col_start = round((chunk_bbox.west - bbox.west) / lon_deg_per_px)
        row_start = round((bbox.north - chunk_bbox.north) / lat_deg_per_px)  # rows count from the north edge
        row_end = min(row_start + chunk_height, height_px)
        col_end = min(col_start + chunk_width, width_px)
        canvas[:, row_start:row_end, col_start:col_end] = raster[:, : row_end - row_start, : col_end - col_start]

    return canvas


def pixel_to_latlon(row_px: int, col_px: int, raster_bbox: BBox, raster_shape: tuple[int, int]) -> tuple[float, float]:
    """Linear interpolation from a pixel position in the (uncropped) raster to the
    geographic coordinate of that pixel's center, given the raster's bbox and shape."""
    height, width = raster_shape
    lat = raster_bbox.north - ((row_px + 0.5) / height) * (raster_bbox.north - raster_bbox.south)
    lon = raster_bbox.west + ((col_px + 0.5) / width) * (raster_bbox.east - raster_bbox.west)
    return lat, lon
