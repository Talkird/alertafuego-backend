"""Restrict detections to actual Argentine territory.

`argentina_bbox` is a rectangle - it necessarily includes slivers of Paraguay,
Bolivia, Chile, Brazil and Uruguay near the borders (confirmed in production: a
stored detection landed in Paraguay). Filtering against the real country polygon
fixes this.
"""

import logging

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape

from model.data_pipeline.config import BBox
from model.data_pipeline.negative_sampling import load_argentina_land_geometry

logger = logging.getLogger(__name__)


def load_argentina_polygon(bbox: BBox) -> BaseGeometry:
    """Fetch Argentina's real land boundary once as a shapely geometry, so detections
    can be point-in-polygon tested locally without hitting Earth Engine per request."""
    geojson = load_argentina_land_geometry(bbox).getInfo()
    return shape(geojson)


def filter_to_polygon(
    detections: list[tuple[float, float, float]], polygon: BaseGeometry
) -> list[tuple[float, float, float]]:
    """Keep only (lat, lon, probability) detections whose point falls within polygon."""
    kept = [(lat, lon, prob) for lat, lon, prob in detections if polygon.contains(Point(lon, lat))]
    dropped = len(detections) - len(kept)
    if dropped:
        logger.info("Dropped %d detection(s) outside Argentina's actual border", dropped)
    return kept
