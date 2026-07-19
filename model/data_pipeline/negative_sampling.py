"""Sample no-fire locations and timestamps to build negative training examples."""

import logging
import random
from datetime import datetime, timezone
from math import atan2, cos, radians, sin, sqrt

import ee

from model.data_pipeline.config import BBox, GOES_COLLECTION_ID

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
MAX_SAMPLING_ATTEMPTS = 10
OVERSAMPLE_FACTOR = 3


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * atan2(sqrt(a), sqrt(1 - a))


def load_argentina_land_geometry(bbox: BBox) -> ee.Geometry:
    """Argentina's land boundary intersected with bbox, so negatives skip the ocean."""
    countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
    argentina = countries.filter(ee.Filter.eq("ADM0_NAME", "Argentina")).geometry()
    bbox_geometry = ee.Geometry.BBox(bbox.west, bbox.south, bbox.east, bbox.north)
    return argentina.intersection(bbox_geometry, ee.ErrorMargin(1))


def sample_negative_locations(
    n: int,
    land_geometry: ee.Geometry,
    positive_locations: list[tuple[float, float]],
    min_distance_km: float,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """Random (lat, lon) points on land, at least min_distance_km from every known fire."""
    accepted: list[tuple[float, float]] = []
    attempt = 0
    while len(accepted) < n and attempt < MAX_SAMPLING_ATTEMPTS:
        n_to_draw = (n - len(accepted)) * OVERSAMPLE_FACTOR
        candidates = ee.FeatureCollection.randomPoints(
            region=land_geometry, points=n_to_draw, seed=seed + attempt
        )
        candidate_coords = candidates.geometry().coordinates().getInfo()
        for lon, lat in candidate_coords:
            if len(accepted) >= n:
                break
            far_from_fires = all(
                _haversine_km(lat, lon, fire_lat, fire_lon) >= min_distance_km
                for fire_lat, fire_lon in positive_locations
            )
            if far_from_fires:
                accepted.append((lat, lon))
        attempt += 1

    if len(accepted) < n:
        logger.warning("Only sampled %d/%d negative locations after %d attempts", len(accepted), n, attempt)
    return accepted


def sample_negative_timestamps(
    start: datetime, end: datetime, n: int, seed: int = 0
) -> list[datetime]:
    """Pick n real GOES-19 capture times within [start, end) to pair with negative locations."""
    collection = ee.ImageCollection(GOES_COLLECTION_ID).filterDate(
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )
    capture_millis = collection.aggregate_array("system:time_start").getInfo()
    if not capture_millis:
        return []

    rng = random.Random(seed)
    chosen = rng.sample(capture_millis, n) if len(capture_millis) >= n else rng.choices(capture_millis, k=n)
    return [datetime.fromtimestamp(millis / 1000, tz=timezone.utc) for millis in chosen]
