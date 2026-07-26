"""Fetch VIIRS fire detections and match each one to the closest GOES-19 capture.

VIIRS detections carry a per-pixel acquisition epoch (`acq_epoch`), which gives
sub-daily precision even though the source collections are published once a day.
That precision is what lets us find a genuinely close-in-time GOES-19 capture,
since GOES-19 images every 10 minutes.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import ee

from model.data_pipeline.config import BBox, GOES_COLLECTION_ID, VIIRS_COLLECTION_IDS

logger = logging.getLogger(__name__)

#: Earth Engine caps FeatureCollection.getInfo() at 5000 elements per query.
MAX_VIIRS_FEATURES_PER_QUERY = 4000


@dataclass(frozen=True)
class FireDetection:
    lat: float
    lon: float
    timestamp: datetime
    confidence: int
    frp: float


@dataclass(frozen=True)
class MatchedSample:
    detection: FireDetection
    goes_image: ee.Image
    goes_capture_time: datetime


def _fire_points_for_image(image: ee.Image, region: ee.Geometry, min_confidence: int) -> ee.FeatureCollection:
    fire_pixels = image.updateMask(image.select("confidence").gte(min_confidence))
    return fire_pixels.select(["confidence", "frp", "acq_epoch"]).sample(
        region=region, scale=375, geometries=True
    )


def fetch_viirs_detections(
    bbox: BBox, start: datetime, end: datetime, min_confidence: int
) -> list[FireDetection]:
    """Query both VIIRS collections for fire pixels within bbox and [start, end)."""
    region = ee.Geometry.BBox(bbox.west, bbox.south, bbox.east, bbox.north)
    detections: list[FireDetection] = []

    for collection_id in VIIRS_COLLECTION_IDS:
        collection = (
            ee.ImageCollection(collection_id)
            .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            .filterBounds(region)
        )
        image_list = collection.toList(collection.size())
        point_collections = image_list.map(
            lambda img: _fire_points_for_image(ee.Image(img), region, min_confidence)
        )
        merged = ee.FeatureCollection(point_collections).flatten().limit(MAX_VIIRS_FEATURES_PER_QUERY)
        features = merged.getInfo()["features"]
        if len(features) == MAX_VIIRS_FEATURES_PER_QUERY:
            logger.warning(
                "%s hit the %d-feature query cap for %s..%s - some detections were dropped, "
                "narrow the date range or bbox for a complete pull",
                collection_id, MAX_VIIRS_FEATURES_PER_QUERY, start.date(), end.date(),
            )

        for feature in features:
            lon, lat = feature["geometry"]["coordinates"]
            props = feature["properties"]
            detections.append(
                FireDetection(
                    lat=lat,
                    lon=lon,
                    timestamp=datetime.fromtimestamp(props["acq_epoch"], tz=timezone.utc),
                    confidence=int(props["confidence"]),
                    frp=float(props.get("frp", 0.0)),
                )
            )
        logger.info("%s: %d fire detections in range", collection_id, len(features))

    return detections


def find_nearest_goes_capture(ts: datetime, window_minutes: int) -> ee.Image | None:
    """Return the GOES-19 MCMIPF capture closest to ts within +/- window_minutes, or None."""
    window_start = (ts - timedelta(minutes=window_minutes)).isoformat()
    window_end = (ts + timedelta(minutes=window_minutes)).isoformat()
    candidates = ee.ImageCollection(GOES_COLLECTION_ID).filterDate(window_start, window_end)

    if candidates.size().getInfo() == 0:
        return None

    target_millis = int(ts.timestamp() * 1000)
    with_delta = candidates.map(
        lambda img: img.set(
            "time_delta", ee.Number(img.get("system:time_start")).subtract(target_millis).abs()
        )
    )
    return ee.Image(with_delta.sort("time_delta").first())


def match_detections_to_goes(
    detections: list[FireDetection], window_minutes: int
) -> list[MatchedSample]:
    """Pair each detection with its closest-in-time GOES-19 capture. Drops unmatched detections."""
    matched: list[MatchedSample] = []
    for detection in detections:
        goes_image = find_nearest_goes_capture(detection.timestamp, window_minutes)
        if goes_image is None:
            continue
        capture_millis = goes_image.get("system:time_start").getInfo()
        capture_time = datetime.fromtimestamp(capture_millis / 1000, tz=timezone.utc)
        matched.append(
            MatchedSample(detection=detection, goes_image=goes_image, goes_capture_time=capture_time)
        )
    logger.info("Matched %d/%d detections to a GOES-19 capture", len(matched), len(detections))
    return matched
