"""Unit tests for model/data_pipeline: VIIRS parsing and GOES matching, label masks,
negative sampling and band calibration. Earth Engine is replaced by mocks, so these
check our logic around its responses, not Earth Engine itself."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from model.data_pipeline import labels, matching, negative_sampling
from model.data_pipeline.config import PATCH_FILL_VALUE, RAW_FILL_VALUE, BBox
from model.data_pipeline.matching import FireDetection
from model.data_pipeline.patches import CMI_BANDS, calibrate_raw_bands

BBOX = BBox(west=-70.0, south=-40.0, east=-60.0, north=-30.0)


def _viirs_feature(lat, lon, epoch, confidence=2, frp=12.5):
    return {"geometry": {"coordinates": [lon, lat]}, "properties": {"acq_epoch": epoch, "confidence": confidence, "frp": frp}}


def test_fetch_viirs_detections_parses_both_collections(monkeypatch):
    ee = MagicMock()
    merged = ee.FeatureCollection.return_value.flatten.return_value.limit.return_value
    merged.getInfo.side_effect = [
        {"features": [_viirs_feature(-31.0, -64.0, 1_760_000_000)]},
        {"features": [_viirs_feature(-32.0, -65.0, 1_760_003_600, confidence=1)]},
    ]
    monkeypatch.setattr(matching, "ee", ee)

    detections = matching.fetch_viirs_detections(BBOX, datetime(2025, 10, 1), datetime(2025, 10, 2), 1)

    assert detections == [
        FireDetection(-31.0, -64.0, datetime.fromtimestamp(1_760_000_000, tz=timezone.utc), 2, 12.5),
        FireDetection(-32.0, -65.0, datetime.fromtimestamp(1_760_003_600, tz=timezone.utc), 1, 12.5),
    ]
    collection_ids = [call.args[0] for call in ee.ImageCollection.call_args_list]
    assert collection_ids == list(matching.VIIRS_COLLECTION_IDS)


def test_fetch_viirs_detections_warns_when_query_cap_is_hit(monkeypatch, caplog):
    ee = MagicMock()
    capped = {"features": [_viirs_feature(-31.0, -64.0, 1)] * matching.MAX_VIIRS_FEATURES_PER_QUERY}
    ee.FeatureCollection.return_value.flatten.return_value.limit.return_value.getInfo.return_value = capped
    monkeypatch.setattr(matching, "ee", ee)

    matching.fetch_viirs_detections(BBOX, datetime(2025, 10, 1), datetime(2025, 10, 2), 1)

    assert "query cap" in caplog.text


def test_find_nearest_goes_capture_none_when_window_empty(monkeypatch):
    ee = MagicMock()
    ee.ImageCollection.return_value.filterDate.return_value.size.return_value.getInfo.return_value = 0
    monkeypatch.setattr(matching, "ee", ee)
    assert matching.find_nearest_goes_capture(datetime(2025, 10, 1, tzinfo=timezone.utc), 15) is None


def test_match_detections_to_goes_drops_unmatched(monkeypatch):
    image = MagicMock()
    image.get.return_value.getInfo.return_value = 1_760_000_600_000
    found = FireDetection(-31.0, -64.0, datetime(2025, 10, 1, tzinfo=timezone.utc), 2, 1.0)
    missing = FireDetection(-32.0, -65.0, datetime(2025, 10, 2, tzinfo=timezone.utc), 2, 1.0)
    monkeypatch.setattr(matching, "find_nearest_goes_capture", lambda ts, window: image if ts.day == 1 else None)

    samples = matching.match_detections_to_goes([found, missing], 15)

    assert len(samples) == 1
    assert samples[0].detection == found
    assert samples[0].goes_capture_time == datetime.fromtimestamp(1_760_000_600, tz=timezone.utc)


def test_sample_mask_reads_label_band():
    label_image = MagicMock()
    label_image.sampleRectangle.return_value.getInfo.return_value = {
        "properties": {labels.LABEL_BAND_NAME: [[0, 1], [1, 0]]}
    }
    mask = labels.sample_mask(label_image, region=object())
    np.testing.assert_array_equal(mask, [[0, 1], [1, 0]])
    assert mask.dtype == np.uint8


def test_build_label_image_paints_points_and_reprojects(monkeypatch):
    ee = MagicMock()
    monkeypatch.setattr(labels, "ee", ee)
    projection = object()
    points = [FireDetection(-31.0, -64.0, datetime(2025, 10, 1), 2, 1.0)]

    labels.build_label_image(points, projection)

    ee.Geometry.Point.assert_called_once_with([-64.0, -31.0])  # GeoJSON order: lon, lat
    reduced = ee.FeatureCollection.return_value.reduceToImage.return_value
    reduced.rename.assert_called_once_with(labels.LABEL_BAND_NAME)
    reduced.rename.return_value.unmask.assert_called_once_with(0)
    reduced.rename.return_value.unmask.return_value.reproject.assert_called_once_with(projection)


def test_haversine_known_distance():
    # Buenos Aires - Córdoba, about 646 km.
    assert negative_sampling._haversine_km(-34.6037, -58.3816, -31.4201, -64.1888) == pytest.approx(646, abs=5)
    assert negative_sampling._haversine_km(-31.0, -64.0, -31.0, -64.0) == 0


def test_sample_negative_locations_skips_points_near_fires(monkeypatch):
    ee = MagicMock()
    # Earth Engine returns [lon, lat]; the first candidate is 0 km from the fire.
    ee.FeatureCollection.randomPoints.return_value.geometry.return_value.coordinates.return_value.getInfo.return_value = [
        [-64.0, -31.0],
        [-60.0, -35.0],
        [-68.0, -38.0],
    ]
    monkeypatch.setattr(negative_sampling, "ee", ee)

    accepted = negative_sampling.sample_negative_locations(
        2, land_geometry=object(), positive_locations=[(-31.0, -64.0)], min_distance_km=50
    )

    assert accepted == [(-35.0, -60.0), (-38.0, -68.0)]


def test_sample_negative_locations_gives_up_after_max_attempts(monkeypatch, caplog):
    ee = MagicMock()
    ee.FeatureCollection.randomPoints.return_value.geometry.return_value.coordinates.return_value.getInfo.return_value = [
        [-64.0, -31.0]
    ]
    monkeypatch.setattr(negative_sampling, "ee", ee)

    accepted = negative_sampling.sample_negative_locations(1, object(), [(-31.0, -64.0)], 50)

    assert accepted == []
    assert ee.FeatureCollection.randomPoints.call_count == negative_sampling.MAX_SAMPLING_ATTEMPTS
    assert "Only sampled 0/1" in caplog.text


def test_sample_negative_timestamps_picks_real_capture_times(monkeypatch):
    ee = MagicMock()
    captures = [1_760_000_000_000, 1_760_000_600_000, 1_760_001_200_000]
    ee.ImageCollection.return_value.filterDate.return_value.aggregate_array.return_value.getInfo.return_value = captures
    monkeypatch.setattr(negative_sampling, "ee", ee)

    chosen = negative_sampling.sample_negative_timestamps(datetime(2025, 10, 1), datetime(2025, 10, 2), 2, seed=1)
    more = negative_sampling.sample_negative_timestamps(datetime(2025, 10, 1), datetime(2025, 10, 2), 5, seed=1)

    allowed = {datetime.fromtimestamp(ms / 1000, tz=timezone.utc) for ms in captures}
    assert len(chosen) == 2 and len(set(chosen)) == 2 and set(chosen) <= allowed
    assert len(more) == 5 and set(more) <= allowed  # with replacement when n > captures


def test_sample_negative_timestamps_empty_range(monkeypatch):
    ee = MagicMock()
    ee.ImageCollection.return_value.filterDate.return_value.aggregate_array.return_value.getInfo.return_value = []
    monkeypatch.setattr(negative_sampling, "ee", ee)
    assert negative_sampling.sample_negative_timestamps(datetime(2025, 10, 1), datetime(2025, 10, 2), 3) == []


def test_calibrate_raw_bands_applies_scale_offset_and_fill():
    properties = {band: [[RAW_FILL_VALUE, 100]] for band in CMI_BANDS}
    scale_offsets = {band: (0.5, 10.0) for band in CMI_BANDS}
    patch = calibrate_raw_bands(properties, scale_offsets)
    assert patch.shape == (16, 1, 2)
    assert (patch[:, 0, 0] == PATCH_FILL_VALUE).all()
    assert (patch[:, 0, 1] == 60.0).all()
