"""Unit tests for model/inference: raster assembly, tiling, pixel geolocation and the
Argentina border filter. No Earth Engine: the projection is stubbed."""

import numpy as np
import pytest
from shapely.geometry import box

from model.data_pipeline.config import PATCH_FILL_VALUE, BBox
from model.inference import tiling
from model.inference.land_filter import filter_to_polygon


@pytest.fixture
def one_degree_per_pixel(monkeypatch):
    """1 degree per pixel, so bbox arithmetic reads as pixel arithmetic."""
    monkeypatch.setattr(tiling, "target_projection", lambda: None)
    monkeypatch.setattr(tiling, "degrees_per_pixel", lambda projection: (1.0, 1.0))


def test_tile_raster_crops_to_multiple_and_returns_offsets():
    raster = np.arange(2 * 70 * 100, dtype=np.float32).reshape(2, 70, 100)
    tiles, offsets = tiling.tile_raster(raster, tile_size=32)
    # 70x100 crops to 64x96: 2 rows x 3 columns of tiles.
    assert tiles.shape == (6, 2, 32, 32)
    assert offsets == [(0, 0), (0, 32), (0, 64), (32, 0), (32, 32), (32, 64)]


def test_tile_raster_tiles_cover_the_centered_crop():
    raster = np.random.default_rng(0).random((1, 40, 40), dtype=np.float32)
    tiles, offsets = tiling.tile_raster(raster, tile_size=32)
    assert offsets == [(0, 0)]
    # (40 - 32) // 2 = 4 pixels cut on each side.
    np.testing.assert_array_equal(tiles[0], raster[:, 4:36, 4:36])


def test_pixel_to_latlon_returns_pixel_centers():
    bbox = BBox(west=-70.0, south=-40.0, east=-60.0, north=-30.0)
    assert tiling.pixel_to_latlon(0, 0, bbox, (10, 10)) == pytest.approx((-30.5, -69.5))
    assert tiling.pixel_to_latlon(9, 9, bbox, (10, 10)) == pytest.approx((-39.5, -60.5))


def test_assemble_raster_places_chunks_by_position_north_up(one_degree_per_pixel):
    bbox = BBox(west=0.0, south=0.0, east=4.0, north=2.0)
    north_west = (BBox(0.0, 1.0, 2.0, 2.0), np.full((1, 1, 2), 1.0, dtype=np.float32))
    south_east = (BBox(2.0, 0.0, 4.0, 1.0), np.full((1, 1, 2), 2.0, dtype=np.float32))

    canvas = tiling.assemble_raster([south_east, north_west], bbox)

    expected = np.array([[[1, 1, PATCH_FILL_VALUE, PATCH_FILL_VALUE], [PATCH_FILL_VALUE, PATCH_FILL_VALUE, 2, 2]]])
    np.testing.assert_array_equal(canvas, expected)


def test_assemble_raster_clips_chunks_overflowing_the_bbox(one_degree_per_pixel):
    bbox = BBox(west=0.0, south=0.0, east=2.0, north=2.0)
    # Chunks fetched from Earth Engine can come back one pixel larger than asked.
    oversized = (BBox(0.0, 0.0, 2.0, 2.0), np.ones((1, 3, 3), dtype=np.float32))
    canvas = tiling.assemble_raster([oversized], bbox)
    assert canvas.shape == (1, 2, 2)
    assert (canvas == 1).all()


def test_filter_to_polygon_keeps_points_inside_and_passes_extra_fields():
    polygon = box(-70, -40, -60, -30)  # lon/lat
    inside = (-35.0, -65.0, 0.9, b"png")
    outside = (-25.0, -57.0, 0.8, b"png")  # Paraguay-ish
    assert filter_to_polygon([inside, outside], polygon) == [inside]


def test_filter_to_polygon_reads_lat_lon_order():
    polygon = box(-70, -40, -60, -30)
    # (lat, lon): swapped coordinates would land outside.
    assert filter_to_polygon([(-35.0, -65.0)], polygon) == [(-35.0, -65.0)]
    assert filter_to_polygon([(-65.0, -35.0)], polygon) == []
