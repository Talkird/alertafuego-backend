"""Rasterize VIIRS fire points into a binary mask aligned to a GOES-19 patch's pixel grid."""

import ee
import numpy as np

from model.data_pipeline.matching import FireDetection

LABEL_BAND_NAME = "has_fire"


def build_label_image(fire_points: list[FireDetection], projection: ee.Projection) -> ee.Image:
    """Paint fire_points onto a binary image reprojected onto the GOES pixel grid.

    Reprojecting onto the same `projection` used by the GOES patch (see
    patches.native_projection) is what guarantees sample_mask() below lines up
    pixel-for-pixel with patches.extract_patch()'s output.
    """
    features = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point([d.lon, d.lat]), {LABEL_BAND_NAME: 1}) for d in fire_points]
    )
    # reduceToImage does not reliably name its output band after the input property,
    # so the band is renamed explicitly rather than relied upon implicitly.
    label = features.reduceToImage(properties=[LABEL_BAND_NAME], reducer=ee.Reducer.max()).rename(
        LABEL_BAND_NAME
    )
    return label.unmask(0).reproject(projection)


def sample_mask(label_image: ee.Image, region: ee.Geometry) -> np.ndarray:
    """Sample label_image over the same region passed to patches.extract_patch()."""
    sampled = label_image.sampleRectangle(region=region, defaultValue=0)
    mask = sampled.getInfo()["properties"][LABEL_BAND_NAME]
    return np.array(mask, dtype=np.uint8)
