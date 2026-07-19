"""Extract fixed-size, quality-masked GOES-19 patches around a lat/lon point."""

import logging

import ee
import numpy as np

from model.data_pipeline.config import GOES_SCALE_METERS, PATCH_FILL_VALUE

logger = logging.getLogger(__name__)

CMI_BANDS = [f"CMI_C{i:02d}" for i in range(1, 17)]
DQF_BANDS = [f"DQF_C{i:02d}" for i in range(1, 17)]

#: DQF values <= this threshold are kept; 0=good, 1=conditionally usable.
MAX_ACCEPTABLE_DQF = 1


def compute_patch_region(lat: float, lon: float, patch_size_px: int, scale_meters: float) -> ee.Geometry:
    """Square region of patch_size_px x patch_size_px pixels centered on (lat, lon).

    Shared with labels.py so the GOES patch and its VIIRS-derived mask are sampled
    over the identical region and end up on the same pixel grid.
    """
    half_extent_m = (patch_size_px / 2) * scale_meters
    return ee.Geometry.Point([lon, lat]).buffer(half_extent_m).bounds()


def native_projection(goes_image: ee.Image) -> ee.Projection:
    """GOES's own pixel grid, needed to reproject the label mask onto the same grid."""
    return goes_image.select(CMI_BANDS[0]).projection()


def _apply_quality_mask(goes_image: ee.Image) -> ee.Image:
    masked_bands = [
        goes_image.select(cmi_band).updateMask(goes_image.select(dqf_band).lte(MAX_ACCEPTABLE_DQF))
        for cmi_band, dqf_band in zip(CMI_BANDS, DQF_BANDS)
    ]
    return ee.Image.cat(masked_bands)


def extract_patch(goes_image: ee.Image, region: ee.Geometry, patch_size_px: int) -> np.ndarray:
    """Sample the 16 CMI bands over region, masked pixels filled with PATCH_FILL_VALUE.

    Returns a float32 array of shape (16, H, W). H and W should equal patch_size_px
    but can be off by a pixel at the region's edges depending on GOES's grid alignment.
    """
    masked = _apply_quality_mask(goes_image)
    sampled = masked.sampleRectangle(region=region, defaultValue=PATCH_FILL_VALUE)
    # sampleRectangle attaches each band's pixel window as an image property
    # (a nested list), not as normal pixel data - hence reading via "properties".
    properties = sampled.getInfo()["properties"]
    band_arrays = [np.array(properties[band], dtype=np.float32) for band in CMI_BANDS]
    patch = np.stack(band_arrays, axis=0)

    if patch.shape[1:] != (patch_size_px, patch_size_px):
        logger.warning(
            "Patch shape %s does not match requested size %d", patch.shape, patch_size_px
        )
    return patch
