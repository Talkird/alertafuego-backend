"""Extract fixed-size, quality-masked GOES-19 patches around a lat/lon point."""

import logging

import ee
import numpy as np

from model.data_pipeline.config import GOES_SCALE_METERS, PATCH_FILL_VALUE, RAW_FILL_VALUE

logger = logging.getLogger(__name__)

CMI_BANDS = [f"CMI_C{i:02d}" for i in range(1, 17)]
DQF_BANDS = [f"DQF_C{i:02d}" for i in range(1, 17)]

#: DQF values <= this threshold are kept; 0=good, 1=conditionally usable.
MAX_ACCEPTABLE_DQF = 1


def target_projection() -> ee.Projection:
    """Fixed EPSG:4326 grid at GOES's native scale.

    GOES-19's own default projection is the satellite's geostationary fixed grid,
    whose linear unit is not meters - passing it to buffer()/reproject() as if it
    were produces wildly wrong extents. Using a plain, known CRS+scale instead
    (applied identically to the patch and to the label mask) keeps both aligned
    without depending on that native grid's units.
    """
    return ee.Projection("EPSG:4326").atScale(GOES_SCALE_METERS)


def degrees_per_pixel(projection: ee.Projection) -> tuple[float, float]:
    """(lon_deg_per_px, lat_deg_per_px), read from the projection's own transform
    rather than assumed/derived - the safe way to reason about this API's geometry
    units after getting it wrong from first principles more than once (see
    docs/DECISIONS.md: native projection units, buffer distance units)."""
    transform = projection.getInfo()["transform"]
    return abs(transform[0]), abs(transform[4])


def compute_patch_region(lat: float, lon: float, patch_size_px: int, projection: ee.Projection) -> ee.Geometry:
    """Square region of patch_size_px x patch_size_px pixels centered on (lat, lon).

    projection must carry an embedded scale (see target_projection()) - buffer()'s
    distance argument is then interpreted in that grid's own pixel units, not
    meters, so the radius is passed as pixels rather than pixels * scale_meters.

    Shared with labels.py so the GOES patch and its VIIRS-derived mask are sampled
    over the identical region and end up on the same pixel grid.
    """
    half_extent_px = patch_size_px / 2
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(half_extent_px, proj=projection).bounds(proj=projection)


def apply_quality_mask(goes_image: ee.Image) -> ee.Image:
    masked_bands = [
        goes_image.select(cmi_band).updateMask(goes_image.select(dqf_band).lte(MAX_ACCEPTABLE_DQF))
        for cmi_band, dqf_band in zip(CMI_BANDS, DQF_BANDS)
    ]
    return ee.Image.cat(masked_bands)


def band_scale_offsets(goes_image: ee.Image) -> dict[str, tuple[float, float]]:
    """Per-band linear calibration: physical_value = raw * scale + offset.

    CMI bands are stored as raw scaled integers, not physical units - these
    factors live as image properties (e.g. CMI_C07_scale/CMI_C07_offset).
    """
    keys = [f"{band}_scale" for band in CMI_BANDS] + [f"{band}_offset" for band in CMI_BANDS]
    props = goes_image.toDictionary(keys).getInfo()
    return {band: (props[f"{band}_scale"], props[f"{band}_offset"]) for band in CMI_BANDS}


def calibrate_raw_bands(properties: dict, scale_offsets: dict[str, tuple[float, float]]) -> np.ndarray:
    """Apply per-band physical = raw*scale + offset; RAW_FILL_VALUE positions become
    PATCH_FILL_VALUE. properties is a sampleRectangle().getInfo()["properties"] dict.

    Shared by extract_patch() (single small patch) and inference's chunked raster
    fetch (bigger regions, same calibration pipeline) - kept as one function so the
    calibration logic only exists once.
    """
    band_arrays = []
    for band in CMI_BANDS:
        raw = np.array(properties[band], dtype=np.float32)
        was_masked = raw == RAW_FILL_VALUE
        scale, offset = scale_offsets[band]
        physical = raw * scale + offset
        physical[was_masked] = PATCH_FILL_VALUE
        band_arrays.append(physical)
    return np.stack(band_arrays, axis=0)


def extract_patch(goes_image: ee.Image, region: ee.Geometry, patch_size_px: int) -> np.ndarray:
    """Sample the 16 CMI bands over region as calibrated physical values.

    Masked-out pixels are filled with PATCH_FILL_VALUE. Returns a float32 array of
    shape (16, H, W). H and W should equal patch_size_px but can be off by a pixel
    at the region's edges depending on GOES's grid alignment.
    """
    masked = apply_quality_mask(goes_image).reproject(target_projection())
    sampled = masked.sampleRectangle(region=region, defaultValue=RAW_FILL_VALUE)
    # sampleRectangle attaches each band's pixel window as an image property
    # (a nested list), not as normal pixel data - hence reading via "properties".
    properties = sampled.getInfo()["properties"]
    scale_offsets = band_scale_offsets(goes_image)
    patch = calibrate_raw_bands(properties, scale_offsets)

    if patch.shape[1:] != (patch_size_px, patch_size_px):
        logger.warning(
            "Patch shape %s does not match requested size %d", patch.shape, patch_size_px
        )
    return patch
