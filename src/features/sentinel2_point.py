"""
Fetch Sentinel-2 NDVI/NDMI at a single point via Planetary Computer (on-demand).

Used when statewide composite rasters have nodata at a fire centroid.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import rasterio.errors

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MONTHS = (6, 7, 8, 9)
DEFAULT_CLOUD_THRESHOLD = 20
DEFAULT_MAX_SCENES = 8
DEFAULT_RETRIES = 3
PRE_ALARM_DAYS = 60
POST_ALARM_DAYS = 30
REFLECTANCE_SCALE = 10000.0
# Pixels around centroid (Sentinel-2 10 m → 5 ≈ 50 m footprint)
SAMPLE_WINDOW_PX = 5


def _datetime_range(
    year: Optional[int],
    months: tuple[int, ...],
    alarm_date: Optional[datetime],
) -> str:
    if alarm_date is not None:
        start = (alarm_date - timedelta(days=PRE_ALARM_DAYS)).strftime("%Y-%m-%d")
        end = (alarm_date + timedelta(days=POST_ALARM_DAYS)).strftime("%Y-%m-%d")
        return f"{start}/{end}"
    use_year = year or datetime.now().year
    return f"{use_year}-{min(months):02d}-01/{use_year}-{max(months):02d}-28"


def _dn_to_reflectance(value: float) -> float:
    if value > 2.0:
        return value / REFLECTANCE_SCALE
    return value


def _ndvi_from_bands(red: float, nir: float) -> float:
    red = _dn_to_reflectance(red)
    nir = _dn_to_reflectance(nir)
    denom = nir + red
    if denom == 0 or not np.isfinite(red) or not np.isfinite(nir):
        return float("nan")
    return float((nir - red) / denom)


def _read_band_at_point(href: str, lon: float, lat: float) -> float:
    """Median reflectance in a small window; center pixel is often cloud-masked at fire scars."""
    import rasterio
    from pyproj import Transformer

    with rasterio.open(href) as src:
        if src.crs is None:
            x, y = lon, lat
        else:
            x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(
                lon, lat
            )
        row, col = src.index(x, y)
        half = SAMPLE_WINDOW_PX // 2
        data = src.read(
            1,
            window=((row - half, row + half + 1), (col - half, col + half + 1)),
            masked=True,
        )
        valid = np.asarray(data.compressed(), dtype=np.float64)
        if valid.size == 0:
            return float("nan")
        return float(np.median(valid))


def _ndvi_from_item(item: object, lon: float, lat: float, pc: object) -> float:
    """Read B04/B08 at the point from freshly signed COG URLs (fast, no dask stack)."""
    signed = pc.sign(item)
    red_href = signed.assets["B04"].href
    nir_href = signed.assets["B08"].href
    red = _read_band_at_point(red_href, lon, lat)
    nir = _read_band_at_point(nir_href, lon, lat)
    return _ndvi_from_bands(red, nir)


def _try_scenes_at_point(
    items: list,
    lon: float,
    lat: float,
    pc: object,
    retries: int,
) -> tuple[float, float]:
    for item in items:
        scene_id = getattr(item, "id", "unknown")
        for attempt in range(1, retries + 1):
            try:
                ndvi_val = _ndvi_from_item(item, lon, lat, pc)
                if np.isfinite(ndvi_val):
                    logger.info(
                        "Point NDVI %.3f at (%.4f, %.4f) from scene %s",
                        ndvi_val,
                        lon,
                        lat,
                        scene_id,
                    )
                    return ndvi_val, float("nan")
            except Exception as exc:
                err = str(exc)
                if attempt == retries:
                    logger.warning(
                        "Scene %s failed (%s attempts): %s",
                        scene_id,
                        retries,
                        err[:200],
                    )
                elif "403" in err or "401" in err:
                    time.sleep(2.0 * attempt)
                else:
                    time.sleep(1.0 * attempt)
    return float("nan"), float("nan")


def fetch_indices_at_point(
    lon: float,
    lat: float,
    year: Optional[int] = None,
    alarm_date: Optional[datetime] = None,
    months: tuple[int, ...] = DEFAULT_MONTHS,
    cloud_threshold: int = DEFAULT_CLOUD_THRESHOLD,
    max_scenes: int = DEFAULT_MAX_SCENES,
    buffer_deg: float = 0.02,  # kept for API compatibility
    resolution_m: int = 20,
    retries: int = DEFAULT_RETRIES,
) -> tuple[float, float]:
    """
    NDVI at (lon, lat) from low-cloud Sentinel-2 L2A scenes.

    Prefers scenes in a window around ``alarm_date``; falls back to ``year`` + ``months``.
    Returns (nan, nan) when no scenes are found or all reads fail.
    """
    del buffer_deg, resolution_m  # direct COG sample does not stack a bbox

    try:
        import planetary_computer as pc
        import pystac_client
    except ImportError as exc:
        raise ImportError(
            "Point Sentinel-2 fetch requires planetary-computer and pystac-client. "
            "Install with: pip install -r requirements-sentinel2.txt"
        ) from exc

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    datetime_range = _datetime_range(year, months, alarm_date)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": cloud_threshold}},
    )
    items = sorted(
        search.items(),
        key=lambda item: item.properties.get("eo:cloud_cover", 100),
    )[:max_scenes]
    if not items:
        logger.warning(
            "No Sentinel-2 scenes for (%.4f, %.4f) in %s (cloud<%s)",
            lon,
            lat,
            datetime_range,
            cloud_threshold,
        )
        return float("nan"), float("nan")

    logger.info(
        "Point fetch: %s scenes for (%.4f, %.4f), range %s",
        len(items),
        lon,
        lat,
        datetime_range,
    )
    return _try_scenes_at_point(items, lon, lat, pc, retries)
