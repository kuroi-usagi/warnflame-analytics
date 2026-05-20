"""Tests for src.features.vegetation_features."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point

from unittest.mock import patch

from src.features.vegetation_features import (
    VegetationFeatureExtractor,
    _sample_raster,
    ndvi_to_vegetation_density,
)


def _write_ndvi_raster(path: Path, value: float = 0.4) -> None:
    transform = from_bounds(-122.0, 37.0, -121.0, 38.0, 20, 20)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.full((20, 20), value, dtype=np.float32), 1)


def test_ndvi_to_vegetation_density():
    assert ndvi_to_vegetation_density(-1.0) == pytest.approx(0.0)
    assert ndvi_to_vegetation_density(1.0) == pytest.approx(1.0)
    assert ndvi_to_vegetation_density(0.0) == pytest.approx(0.5)


def test_extract_vegetation_from_raster(tmp_path):
    ndvi_path = tmp_path / "ndvi.tif"
    ndmi_path = tmp_path / "ndmi.tif"
    _write_ndvi_raster(ndvi_path, value=0.2)
    _write_ndvi_raster(ndmi_path, value=0.1)

    extractor = VegetationFeatureExtractor(
        ndvi_path=str(ndvi_path),
        ndmi_path=str(ndmi_path),
        allow_fallback=False,
    )
    feats = extractor.extract_vegetation_at_point(-121.5, 37.5)

    assert feats["ndvi_mean"] == pytest.approx(0.2, abs=0.01)
    assert feats["ndmi_mean"] == pytest.approx(0.1, abs=0.01)
    assert feats["vegetation_density"] == pytest.approx(0.6, abs=0.01)


def test_extract_vegetation_fallback(tmp_path):
    extractor = VegetationFeatureExtractor(
        ndvi_path=str(tmp_path / "missing.tif"),
        allow_fallback=True,
    )
    feats = extractor.extract_vegetation_at_point(-121.5, 37.5)
    assert feats["ndvi_mean"] == 0.5
    assert feats["vegetation_density"] == pytest.approx(0.75)


def test_extract_vegetation_batch(tmp_path):
    ndvi_path = tmp_path / "ndvi.tif"
    ndmi_path = tmp_path / "ndmi.tif"
    _write_ndvi_raster(ndvi_path, value=0.0)
    _write_ndvi_raster(ndmi_path, value=0.1)

    fires = gpd.GeoDataFrame(
        {
            "OBJECTID": [1, 2],
            "ALARM_DATE": [1_600_000_000_000, 1_600_000_000_000],
            "centroid_lon": [-121.5, -121.4],
            "centroid_lat": [37.5, 37.6],
            "geometry": [Point(-121.5, 37.5), Point(-121.4, 37.6)],
        },
        crs="EPSG:4326",
    )

    extractor = VegetationFeatureExtractor(
        ndvi_path=str(ndvi_path),
        ndmi_path=str(ndmi_path),
        allow_fallback=False,
    )
    result = extractor.extract_vegetation_batch(fires, batch_size=1)

    assert len(result) == 2
    assert set(result.columns) >= {"OBJECTID", "ndvi_mean", "ndmi_mean", "vegetation_density"}


def test_sample_raster_treats_nan_as_missing(tmp_path):
    """GeoTIFFs without nodata must not return 0.0 for NaN pixels (rasterio quirk)."""
    path = tmp_path / "sparse_ndvi.tif"
    transform = from_bounds(-122.0, 37.0, -121.0, 38.0, 4, 4)
    data = np.full((4, 4), np.nan, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    assert np.isnan(_sample_raster(path, -121.5, 37.5))


def test_extract_nodata_without_fallback_raises(tmp_path):
    path = tmp_path / "sparse_ndvi.tif"
    transform = from_bounds(-122.0, 37.0, -121.0, 38.0, 4, 4)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.full((4, 4), np.nan, dtype=np.float32), 1)

    extractor = VegetationFeatureExtractor(
        ndvi_path=str(path),
        ndmi_path=str(path),
        allow_fallback=False,
    )
    with pytest.raises(ValueError, match="No Sentinel-2 coverage"):
        extractor.extract_vegetation_at_point(-121.5, 37.5)


def test_extract_nodata_with_fallback_uses_defaults(tmp_path):
    path = tmp_path / "sparse_ndvi.tif"
    transform = from_bounds(-122.0, 37.0, -121.0, 38.0, 4, 4)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.full((4, 4), np.nan, dtype=np.float32), 1)

    extractor = VegetationFeatureExtractor(
        ndvi_path=str(path),
        ndmi_path=str(path),
        allow_fallback=True,
    )
    feats = extractor.extract_vegetation_at_point(-121.5, 37.5)
    assert feats["ndvi_mean"] == 0.5
    assert feats["vegetation_density"] == pytest.approx(0.75)


def test_extract_point_fallback_when_raster_nodata(tmp_path):
    path = tmp_path / "sparse_ndvi.tif"
    transform = from_bounds(-122.0, 37.0, -121.0, 38.0, 4, 4)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(np.full((4, 4), -9999, dtype=np.float32), 1)

    extractor = VegetationFeatureExtractor(
        ndvi_path=str(path),
        ndmi_path=str(path),
        allow_fallback=False,
        point_fallback=True,
    )
    with patch(
        "src.features.vegetation_features.fetch_indices_at_point",
        return_value=(0.35, 0.12),
    ):
        feats = extractor.extract_vegetation_at_point(
            -121.5, 37.5, year=2020, alarm_date=None
        )

    assert feats["ndvi_mean"] == pytest.approx(0.35)
    assert feats["ndmi_mean"] == pytest.approx(0.12)


def test_extract_raises_without_fallback(tmp_path):
    extractor = VegetationFeatureExtractor(
        ndvi_path=str(tmp_path / "missing.tif"),
        allow_fallback=False,
    )
    with pytest.raises(FileNotFoundError):
        extractor.extract_vegetation_at_point(-121.5, 37.5)
