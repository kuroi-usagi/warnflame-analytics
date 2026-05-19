"""Tests for src.features.terrain_features."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import src.features.terrain_features as terrain_module
from src.features.terrain_features import (
    TerrainFeatureExtractor,
    aspect_to_south_factor,
    compute_slope_aspect,
)


def test_aspect_to_south_factor():
    assert aspect_to_south_factor(180.0) == pytest.approx(1.0)
    assert aspect_to_south_factor(0.0) == pytest.approx(0.0)


def test_compute_slope_aspect_flat():
    window = np.full((3, 3), 100.0)
    slope, aspect = compute_slope_aspect(window, pixel_size=30.0)
    assert slope == pytest.approx(0.0, abs=1e-6)


def test_sample_terrain_from_mosaic():
    dem = np.arange(9, dtype=float).reshape(3, 3)
    masked = np.ma.array(dem, mask=False)
    extractor = TerrainFeatureExtractor(
        dem_path="cache/test_dem.tif",
        resolution=30,
        mode="mosaic",
    )
    extractor._dem = masked
    extractor._transform = rasterio_affine()
    extractor._crs = "EPSG:5070"

    with patch.object(extractor, "_load_dem"):
        feats = extractor.sample_terrain_from_mosaic(-121.0, 38.0)

    assert "elevation_meters" in feats
    assert "slope_degrees" in feats
    assert "aspect_south_factor" in feats


def rasterio_affine():
    from rasterio.transform import Affine

    return Affine(30.0, 0.0, 0.0, 0.0, -30.0, 0.0)


@patch.object(TerrainFeatureExtractor, "sample_terrain_at_point")
def test_extract_terrain_batch(mock_sample):
    mock_sample.return_value = {
        "elevation_meters": 500.0,
        "slope_degrees": 12.0,
        "aspect_south_factor": 0.8,
    }

    fires = gpd.GeoDataFrame(
        {
            "OBJECTID": [1, 2],
            "centroid_lon": [-121.0, -120.5],
            "centroid_lat": [38.0, 38.5],
            "geometry": [Point(-121.0, 38.0), Point(-120.5, 38.5)],
        },
        crs="EPSG:4326",
    )

    extractor = TerrainFeatureExtractor(mode="patch")
    with patch.object(extractor, "ensure_dem"):
        result = extractor.extract_terrain_batch(fires, batch_size=1, limit=2)

    assert len(result) == 2
    assert set(result.columns) >= {
        "OBJECTID",
        "elevation_meters",
        "slope_degrees",
        "aspect_south_factor",
    }
    assert result.iloc[0]["elevation_meters"] == 500.0


def test_sample_terrain_from_patch():
    mock_ds = MagicMock()
    mock_ds.rio.crs = "EPSG:5070"
    elev = MagicMock()
    elev.values = 450.0
    slope = MagicMock()
    slope.values = 15.0
    aspect = MagicMock()
    aspect.values = 180.0
    mock_ds.sel.return_value = {
        "elevation": elev,
        "slope_degrees": slope,
        "aspect_degrees": aspect,
    }

    mock_py3dep = MagicMock()
    mock_py3dep.get_map.return_value = mock_ds

    extractor = TerrainFeatureExtractor(mode="patch", resolution=30)
    with patch.object(terrain_module, "py3dep", mock_py3dep):
        feats = extractor.sample_terrain_from_patch(-121.0, 38.0)

    assert feats["elevation_meters"] == 450.0
    assert feats["slope_degrees"] == 15.0
    assert feats["aspect_south_factor"] == pytest.approx(1.0)
    mock_py3dep.get_map.assert_called_once()
