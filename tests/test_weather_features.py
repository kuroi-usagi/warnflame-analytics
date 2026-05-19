"""Tests for src.features.weather_features."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

import src.features.weather_features as weather_module
from src.features.weather_features import WeatherFeatureExtractor


@pytest.fixture
def extractor():
    return WeatherFeatureExtractor(
        variables=["tmmx", "pr", "erc"],
        windows=[7, 30],
        cache_dir="cache/test_gridmet",
        required_completeness=0.9,
        config_path="config/pipeline_config.yaml",
    )


@pytest.fixture
def sample_weather():
    dates = pd.date_range("2020-07-01", periods=31, freq="D")
    return pd.DataFrame(
        {
            "tmmx": range(300, 331),
            "pr": [0.0] * 31,
            "erc": [50.0] * 31,
        },
        index=dates,
    )


def test_compute_window_features(extractor, sample_weather):
    ignition = datetime(2020, 7, 31)
    features = extractor.compute_window_features(sample_weather, ignition, window_days=7)

    assert "tmmx_max_7d" in features
    assert "tmmx_mean_7d" in features
    assert "pr_sum_7d" in features
    assert features["tmmx_max_7d"] == sample_weather.iloc[-7:]["tmmx"].max()


def test_extract_weather_for_fire(extractor, sample_weather):
    mock_gridmet = MagicMock()
    mock_gridmet.get_bycoords.return_value = sample_weather

    with patch.object(weather_module, "gridmet", mock_gridmet):
        result = extractor.extract_weather_for_fire(
            -121.0, 38.0, datetime(2020, 7, 31)
        )

    assert result is not None
    assert len(result) == 31
    mock_gridmet.get_bycoords.assert_called_once()


@patch.object(WeatherFeatureExtractor, "extract_weather_for_fire")
def test_extract_features_batch(mock_extract, extractor):
    mock_extract.return_value = pd.DataFrame(
        {"tmmx": [300] * 31, "pr": [0.0] * 31, "erc": [50.0] * 31},
        index=pd.date_range("2020-07-01", periods=31, freq="D"),
    )

    fires = gpd.GeoDataFrame(
        {
            "OBJECTID": [1],
            "ALARM_DATE": [pd.Timestamp("2020-08-01").value // 10**6],
            "centroid_lon": [-121.0],
            "centroid_lat": [38.0],
            "geometry": [Point(-121.0, 38.0)],
        },
        crs="EPSG:4326",
    )

    features = extractor.extract_features_batch(fires, batch_size=1)

    assert len(features) == 1
    assert "erc_max_30d" in features.columns or "erc_max_7d" in features.columns
