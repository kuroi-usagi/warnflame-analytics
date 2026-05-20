"""Tests for src.features.build_features."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from src.features.build_features import FeatureMatrixBuilder, _merge_features


@pytest.fixture
def sample_fires(tmp_path):
    fires = gpd.GeoDataFrame(
        {
            "OBJECTID": [1, 2],
            "centroid_lon": [-121.0, -120.5],
            "centroid_lat": [38.0, 38.5],
            "ALARM_DATE": [1_600_000_000_000, 1_600_100_000_000],
            "geometry": [Point(-121.0, 38.0), Point(-120.5, 38.5)],
        },
        crs="EPSG:4326",
    )
    path = tmp_path / "fires.gpkg"
    fires.to_file(path, driver="GPKG")
    return path, fires


def test_merge_features_drops_duplicate_columns():
    base = pd.DataFrame({"OBJECTID": [1], "slope_degrees": [0.1]})
    extra = pd.DataFrame({"OBJECTID": [1], "slope_degrees": [99.0], "elevation_meters": [100.0]})
    merged = _merge_features(base, extra, "terrain")
    assert "elevation_meters" in merged.columns
    assert merged.loc[0, "slope_degrees"] == 0.1


def test_join_fire_features(tmp_path, sample_fires):
    fires_path, fires = sample_fires
    weather = tmp_path / "weather.parquet"
    pd.DataFrame(
        {
            "OBJECTID": [1, 2],
            "erc_max_7d": [80.0, 75.0],
        }
    ).to_parquet(weather, index=False)

    builder = FeatureMatrixBuilder(config_path="config/pipeline_config.yaml")
    builder.fires_path = fires_path
    builder.feature_paths = {
        "weather": weather,
        "terrain": tmp_path / "missing_terrain.parquet",
        "spatial": tmp_path / "missing_spatial.parquet",
        "vegetation": tmp_path / "missing_veg.parquet",
    }
    builder.required_modalities = ["weather"]

    matrix = builder.join_fire_features(fires)
    assert len(matrix) == 2
    assert matrix["is_fire"].tolist() == [1, 1]
    assert "erc_max_7d" in matrix.columns


def test_generate_pseudo_points_respects_min_distance(sample_fires):
    _, fires = sample_fires
    builder = FeatureMatrixBuilder(config_path="config/pipeline_config.yaml")
    builder.min_distance_km = 5.0
    builder.max_placement_attempts = 2000
    builder.random_state = 42

    pseudo = builder.generate_pseudo_points(fires, n_pseudo=3)
    assert len(pseudo) >= 1
    assert (pseudo["OBJECTID"] < 0).all()

    fire_proj = fires.to_crs("EPSG:5070")
    pseudo_proj = pseudo.to_crs("EPSG:5070")
    for _, prow in pseudo_proj.iterrows():
        dists = fire_proj.geometry.distance(prow.geometry)
        assert dists.min() >= 5000.0


def test_build_matrix_skip_pseudo(tmp_path, sample_fires):
    fires_path, fires = sample_fires
    weather = tmp_path / "weather.parquet"
    pd.DataFrame({"OBJECTID": [1, 2], "erc_max_7d": [1.0, 2.0]}).to_parquet(
        weather, index=False
    )

    builder = FeatureMatrixBuilder(config_path="config/pipeline_config.yaml")
    builder.fires_path = fires_path
    builder.feature_paths = {
        "weather": weather,
        "terrain": tmp_path / "t.parquet",
        "spatial": tmp_path / "s.parquet",
        "vegetation": tmp_path / "v.parquet",
    }
    builder.required_modalities = ["weather"]

    matrix = builder.build_matrix(limit=2, skip_pseudo=True)
    assert len(matrix) == 2
    assert matrix["is_fire"].sum() == 2
