"""Tests for src.models.spatial_cv."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.spatial_cv import (
    SpatialCVAssigner,
    assign_spatial_groups,
    check_spatial_separation,
    feature_columns,
)


@pytest.fixture
def sample_matrix():
    lons = np.linspace(-122.0, -118.0, 20)
    lats = np.linspace(35.0, 39.0, 20)
    return pd.DataFrame(
        {
            "OBJECTID": range(20),
            "centroid_lon": lons,
            "centroid_lat": lats,
            "is_fire": [1] * 10 + [0] * 10,
            "elevation_meters": np.linspace(100, 500, 20),
            "slope_degrees": np.linspace(0, 30, 20),
        }
    )


def test_assign_spatial_groups_labels():
    lon = np.array([-122.0, -121.0, -120.0, -119.0, -118.0, -117.5])
    lat = np.array([35.0, 35.5, 36.0, 37.0, 38.0, 39.0])
    groups = assign_spatial_groups(lon, lat, n_groups=3, random_state=42)
    assert len(groups) == 6
    assert set(groups) <= {0, 1, 2}


def test_check_spatial_separation_passes_for_spread_points():
    lon = np.linspace(-124.0, -114.0, 30)
    lat = np.linspace(33.0, 42.0, 30)
    groups = assign_spatial_groups(lon, lat, n_groups=5, random_state=0)
    report = check_spatial_separation(groups, lon, lat, min_separation_km=1.0)
    assert report["min_centroid_distance_km"] is not None
    assert report["min_centroid_distance_km"] >= 1.0


def test_feature_columns_excludes_metadata(sample_matrix):
    cols = feature_columns(sample_matrix)
    assert "elevation_meters" in cols
    assert "slope_degrees" in cols
    assert "OBJECTID" not in cols
    assert "is_fire" not in cols


def test_spatial_cv_assigner_save(tmp_path, sample_matrix):
    matrix_path = tmp_path / "matrix.parquet"
    sample_matrix.to_parquet(matrix_path, index=False)

    groups_path = tmp_path / "groups.npy"
    results_path = tmp_path / "results.json"

    assigner = SpatialCVAssigner(config_path="config/pipeline_config.yaml")
    assigner.matrix_path = matrix_path
    assigner.groups_path = groups_path
    assigner.results_path = results_path

    groups, report = assigner.run(n_groups=5)
    assigner.save(groups, report)

    assert groups_path.is_file()
    loaded = np.load(groups_path)
    assert len(loaded) == len(sample_matrix)
    with open(results_path, encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["n_records"] == 20
    assert "separation" in saved
