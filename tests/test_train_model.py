"""Tests for src.models.train_model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.train_model import (
    FireRiskTrainer,
    alarm_years,
    build_synthetic_negatives,
)


@pytest.fixture
def sample_matrix():
    n = 40
    lons = np.linspace(-122.0, -118.0, n)
    lats = np.linspace(35.0, 39.0, n)
    years = np.array([2018 + (i % 7) for i in range(n)])
    alarms = [pd.Timestamp(year=int(y), month=7, day=1, tz="UTC").value for y in years]
    return pd.DataFrame(
        {
            "OBJECTID": range(n),
            "centroid_lon": lons,
            "centroid_lat": lats,
            "ALARM_DATE": alarms,
            "is_fire": [1] * 20 + [0] * 20,
            "elevation_meters": np.linspace(100, 800, n),
            "slope_degrees": np.linspace(0, 25, n),
            "erc_max_7d": np.linspace(50, 90, n),
        }
    )


def test_alarm_years_from_epoch_ms():
    ms = int(pd.Timestamp("2023-06-15", tz="UTC").timestamp() * 1000)
    years = alarm_years(pd.Series([ms], dtype=float))
    assert int(years[0]) == 2023


def test_build_synthetic_negatives_count_and_label():
    fires = pd.DataFrame(
        {
            "OBJECTID": [1, 2],
            "centroid_lon": [-120.0, -121.0],
            "centroid_lat": [36.0, 37.0],
            "ALARM_DATE": [1.0, 2.0],
            "is_fire": [1, 1],
            "elevation_meters": [100.0, 200.0],
        }
    )
    neg = build_synthetic_negatives(
        fires,
        ["elevation_meters"],
        n_neg=3,
        california_bbox=(-124.5, 32.5, -114.0, 42.0),
        random_state=0,
    )
    assert len(neg) == 3
    assert (neg["is_fire"] == 0).all()
    assert neg["elevation_meters"].iloc[0] == 150.0


def test_trainer_requires_two_classes_without_synthetic():
    fires_only = pd.DataFrame(
        {
            "OBJECTID": [1, 2],
            "centroid_lon": [-120.0, -121.0],
            "centroid_lat": [36.0, 37.0],
            "ALARM_DATE": [1.0, 2.0],
            "is_fire": [1, 1],
            "elevation_meters": [100.0, 200.0],
        }
    )
    trainer = FireRiskTrainer(config_path="config/pipeline_config.yaml")
    with pytest.raises(ValueError, match="fire \\(1\\) and non-fire"):
        trainer.load_training_table(fires_only, synthetic_negatives=False)


def test_trainer_train_and_save(tmp_path, sample_matrix):
    matrix_path = tmp_path / "matrix.parquet"
    sample_matrix.to_parquet(matrix_path, index=False)

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    trainer = FireRiskTrainer(config_path="config/pipeline_config.yaml")
    trainer.matrix_path = matrix_path
    trainer.model_path = model_path
    trainer.metrics_path = metrics_path
    trainer.n_folds = 3

    metrics, model, imputer, feature_cols = trainer.train()
    trainer.save(model, imputer, feature_cols, metrics)

    assert model_path.is_file()
    assert metrics_path.is_file()
    assert metrics["n_samples"] == 40
    assert len(feature_cols) >= 3
    assert metrics["spatial_cv"]["n_folds"] == 3

    with open(metrics_path, encoding="utf-8") as fh:
        saved = json.load(fh)
    assert "temporal_holdout" in saved
    assert "feature_importances" in saved


def test_trainer_synthetic_negatives_on_fires_only(tmp_path):
    fires_only = pd.DataFrame(
        {
            "OBJECTID": [1, 2, 3, 4],
            "centroid_lon": [-120.0, -121.0, -119.0, -118.5],
            "centroid_lat": [36.0, 37.0, 35.5, 38.0],
            "ALARM_DATE": [
                pd.Timestamp("2020-07-01", tz="UTC").value,
                pd.Timestamp("2021-07-01", tz="UTC").value,
                pd.Timestamp("2023-07-01", tz="UTC").value,
                pd.Timestamp("2024-07-01", tz="UTC").value,
            ],
            "is_fire": [1, 1, 1, 1],
            "elevation_meters": [100.0, 200.0, 300.0, 400.0],
            "slope_degrees": [5.0, 10.0, 15.0, 20.0],
        }
    )
    matrix_path = tmp_path / "matrix.parquet"
    fires_only.to_parquet(matrix_path, index=False)

    trainer = FireRiskTrainer(config_path="config/pipeline_config.yaml")
    trainer.matrix_path = matrix_path
    trainer.groups_path = tmp_path / "missing_groups.npy"
    trainer.n_folds = 2

    metrics, _, _, _ = trainer.train(synthetic_negatives=True)
    assert metrics["n_samples"] == 8
    assert metrics["class_counts"]["fire"] == 4
    assert metrics["class_counts"]["non_fire"] == 4
    assert metrics["synthetic_negatives_used"] is True
