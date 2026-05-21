"""Tests for src.models.export_weights."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models.export_weights import (
    RiskWeightExporter,
    RiskWeights,
    aggregate_importance,
    importances_to_raw_weights,
    normalize_weights,
)


@pytest.fixture
def sample_importances():
    cols = [
        "vegetation_density",
        "slope_degrees",
        "aspect_south_factor",
        "elevation_meters",
        "distance_to_roads_km",
        "erc_max_7d",
        "erc_max_30d",
    ]
    values = [0.1, 0.2, 0.05, 0.15, 0.1, 0.25, 0.15]
    return dict(zip(cols, values))


def test_aggregate_importance_max():
    imp = {"erc_max_7d": 0.2, "erc_max_30d": 0.5}
    assert aggregate_importance(imp, ["erc_max_7d", "erc_max_30d", "erc_max_14d"]) == 0.5


def test_normalize_weights_sums_to_one():
    raw = {
        "vegetation_density": 1.0,
        "slope_degrees": 2.0,
        "aspect_south_factor": 1.0,
        "elevation_meters": 1.0,
        "infrastructure_distance_km": 1.0,
    }
    norm = normalize_weights(raw)
    assert sum(norm.values()) == pytest.approx(1.0, abs=0.001)


def test_risk_weights_validation_rejects_bad_sum():
    with pytest.raises(ValidationError):
        RiskWeights(
            vegetation_density=0.5,
            slope_degrees=0.5,
            aspect_south_factor=0.5,
            elevation_meters=0.5,
            infrastructure_distance_km=0.5,
        )


def test_risk_weights_accepts_valid_payload():
    w = RiskWeights(
        vegetation_density=0.2,
        slope_degrees=0.2,
        aspect_south_factor=0.2,
        elevation_meters=0.2,
        infrastructure_distance_km=0.2,
    )
    assert sum(
        [
            w.vegetation_density,
            w.slope_degrees,
            w.aspect_south_factor,
            w.elevation_meters,
            w.infrastructure_distance_km,
        ]
    ) == pytest.approx(1.0)


def test_importances_to_raw_weights_includes_weather(sample_importances):
    raw = importances_to_raw_weights(sample_importances, include_weather=True)
    assert "weather_erc_max" in raw
    assert raw["weather_erc_max"] == pytest.approx(0.25)


def test_exporter_save(tmp_path, sample_importances):
    model_path = tmp_path / "model.joblib"
    output_path = tmp_path / "risk_weights.json"

    exporter = RiskWeightExporter(config_path="config/pipeline_config.yaml")
    exporter.model_path = model_path
    exporter.metrics_path = tmp_path / "missing.json"
    exporter.output_path = output_path

    weights = exporter.export(importances=sample_importances)
    exporter.save(weights)

    assert output_path.is_file()
    with open(output_path, encoding="utf-8") as fh:
        saved = json.load(fh)
    RiskWeights(**saved)
    assert sum(saved.values()) == pytest.approx(1.0, abs=0.01)
