"""Tests for src.visualization.performance_plots."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.visualization.performance_plots import (
    PerformancePlotter,
    plot_risk_weights,
    plot_spatial_cv_scores,
)


@pytest.fixture
def sample_metrics():
    return {
        "spatial_cv": {
            "folds": [
                {"fold": 0, "roc_auc": 0.8, "accuracy": 0.75},
                {"fold": 1, "roc_auc": 0.85, "accuracy": 0.78},
            ],
            "roc_auc_mean": 0.825,
        },
        "feature_importances": {
            "elevation_meters": 0.4,
            "slope_degrees": 0.3,
            "vegetation_density": 0.3,
        },
    }


def test_plot_spatial_cv_scores_writes_file(tmp_path, sample_metrics):
    out = tmp_path / "cv.png"
    with patch("src.visualization.performance_plots.plt.close"):
        result = plot_spatial_cv_scores(sample_metrics, out, figsize=(6, 4))
    assert result == out
    assert out.is_file()


def test_plot_risk_weights_writes_file(tmp_path):
    out = tmp_path / "weights.png"
    weights = {
        "vegetation_density": 0.4,
        "slope_degrees": 0.3,
        "aspect_south_factor": 0.1,
        "elevation_meters": 0.1,
        "infrastructure_distance_km": 0.1,
    }
    with patch("src.visualization.performance_plots.plt.close"):
        plot_risk_weights(weights, out, figsize=(6, 4))
    assert out.is_file()


def test_performance_plotter_generate_all(tmp_path, sample_metrics):
    metrics_path = tmp_path / "training_metrics.json"
    weights_path = tmp_path / "risk_weights.json"
    figures_dir = tmp_path / "figures"

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(sample_metrics, fh)
    with open(weights_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "vegetation_density": 0.4,
                "slope_degrees": 0.2,
                "aspect_south_factor": 0.1,
                "elevation_meters": 0.2,
                "infrastructure_distance_km": 0.1,
            },
            fh,
        )

    plotter = PerformancePlotter(config_path="config/pipeline_config.yaml")
    plotter.metrics_path = metrics_path
    plotter.weights_path = weights_path
    plotter.figures_dir = figures_dir

    with patch("src.visualization.performance_plots.plt.close"):
        saved = plotter.generate_all()

    assert len(saved) >= 2
    assert all(path.is_file() for path in saved)
