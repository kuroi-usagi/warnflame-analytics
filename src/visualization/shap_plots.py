"""
SHAP explainability plots for the trained Random Forest model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.models.spatial_cv import feature_columns
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = "models/random_forest.joblib"
DEFAULT_MATRIX_PATH = "data/processed/feature_matrix.parquet"
DEFAULT_FIGURES_DIR = "reports/figures"


def _plot_config(config_path: str) -> tuple[int, tuple[float, float], str, int]:
    config = load_config(config_path)
    viz_cfg = config.get("visualization", {})
    plot_cfg = viz_cfg.get("plots", {})
    shap_cfg = viz_cfg.get("shap", {})
    dpi = int(plot_cfg.get("dpi", 300))
    figsize = tuple(plot_cfg.get("figsize", [10, 8]))
    fmt = str(plot_cfg.get("format", "png"))
    n_samples = int(shap_cfg.get("n_samples", 1000))
    return dpi, figsize, fmt, n_samples


def plot_shap_summary(
    model_path: Path,
    matrix_path: Path,
    output_path: Path,
    n_samples: int = 1000,
    random_state: int = 42,
) -> Optional[Path]:
    """
    Beeswarm-style SHAP summary for the top features (bar fallback).
    """
    try:
        import shap
    except ImportError as exc:
        logger.error("SHAP not installed: %s", exc)
        return None

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]
    imputer: Optional[SimpleImputer] = artifact.get("imputer")

    matrix = pd.read_parquet(matrix_path)
    cols = feature_cols or feature_columns(matrix)
    X = matrix[cols]
    n = min(n_samples, len(X))
    X_sample = X.sample(n=n, random_state=random_state)
    if imputer is not None:
        X_imputed = imputer.transform(X_sample)
    else:
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X_sample)

    logger.info("Computing SHAP values for %s samples...", n)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imputed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_imputed,
        feature_names=cols,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary to %s", output_path)
    return output_path


class ShapPlotter:
    """Generate SHAP figures from a saved model artifact."""

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        config = load_config(config_path)
        output_cfg = config.get("output", {})
        train_cfg = config.get("model", {}).get("training", {})

        self.model_path = Path(train_cfg.get("model_path", DEFAULT_MODEL_PATH))
        self.matrix_path = Path(
            train_cfg.get("feature_matrix_path", DEFAULT_MATRIX_PATH)
        )
        self.figures_dir = Path(output_cfg.get("figures_dir", DEFAULT_FIGURES_DIR))
        self.config_path = config_path

    def generate(self, n_samples: Optional[int] = None) -> Optional[Path]:
        dpi, figsize, fmt, default_n = _plot_config(self.config_path)
        n_samples = n_samples or default_n
        output_path = self.figures_dir / f"shap_summary.{fmt}"
        result = plot_shap_summary(
            self.model_path,
            self.matrix_path,
            output_path,
            n_samples=n_samples,
        )
        if result:
            logger.info("SHAP plot dpi=%s figsize=%s", dpi, figsize)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP summary plot")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--matrix", default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    plotter = ShapPlotter(config_path=args.config)
    plotter.model_path = Path(args.model)
    plotter.matrix_path = Path(args.matrix)
    plotter.figures_dir = Path(args.output_dir)
    plotter.generate(n_samples=args.n_samples)


if __name__ == "__main__":
    main()
