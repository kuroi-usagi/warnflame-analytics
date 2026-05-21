"""
Generate model performance figures from training metrics and exported weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_METRICS_PATH = "models/training_metrics.json"
DEFAULT_WEIGHTS_PATH = "models/risk_weights.json"
DEFAULT_FIGURES_DIR = "reports/figures"


def _plot_config(config_path: str) -> tuple[int, tuple[float, float], str]:
    config = load_config(config_path)
    plot_cfg = config.get("visualization", {}).get("plots", {})
    dpi = int(plot_cfg.get("dpi", 300))
    figsize = tuple(plot_cfg.get("figsize", [10, 8]))
    fmt = str(plot_cfg.get("format", "png"))
    return dpi, figsize, fmt


def plot_spatial_cv_scores(
    metrics: dict[str, Any],
    output_path: Path,
    figsize: tuple[float, float],
) -> Optional[Path]:
    folds = metrics.get("spatial_cv", {}).get("folds", [])
    if not folds:
        logger.warning("No spatial CV folds in metrics; skipping CV plot")
        return None

    folds = sorted(folds, key=lambda row: row["fold"])
    labels = [f"Fold {row['fold']}" for row in folds]
    aucs = [row.get("roc_auc") for row in folds]
    accs = [row.get("accuracy", 0) for row in folds]

    fig, ax1 = plt.subplots(figsize=figsize)
    x = range(len(labels))
    ax1.bar(x, [a if a is not None else 0 for a in aucs], color="#2c7fb8", label="ROC-AUC")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("ROC-AUC")
    ax1.set_xticks(list(x), labels, rotation=15)

    ax2 = ax1.twinx()
    ax2.plot(x, accs, color="#e6550d", marker="o", label="Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Accuracy")

    mean_auc = metrics.get("spatial_cv", {}).get("roc_auc_mean")
    title = "Spatial GroupKFold performance"
    if mean_auc is not None:
        title += f" (mean ROC-AUC={mean_auc:.3f})"
    ax1.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


def plot_risk_weights(
    weights: dict[str, float],
    output_path: Path,
    figsize: tuple[float, float],
) -> Path:
    keys = list(weights.keys())
    values = [weights[k] for k in keys]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(keys, values, color="#41ab5d")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Normalized weight")
    ax.set_title("Warnflame risk_weights.json")
    for idx, val in enumerate(values):
        ax.text(val + 0.01, idx, f"{val:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


def plot_top_importances(
    importances: dict[str, float],
    output_path: Path,
    figsize: tuple[float, float],
    top_n: int = 15,
) -> Path:
    ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:top_n]
    names = [name for name, _ in ranked][::-1]
    values = [value for _, value in ranked][::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(names, values, color="#756bb1")
    ax.set_xlabel("Random Forest importance")
    ax.set_title(f"Top {top_n} feature importances")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


class PerformancePlotter:
    """Build standard performance figures for reports."""

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        config = load_config(config_path)
        output_cfg = config.get("output", {})
        train_cfg = config.get("model", {}).get("training", {})
        viz_cfg = config.get("visualization", {})

        self.metrics_path = Path(
            train_cfg.get("metrics_path", DEFAULT_METRICS_PATH)
        )

        self.weights_path = Path(
            output_cfg.get("risk_weights_path", DEFAULT_WEIGHTS_PATH)
        )
        self.figures_dir = Path(
            output_cfg.get("figures_dir", DEFAULT_FIGURES_DIR)
        )
        self.top_features = int(viz_cfg.get("shap", {}).get("top_features", 15))
        self.config_path = config_path

    def generate_all(self) -> list[Path]:
        dpi, figsize, fmt = _plot_config(self.config_path)
        saved: list[Path] = []

        with open(self.metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)

        cv_path = self.figures_dir / f"spatial_cv_scores.{fmt}"
        if plot_spatial_cv_scores(metrics, cv_path, figsize):
            saved.append(cv_path)

        with open(self.weights_path, encoding="utf-8") as fh:
            weights = json.load(fh)
        weights_path = self.figures_dir / f"risk_weights.{fmt}"
        saved.append(plot_risk_weights(weights, weights_path, figsize))

        imp_path = self.figures_dir / f"top_feature_importances.{fmt}"
        importances = metrics.get("feature_importances", {})
        if importances:
            saved.append(
                plot_top_importances(
                    importances,
                    imp_path,
                    figsize,
                    top_n=self.top_features,
                )
            )

        logger.info("Wrote %s figures to %s (dpi=%s)", len(saved), self.figures_dir, dpi)
        return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance plots")
    parser.add_argument("--metrics", default=DEFAULT_METRICS_PATH)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    plotter = PerformancePlotter(config_path=args.config)
    plotter.metrics_path = Path(args.metrics)
    plotter.weights_path = Path(args.weights)
    plotter.figures_dir = Path(args.output_dir)
    plotter.generate_all()


if __name__ == "__main__":
    main()
