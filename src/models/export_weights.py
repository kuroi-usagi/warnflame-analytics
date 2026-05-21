"""
Export normalized Random Forest feature importances for warnflame integration.

Maps sklearn column importances to operational ``risk_weights.json`` keys and
validates weights sum to 1.0 (±0.01) via Pydantic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import joblib
from pydantic import BaseModel, Field, model_validator

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = "models/random_forest.joblib"
DEFAULT_METRICS_PATH = "models/training_metrics.json"
DEFAULT_OUTPUT_PATH = "models/risk_weights.json"
SUM_TOLERANCE = 0.01

# warnflame integration contract → candidate training feature columns
WEIGHT_FEATURE_MAP: dict[str, list[str]] = {
    "vegetation_density": ["vegetation_density", "ndvi_mean"],
    "slope_degrees": ["slope_degrees"],
    "aspect_south_factor": ["aspect_south_factor"],
    "elevation_meters": ["elevation_meters"],
    "infrastructure_distance_km": [
        "infrastructure_distance_km",
        "distance_to_roads_km",
    ],
    "weather_erc_max": [
        "erc_max_7d",
        "erc_max_14d",
        "erc_max_30d",
    ],
}

REQUIRED_KEYS = [
    "vegetation_density",
    "slope_degrees",
    "aspect_south_factor",
    "elevation_meters",
    "infrastructure_distance_km",
]


class RiskWeights(BaseModel):
    """Operational risk weights consumed by warnflame (sum ≈ 1.0)."""

    vegetation_density: float = Field(ge=0.0, le=1.0)
    slope_degrees: float = Field(ge=0.0, le=1.0)
    aspect_south_factor: float = Field(ge=0.0, le=1.0)
    elevation_meters: float = Field(ge=0.0, le=1.0)
    infrastructure_distance_km: float = Field(ge=0.0, le=1.0)
    weather_erc_max: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> "RiskWeights":
        values = [
            self.vegetation_density,
            self.slope_degrees,
            self.aspect_south_factor,
            self.elevation_meters,
            self.infrastructure_distance_km,
        ]
        if self.weather_erc_max is not None:
            values.append(self.weather_erc_max)
        total = float(sum(values))
        if abs(total - 1.0) > SUM_TOLERANCE:
            raise ValueError(
                f"Risk weights must sum to 1.0 ± {SUM_TOLERANCE}, got {total:.4f}"
            )
        return self


def load_feature_importances(
    model_path: Path,
    metrics_path: Path,
) -> dict[str, float]:
    """Load raw importances from joblib artifact or training metrics JSON."""
    if model_path.is_file():
        artifact = joblib.load(model_path)
        model = artifact["model"]
        columns = artifact["feature_columns"]
        return dict(zip(columns, model.feature_importances_.astype(float).tolist()))

    if metrics_path.is_file():
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)
        raw = metrics.get("feature_importances", {})
        return {str(k): float(v) for k, v in raw.items()}

    raise FileNotFoundError(
        f"No model at {model_path} or metrics at {metrics_path}. "
        "Run: python src/models/train_model.py"
    )


def aggregate_importance(
    importances: dict[str, float],
    candidate_columns: list[str],
) -> float:
    """Use the max importance among columns mapped to one operational factor."""
    values = [importances[col] for col in candidate_columns if col in importances]
    if not values:
        return 0.0
    return float(max(values))


def importances_to_raw_weights(
    importances: dict[str, float],
    include_weather: bool = True,
) -> dict[str, float]:
    """Collapse sklearn columns into warnflame contract keys (unnormalized)."""
    raw: dict[str, float] = {}
    for key in REQUIRED_KEYS:
        raw[key] = aggregate_importance(importances, WEIGHT_FEATURE_MAP[key])

    if include_weather:
        erc = aggregate_importance(importances, WEIGHT_FEATURE_MAP["weather_erc_max"])
        if erc > 0:
            raw["weather_erc_max"] = erc

    total = sum(raw.values())
    if total <= 0:
        raise ValueError(
            "No matching feature importances for warnflame weight keys. "
            f"Available columns sample: {list(importances.keys())[:8]}"
        )
    return raw


def normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    """Scale raw scores to sum to 1.0."""
    total = float(sum(raw.values()))
    if total <= 0:
        raise ValueError("Cannot normalize weights: total is zero")
    return {key: float(value / total) for key, value in raw.items()}


class RiskWeightExporter:
    """Build and validate ``risk_weights.json`` from a trained model."""

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        config = load_config(config_path)
        output_cfg = config.get("output", {})
        train_cfg = config.get("model", {}).get("training", {})

        self.model_path = Path(
            train_cfg.get("model_path", DEFAULT_MODEL_PATH)
        )
        self.metrics_path = Path(
            train_cfg.get("metrics_path", DEFAULT_METRICS_PATH)
        )
        self.output_path = Path(
            output_cfg.get("risk_weights_path", DEFAULT_OUTPUT_PATH)
        )

    def export(
        self,
        importances: Optional[dict[str, float]] = None,
        include_weather: bool = True,
    ) -> RiskWeights:
        importances = importances or load_feature_importances(
            self.model_path,
            self.metrics_path,
        )
        raw = importances_to_raw_weights(importances, include_weather=include_weather)
        normalized = normalize_weights(raw)
        weights = RiskWeights(**normalized)

        logger.info("Exported risk weights (sum=%.4f):", sum(normalized.values()))
        for key, value in sorted(normalized.items()):
            logger.info("  %s: %.4f", key, value)

        return weights

    def save(self, weights: RiskWeights) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = weights.model_dump(exclude_none=True)
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Saved risk weights to %s", self.output_path)
        return self.output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export normalized risk_weights.json for warnflame"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Trained model joblib from train_model.py",
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS_PATH,
        help="Fallback training_metrics.json",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path",
    )
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Omit weather_erc_max from export (5-factor weights only)",
    )
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    exporter = RiskWeightExporter(config_path=args.config)
    exporter.model_path = Path(args.model)
    exporter.metrics_path = Path(args.metrics)
    exporter.output_path = Path(args.output)

    weights = exporter.export(include_weather=not args.no_weather)
    exporter.save(weights)


if __name__ == "__main__":
    main()
