"""
Train Random Forest fire-risk classifier with spatial GroupKFold and temporal holdout.

Expects ``feature_matrix.parquet`` (fires + pseudo-absences) and aligned
``spatial_groups.npy`` from ``spatial_cv.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from src.models.spatial_cv import assign_spatial_groups, feature_columns
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MATRIX_PATH = "data/processed/feature_matrix.parquet"
DEFAULT_GROUPS_PATH = "models/spatial_groups.npy"
DEFAULT_MODEL_PATH = "models/random_forest.joblib"
DEFAULT_METRICS_PATH = "models/training_metrics.json"
LABEL_COLUMN = "is_fire"


def alarm_years(alarm_values: pd.Series) -> np.ndarray:
    """Extract calendar year from ALARM_DATE (epoch ms, ns, or datetime)."""
    numeric = pd.to_numeric(alarm_values, errors="coerce")
    if numeric.notna().any():
        vmax = float(numeric.max())
        if vmax > 1e14:
            parsed = pd.to_datetime(numeric, unit="ns", utc=True, errors="coerce")
        elif vmax > 1e11:
            parsed = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        else:
            parsed = pd.to_datetime(alarm_values, utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(alarm_values, utc=True, errors="coerce")
    return parsed.dt.year.fillna(-1).astype(np.int32).to_numpy()


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def build_synthetic_negatives(
    fires: pd.DataFrame,
    feature_cols: list[str],
    n_neg: int,
    california_bbox: tuple[float, float, float, float],
    random_state: int,
) -> pd.DataFrame:
    """
    Median-imputed pseudo-absences for smoke training when matrix has fires only.

    Not a substitute for real pseudo-absences from ``build_features.py``.
    """
    rng = np.random.default_rng(random_state)
    west, south, east, north = california_bbox
    medians = fires[feature_cols].median(numeric_only=True)

    records: list[dict[str, Any]] = []
    for i in range(n_neg):
        row: dict[str, Any] = {
            "OBJECTID": -(i + 1),
            "centroid_lon": float(rng.uniform(west, east)),
            "centroid_lat": float(rng.uniform(south, north)),
            "ALARM_DATE": float(
                pd.Timestamp(
                    year=int(rng.integers(2000, 2025)),
                    month=int(rng.integers(6, 10)),
                    day=int(rng.integers(1, 28)),
                    tz="UTC",
                ).value
            ),
            LABEL_COLUMN: 0,
        }
        for col in feature_cols:
            row[col] = float(medians.get(col, np.nan))
        records.append(row)

    return pd.DataFrame(records)


class FireRiskTrainer:
    """Train and evaluate Random Forest with spatial and temporal validation."""

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        config = load_config(config_path)
        model_cfg = config.get("model", {})
        train_cfg = model_cfg.get("training", {})
        rf_cfg = model_cfg.get("random_forest", {})
        cv_cfg = model_cfg.get("cross_validation", {})
        spatial_cfg = config.get("model", {}).get("spatial_cv", {})

        self.matrix_path = Path(
            train_cfg.get("feature_matrix_path", DEFAULT_MATRIX_PATH)
        )
        self.groups_path = Path(train_cfg.get("groups_path", DEFAULT_GROUPS_PATH))
        self.model_path = Path(train_cfg.get("model_path", DEFAULT_MODEL_PATH))
        self.metrics_path = Path(
            train_cfg.get("metrics_path", DEFAULT_METRICS_PATH)
        )
        self.holdout_year = int(train_cfg.get("temporal_holdout_year", 2024))
        self.california_bbox = tuple(
            train_cfg.get(
                "california_bbox",
                [-124.5, 32.5, -114.0, 42.0],
            )
        )
        self.n_folds = int(cv_cfg.get("n_folds", 5))
        self.n_spatial_groups = int(
            spatial_cfg.get("n_spatial_groups", 10)
        )
        self.random_state = int(rf_cfg.get("random_state", 42))

        self.rf_params = {
            "n_estimators": int(rf_cfg.get("n_estimators", 100)),
            "max_depth": rf_cfg.get("max_depth", 20),
            "min_samples_split": int(rf_cfg.get("min_samples_split", 50)),
            "min_samples_leaf": int(rf_cfg.get("min_samples_leaf", 20)),
            "max_features": rf_cfg.get("max_features", "sqrt"),
            "class_weight": rf_cfg.get("class_weight", "balanced"),
            "random_state": self.random_state,
            "n_jobs": rf_cfg.get("n_jobs", -1),
        }

    def load_training_table(
        self,
        matrix: Optional[pd.DataFrame] = None,
        synthetic_negatives: bool = False,
    ) -> pd.DataFrame:
        matrix = matrix if matrix is not None else pd.read_parquet(self.matrix_path)
        if matrix.empty:
            raise ValueError("Feature matrix is empty")

        for col in (LABEL_COLUMN, "centroid_lon", "centroid_lat", "ALARM_DATE"):
            if col not in matrix.columns:
                raise ValueError(f"Feature matrix missing required column: {col}")

        labels = matrix[LABEL_COLUMN].astype(int)
        n_classes = labels.nunique()
        if n_classes < 2:
            if not synthetic_negatives:
                raise ValueError(
                    f"Training requires fire (1) and non-fire (0) labels; "
                    f"found only class {labels.unique().tolist()}. "
                    "Rebuild with: python src/features/build_features.py "
                    "or pass --synthetic-negatives for median-imputed smoke negatives."
                )
            logger.warning(
                "Matrix has only fires — adding synthetic pseudo-absences "
                "(median features). Rebuild feature matrix with real negatives for production."
            )
            feature_cols = feature_columns(matrix)
            negatives = build_synthetic_negatives(
                matrix,
                feature_cols,
                n_neg=len(matrix),
                california_bbox=self.california_bbox,
                random_state=self.random_state,
            )
            matrix = pd.concat([matrix, negatives], ignore_index=True)

        logger.info(
            "Training table: %s rows (fires=%s, non-fires=%s)",
            len(matrix),
            int((matrix[LABEL_COLUMN] == 1).sum()),
            int((matrix[LABEL_COLUMN] == 0).sum()),
        )
        return matrix

    def _assign_groups_from_coords(self, matrix: pd.DataFrame) -> np.ndarray:
        n_groups = min(self.n_spatial_groups, len(matrix))
        if n_groups < self.n_folds:
            n_groups = min(len(matrix), self.n_folds)
        return assign_spatial_groups(
            matrix["centroid_lon"].astype(float).values,
            matrix["centroid_lat"].astype(float).values,
            n_groups=n_groups,
            random_state=self.random_state,
        )

    def align_spatial_groups(
        self,
        matrix: pd.DataFrame,
        groups: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if groups is None and self.groups_path.is_file():
            groups = np.load(self.groups_path)

        if groups is not None and len(groups) != len(matrix):
            logger.warning(
                "Group count (%s) != matrix rows (%s); re-assigning spatial groups",
                len(groups),
                len(matrix),
            )
            groups = None

        if groups is None:
            logger.info(
                "Assigning K-means spatial groups for %s records",
                len(matrix),
            )
            return self._assign_groups_from_coords(matrix)

        return groups.astype(np.int32)

    def prepare_xy(
        self,
        matrix: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
        feature_cols = feature_columns(matrix)
        if not feature_cols:
            raise ValueError("No numeric feature columns found in matrix")

        X = matrix[feature_cols].copy()
        y = matrix[LABEL_COLUMN].astype(int).values
        years = alarm_years(matrix["ALARM_DATE"])
        return X, y, years, feature_cols

    def spatial_cross_validate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> dict[str, Any]:
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        gkf = GroupKFold(n_splits=self.n_folds)
        fold_scores: list[dict[str, Any]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_imputed, y, groups)):
            model = RandomForestClassifier(**self.rf_params)
            model.fit(X_imputed[train_idx], y[train_idx])
            prob = model.predict_proba(X_imputed[test_idx])[:, 1]
            auc = _safe_roc_auc(y[test_idx], prob)
            acc = float(accuracy_score(y[test_idx], model.predict(X_imputed[test_idx])))
            fold_scores.append(
                {
                    "fold": fold_idx,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "roc_auc": auc,
                    "accuracy": acc,
                }
            )
            logger.info(
                "Spatial CV fold %s/%s: ROC-AUC=%s accuracy=%.3f",
                fold_idx + 1,
                self.n_folds,
                f"{auc:.3f}" if auc is not None else "n/a",
                acc,
            )

        aucs = [f["roc_auc"] for f in fold_scores if f["roc_auc"] is not None]
        return {
            "n_folds": self.n_folds,
            "folds": fold_scores,
            "roc_auc_mean": float(np.mean(aucs)) if aucs else None,
            "roc_auc_std": float(np.std(aucs)) if aucs else None,
        }

    def temporal_holdout_eval(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        years: np.ndarray,
        groups: np.ndarray,
    ) -> dict[str, Any]:
        train_mask = (years >= 0) & (years < self.holdout_year)
        test_mask = years == self.holdout_year

        report: dict[str, Any] = {
            "holdout_year": self.holdout_year,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "roc_auc": None,
            "accuracy": None,
            "confusion_matrix": None,
        }

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            logger.warning(
                "Temporal holdout skipped: train=%s test=%s for year %s",
                train_mask.sum(),
                test_mask.sum(),
                self.holdout_year,
            )
            return report

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X.loc[train_mask])
        X_test = imputer.transform(X.loc[test_mask])
        y_train = y[train_mask]
        y_test = y[test_mask]

        model = RandomForestClassifier(**self.rf_params)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)

        auc = _safe_roc_auc(y_test, prob)
        acc = float(accuracy_score(y_test, pred))
        report["roc_auc"] = auc
        report["accuracy"] = acc
        report["confusion_matrix"] = confusion_matrix(y_test, pred).tolist()
        report["n_train_fires"] = int(y_train.sum())
        report["n_test_fires"] = int(y_test.sum())

        logger.info(
            "Temporal holdout (year<%s vs %s): ROC-AUC=%s accuracy=%.3f "
            "(train=%s test=%s)",
            self.holdout_year,
            self.holdout_year,
            f"{auc:.3f}" if auc is not None else "n/a",
            acc,
            train_mask.sum(),
            test_mask.sum(),
        )
        return report

    def fit_final_model(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        years: np.ndarray,
    ) -> tuple[RandomForestClassifier, SimpleImputer, dict[str, float]]:
        """Fit on pre-holdout-year rows for deployment (avoids peeking at 2024)."""
        train_mask = (years >= 0) & (years < self.holdout_year)
        if train_mask.sum() < 10:
            train_mask = np.ones(len(y), dtype=bool)

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X.loc[train_mask])
        y_train = y[train_mask]

        model = RandomForestClassifier(**self.rf_params)
        model.fit(X_train, y_train)

        importances = dict(
            zip(X.columns, model.feature_importances_.astype(float).tolist())
        )
        top = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:10]
        logger.info("Top feature importances: %s", top)

        return model, imputer, importances

    def train(
        self,
        matrix: Optional[pd.DataFrame] = None,
        groups: Optional[np.ndarray] = None,
        synthetic_negatives: bool = False,
    ) -> tuple[dict[str, Any], RandomForestClassifier, SimpleImputer, list[str]]:
        used_synthetic = synthetic_negatives
        matrix = self.load_training_table(matrix, synthetic_negatives=synthetic_negatives)
        groups = self.align_spatial_groups(matrix, groups)
        X, y, years, feature_cols = self.prepare_xy(matrix)

        spatial_cv = self.spatial_cross_validate(X, y, groups)
        temporal = self.temporal_holdout_eval(X, y, years, groups)
        model, imputer, importances = self.fit_final_model(X, y, years)

        metrics: dict[str, Any] = {
            "n_samples": len(matrix),
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
            "class_counts": {
                "fire": int((y == 1).sum()),
                "non_fire": int((y == 0).sum()),
            },
            "spatial_cv": spatial_cv,
            "temporal_holdout": temporal,
            "feature_importances": importances,
            "model_params": self.rf_params,
            "synthetic_negatives_used": used_synthetic,
        }
        return metrics, model, imputer, feature_cols

    def save(
        self,
        model: RandomForestClassifier,
        imputer: SimpleImputer,
        feature_cols: list[str],
        metrics: dict[str, Any],
    ) -> tuple[Path, Path]:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": model,
            "imputer": imputer,
            "feature_columns": feature_cols,
        }
        joblib.dump(artifact, self.model_path)
        logger.info("Saved model artifact to %s", self.model_path)

        with open(self.metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Saved training metrics to %s", self.metrics_path)

        return self.model_path, self.metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Random Forest with spatial GroupKFold and temporal holdout"
    )
    parser.add_argument(
        "--matrix",
        default=DEFAULT_MATRIX_PATH,
        help="Feature matrix parquet",
    )
    parser.add_argument(
        "--groups",
        default=DEFAULT_GROUPS_PATH,
        help="Spatial group labels (.npy)",
    )
    parser.add_argument(
        "--model-out",
        default=DEFAULT_MODEL_PATH,
        help="Output joblib path (model + imputer + feature list)",
    )
    parser.add_argument(
        "--metrics-out",
        default=DEFAULT_METRICS_PATH,
        help="Output JSON metrics path",
    )
    parser.add_argument(
        "--synthetic-negatives",
        action="store_true",
        help="Add median-imputed pseudo-absences when matrix is fires-only",
    )
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    trainer = FireRiskTrainer(config_path=args.config)
    trainer.matrix_path = Path(args.matrix)
    trainer.groups_path = Path(args.groups)
    trainer.model_path = Path(args.model_out)
    trainer.metrics_path = Path(args.metrics_out)

    groups = np.load(args.groups) if Path(args.groups).is_file() else None
    metrics, model, imputer, feature_cols = trainer.train(
        groups=groups,
        synthetic_negatives=args.synthetic_negatives,
    )
    trainer.save(model, imputer, feature_cols, metrics)

    cv_auc = metrics["spatial_cv"].get("roc_auc_mean")
    holdout_auc = metrics["temporal_holdout"].get("roc_auc")
    logger.info(
        "Training complete — spatial CV ROC-AUC=%s, temporal %s holdout ROC-AUC=%s",
        f"{cv_auc:.3f}" if cv_auc is not None else "n/a",
        trainer.holdout_year,
        f"{holdout_auc:.3f}" if holdout_auc is not None else "n/a",
    )


if __name__ == "__main__":
    main()
