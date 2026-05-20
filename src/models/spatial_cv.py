"""
Assign spatial cross-validation groups for fire / pseudo-absence records.

Uses K-means on fire centroids to form geographic blocks so GroupKFold does not
mix nearby locations across train and test folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MATRIX_PATH = "data/processed/feature_matrix.parquet"
DEFAULT_GROUPS_PATH = "models/spatial_groups.npy"
DEFAULT_RESULTS_PATH = "models/spatial_cv_results.json"
PROJECTED_CRS = "EPSG:5070"

NON_FEATURE_COLUMNS = {
    "OBJECTID",
    "is_fire",
    "ALARM_DATE",
    "centroid_lon",
    "centroid_lat",
    "geometry",
}


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Model input columns (numeric features only)."""
    cols: list[str] = []
    for col in matrix.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(matrix[col]):
            cols.append(col)
    return cols


def assign_spatial_groups(
    lon: np.ndarray,
    lat: np.ndarray,
    n_groups: int,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster (lon, lat) into geographic blocks for GroupKFold."""
    if len(lon) < n_groups:
        raise ValueError(
            f"Need at least {n_groups} records for {n_groups} spatial groups, got {len(lon)}"
        )
    coords = np.column_stack([lon, lat])
    scaled = StandardScaler().fit_transform(coords)
    kmeans = KMeans(n_clusters=n_groups, random_state=random_state, n_init=10)
    return kmeans.fit_predict(scaled).astype(np.int32)


def _cluster_centroids(
    groups: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> gpd.GeoDataFrame:
    df = pd.DataFrame({"spatial_group": groups, "lon": lon, "lat": lat})
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)
    centroids = (
        gdf.groupby("spatial_group", as_index=False)
        .agg(
            x=("geometry", lambda s: float(s.union_all().centroid.x)),
            y=("geometry", lambda s: float(s.union_all().centroid.y)),
            n_records=("spatial_group", "count"),
        )
        .rename(columns={"spatial_group": "group"})
    )
    return centroids


def check_spatial_separation(
    groups: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    min_separation_km: float,
) -> dict[str, Any]:
    """
    Measure minimum distance between cluster centroids (km).

    Returns a report dict; ``passed`` is False when blocks are closer than threshold.
    """
    centroids = _cluster_centroids(groups, lon, lat)
    if len(centroids) < 2:
        return {
            "passed": True,
            "min_centroid_distance_km": None,
            "min_separation_km": min_separation_km,
            "n_groups": len(centroids),
        }

    min_dist_m = float("inf")
    closest_pair: tuple[int, int] | None = None
    for i, row_i in centroids.iterrows():
        for j, row_j in centroids.iterrows():
            if row_i["group"] >= row_j["group"]:
                continue
            dist = float(
                np.hypot(row_i["x"] - row_j["x"], row_i["y"] - row_j["y"])
            )
            if dist < min_dist_m:
                min_dist_m = dist
                closest_pair = (int(row_i["group"]), int(row_j["group"]))

    min_dist_km = min_dist_m / 1000.0
    passed = min_dist_km >= min_separation_km
    return {
        "passed": passed,
        "min_centroid_distance_km": round(min_dist_km, 2),
        "min_separation_km": min_separation_km,
        "closest_groups": closest_pair,
        "n_groups": int(centroids["group"].nunique()),
        "records_per_group": centroids.set_index("group")["n_records"].to_dict(),
    }


def check_groupkfold_leakage(
    groups: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
) -> dict[str, Any]:
    """
    Summarize GroupKFold splits: fires per fold and class balance warnings.
    """
    gkf = GroupKFold(n_splits=n_folds)
    fold_stats: list[dict[str, Any]] = []
    warnings: list[str] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(len(groups)), labels, groups)):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            warnings.append(f"Fold {fold_idx}: group overlap {overlap}")

        test_fires = int(labels[test_idx].sum())
        test_total = len(test_idx)
        fold_stats.append(
            {
                "fold": fold_idx,
                "n_test": test_total,
                "n_test_fires": test_fires,
                "n_test_groups": len(test_groups),
                "n_train_groups": len(train_groups),
            }
        )

    return {"folds": fold_stats, "warnings": warnings, "n_folds": n_folds}


def summarize_groups(
    groups: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Per-group fire vs pseudo counts."""
    summary: dict[str, Any] = {}
    for group in np.unique(groups):
        mask = groups == group
        summary[str(int(group))] = {
            "n_total": int(mask.sum()),
            "n_fires": int(labels[mask].sum()),
            "n_pseudo": int((~labels[mask].astype(bool)).sum()),
        }
    return summary


class SpatialCVAssigner:
    """Build spatial group labels for the feature matrix."""

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        config = load_config(config_path)
        spatial_cfg = config.get("features", {}).get("spatial", {})
        cv_cfg = config.get("model", {}).get("spatial_cv", {})
        model_cv = config.get("model", {}).get("cross_validation", {})

        self.n_groups = int(
            cv_cfg.get("n_spatial_groups", spatial_cfg.get("n_clusters", 10))
        )
        self.min_separation_km = float(
            cv_cfg.get(
                "min_group_separation_km",
                spatial_cfg.get("min_distance_km", 30),
            )
        )
        self.n_folds = int(model_cv.get("n_folds", 5))
        self.random_state = int(
            cv_cfg.get("random_state", config.get("model", {}).get("random_forest", {}).get("random_state", 42))
        )
        self.matrix_path = Path(
            cv_cfg.get("feature_matrix_path", DEFAULT_MATRIX_PATH)
        )
        self.groups_path = Path(cv_cfg.get("groups_path", DEFAULT_GROUPS_PATH))
        self.results_path = Path(
            cv_cfg.get("results_path", DEFAULT_RESULTS_PATH)
        )

    def run(
        self,
        matrix: Optional[pd.DataFrame] = None,
        n_groups: Optional[int] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        matrix = matrix if matrix is not None else pd.read_parquet(self.matrix_path)
        if matrix.empty:
            raise ValueError("Feature matrix is empty")

        for col in ("centroid_lon", "centroid_lat", "is_fire"):
            if col not in matrix.columns:
                raise ValueError(f"Feature matrix missing required column: {col}")

        n_groups = n_groups or self.n_groups
        if n_groups < self.n_folds:
            logger.warning(
                "n_spatial_groups (%s) < n_folds (%s); using %s groups",
                n_groups,
                self.n_folds,
                self.n_folds,
            )
            n_groups = self.n_folds

        lon = matrix["centroid_lon"].astype(float).values
        lat = matrix["centroid_lat"].astype(float).values
        labels = matrix["is_fire"].astype(int).values

        groups = assign_spatial_groups(
            lon,
            lat,
            n_groups=n_groups,
            random_state=self.random_state,
        )

        separation = check_spatial_separation(
            groups, lon, lat, self.min_separation_km
        )
        fold_report = check_groupkfold_leakage(groups, labels, self.n_folds)

        report: dict[str, Any] = {
            "n_records": len(matrix),
            "n_spatial_groups": n_groups,
            "n_folds": self.n_folds,
            "separation": separation,
            "group_summary": summarize_groups(groups, labels),
            "groupkfold": fold_report,
            "feature_columns": feature_columns(matrix),
            "passed": separation["passed"] and not fold_report["warnings"],
        }

        if not separation["passed"]:
            logger.warning(
                "Spatial blocks may be too close: min centroid distance %.2f km "
                "(threshold %.2f km). Consider increasing n_spatial_groups.",
                separation["min_centroid_distance_km"],
                self.min_separation_km,
            )
        else:
            logger.info(
                "Spatial separation OK: min centroid distance %.2f km",
                separation["min_centroid_distance_km"],
            )

        for warning in fold_report["warnings"]:
            logger.warning("%s", warning)

        return groups, report

    def save(
        self,
        groups: np.ndarray,
        report: dict[str, Any],
    ) -> tuple[Path, Path]:
        self.groups_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.groups_path, groups)
        logger.info("Saved spatial groups (%s records) to %s", len(groups), self.groups_path)

        with open(self.results_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Saved spatial CV report to %s", self.results_path)
        return self.groups_path, self.results_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign K-means spatial groups for GroupKFold cross-validation"
    )
    parser.add_argument(
        "--matrix",
        default=DEFAULT_MATRIX_PATH,
        help="Input feature matrix parquet",
    )
    parser.add_argument(
        "--groups-out",
        default=DEFAULT_GROUPS_PATH,
        help="Output .npy path (aligned to matrix row order)",
    )
    parser.add_argument(
        "--results-out",
        default=DEFAULT_RESULTS_PATH,
        help="Output JSON report path",
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=None,
        help="Number of spatial clusters (default from config)",
    )
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    assigner = SpatialCVAssigner(config_path=args.config)
    assigner.matrix_path = Path(args.matrix)
    assigner.groups_path = Path(args.groups_out)
    assigner.results_path = Path(args.results_out)

    groups, report = assigner.run(n_groups=args.n_groups)
    assigner.save(groups, report)

    logger.info(
        "Spatial CV complete: %s groups, passed=%s",
        report["n_spatial_groups"],
        report["passed"],
    )


if __name__ == "__main__":
    main()
