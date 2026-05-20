"""
Assemble fire and pseudo-absence rows into a single feature matrix for modeling.

Joins interim feature parquets (weather, terrain, spatial, vegetation) on OBJECTID,
optionally samples non-fire locations and extracts the same features for negatives.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.data.validate_data import CALIFORNIA_BOUNDS, _parse_epoch_ms
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FIRES_PATH = "data/interim/fires_with_centroids.gpkg"
DEFAULT_WEATHER_PATH = "data/interim/fires_weather_joined.parquet"
DEFAULT_TERRAIN_PATH = "data/interim/fires_terrain_joined.parquet"
DEFAULT_SPATIAL_PATH = "data/interim/fires_spatial_joined.parquet"
DEFAULT_VEGETATION_PATH = "data/interim/fires_vegetation_joined.parquet"
DEFAULT_OUTPUT_PATH = "data/processed/feature_matrix.parquet"

WGS84 = "EPSG:4326"
PROJECTED_CRS = "EPSG:5070"


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "OBJECTID"]


def _merge_features(
    base: pd.DataFrame,
    features: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    overlap = set(base.columns) & set(_feature_columns(features))
    if overlap:
        logger.warning("Dropping duplicate columns from %s: %s", name, sorted(overlap))
        features = features.drop(columns=list(overlap), errors="ignore")
    return base.merge(features, on="OBJECTID", how="left")


class FeatureMatrixBuilder:
    """Join per-modality features and build labeled fire / pseudo-absence rows."""

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        config = load_config(config_path)
        build_cfg = config.get("features", {}).get("build", {})
        data_cfg = config.get("data", {})
        spatial_cfg = config.get("features", {}).get("spatial", {})

        self.fires_path = Path(
            build_cfg.get("fires_path", DEFAULT_FIRES_PATH)
        )
        self.feature_paths = {
            "weather": Path(
                build_cfg.get("weather_path", DEFAULT_WEATHER_PATH)
            ),
            "terrain": Path(
                build_cfg.get("terrain_path", DEFAULT_TERRAIN_PATH)
            ),
            "spatial": Path(
                build_cfg.get("spatial_path", DEFAULT_SPATIAL_PATH)
            ),
            "vegetation": Path(
                build_cfg.get("vegetation_path", DEFAULT_VEGETATION_PATH)
            ),
        }
        self.output_path = Path(
            build_cfg.get("output_path", DEFAULT_OUTPUT_PATH)
        )
        self.pseudo_ratio = float(build_cfg.get("pseudo_absence_ratio", 1.0))
        self.min_distance_km = float(
            build_cfg.get("min_distance_km", spatial_cfg.get("min_distance_km", 2.0))
        )
        self.max_placement_attempts = int(
            build_cfg.get("max_placement_attempts", 500)
        )
        self.random_state = int(build_cfg.get("random_state", 42))
        self.california_bbox = tuple(
            build_cfg.get("california_bbox", [-124.5, 32.5, -114.0, 42.0])
        )
        self.required_modalities = build_cfg.get(
            "required_modalities", ["weather", "terrain", "spatial"]
        )

    def load_fires_table(
        self,
        limit: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        if not self.fires_path.is_file():
            raise FileNotFoundError(f"Fires not found at {self.fires_path}")

        fires = gpd.read_file(self.fires_path)
        if limit:
            fires = fires.head(limit)
        logger.info("Loaded %s fires from %s", len(fires), self.fires_path)
        return fires

    def join_fire_features(self, fires: gpd.GeoDataFrame) -> pd.DataFrame:
        """Left-join all available feature parquets onto fire OBJECTIDs."""
        base_cols = ["OBJECTID", "centroid_lon", "centroid_lat", "ALARM_DATE"]
        missing = [c for c in base_cols if c not in fires.columns]
        if missing:
            raise ValueError(f"Fires table missing columns: {missing}")

        matrix = fires[base_cols].copy()
        matrix["is_fire"] = 1

        for name, path in self.feature_paths.items():
            if not path.is_file():
                if name in self.required_modalities:
                    raise FileNotFoundError(
                        f"Required {name} features missing at {path}. "
                        f"Run the corresponding feature extractor first."
                    )
                logger.warning("Skipping missing %s features: %s", name, path)
                continue
            feat = pd.read_parquet(path)
            if "OBJECTID" not in feat.columns:
                raise ValueError(f"{path} must include OBJECTID column")
            n_before = len(matrix)
            matrix = _merge_features(matrix, feat, name)
            matched = matrix[_feature_columns(feat)[0]].notna().sum() if _feature_columns(feat) else 0
            logger.info(
                "Joined %s: %s/%s fires with data from %s",
                name,
                matched,
                n_before,
                path,
            )

        return matrix

    def generate_pseudo_points(
        self,
        fires: gpd.GeoDataFrame,
        n_pseudo: int,
    ) -> gpd.GeoDataFrame:
        """Sample random CA points at least min_distance_km from all fire centroids."""
        if n_pseudo <= 0:
            return gpd.GeoDataFrame(
                columns=["OBJECTID", "centroid_lon", "centroid_lat", "ALARM_DATE", "geometry"],
                crs=WGS84,
            )

        rng = np.random.default_rng(self.random_state)
        fire_pts = fires.to_crs(PROJECTED_CRS)
        fire_xy = np.column_stack(
            [fire_pts.geometry.x.values, fire_pts.geometry.y.values]
        )
        min_dist_m = self.min_distance_km * 1000.0

        west, south, east, north = self.california_bbox
        alarm_choices = fires["ALARM_DATE"].dropna()
        if alarm_choices.empty:
            alarm_choices = pd.Series([pd.NaT])

        records: list[dict[str, Any]] = []
        pseudo_id = -1

        for attempt in range(self.max_placement_attempts):
            if len(records) >= n_pseudo:
                break
            lon = rng.uniform(west, east)
            lat = rng.uniform(south, north)
            pt = gpd.GeoSeries([Point(lon, lat)], crs=WGS84).to_crs(PROJECTED_CRS).iloc[0]
            dists = np.hypot(fire_xy[:, 0] - pt.x, fire_xy[:, 1] - pt.y)
            if dists.min() < min_dist_m:
                continue

            records.append(
                {
                    "OBJECTID": pseudo_id,
                    "centroid_lon": lon,
                    "centroid_lat": lat,
                    "ALARM_DATE": alarm_choices.sample(1, random_state=rng).iloc[0],
                    "geometry": Point(lon, lat),
                }
            )
            pseudo_id -= 1

        if len(records) < n_pseudo:
            logger.warning(
                "Placed %s/%s pseudo-absences after %s attempts (min_distance_km=%s)",
                len(records),
                n_pseudo,
                self.max_placement_attempts,
                self.min_distance_km,
            )

        return gpd.GeoDataFrame(records, crs=WGS84)

    def extract_features_for_points(
        self,
        points: gpd.GeoDataFrame,
        extract_weather: bool = True,
        extract_terrain: bool = True,
        extract_spatial: bool = True,
        extract_vegetation: bool = True,
    ) -> pd.DataFrame:
        """Run feature extractors for arbitrary lon/lat points (e.g. pseudo-absences)."""
        from src.features.spatial_features import SpatialFeatureExtractor
        from src.features.terrain_features import TerrainFeatureExtractor
        from src.features.vegetation_features import VegetationFeatureExtractor

        work = points.copy()
        if "centroid_lon" not in work.columns or "centroid_lat" not in work.columns:
            work["centroid_lon"] = work.geometry.x
            work["centroid_lat"] = work.geometry.y

        matrix = work[["OBJECTID", "centroid_lon", "centroid_lat", "ALARM_DATE"]].copy()

        if extract_weather:
            from src.features.weather_features import WeatherFeatureExtractor

            weather_ext = WeatherFeatureExtractor()
            weather = weather_ext.extract_features_batch(work, batch_size=50)
            matrix = _merge_features(matrix, weather, "weather")

        if extract_terrain:
            terrain_ext = TerrainFeatureExtractor()
            terrain = terrain_ext.extract_terrain_batch(work, batch_size=50)
            matrix = _merge_features(matrix, terrain, "terrain")

        if extract_spatial:
            spatial_ext = SpatialFeatureExtractor()
            spatial = spatial_ext.extract_spatial_batch(work, batch_size=50)
            matrix = _merge_features(matrix, spatial, "spatial")

        if extract_vegetation:
            veg_ext = VegetationFeatureExtractor()
            records: list[dict[str, Any]] = []
            for _, row in work.iterrows():
                alarm = _parse_epoch_ms(row.get("ALARM_DATE"))
                year = int(alarm.year) if not pd.isna(alarm) else None
                alarm_dt = None if pd.isna(alarm) else alarm.to_pydatetime()
                feats = veg_ext.extract_vegetation_at_point(
                    float(row["centroid_lon"]),
                    float(row["centroid_lat"]),
                    year=year,
                    alarm_date=alarm_dt,
                )
                feats["OBJECTID"] = int(row["OBJECTID"])
                records.append(feats)
            matrix = _merge_features(matrix, pd.DataFrame(records), "vegetation")

        return matrix

    def build_matrix(
        self,
        limit: Optional[int] = None,
        skip_pseudo: bool = False,
        skip_pseudo_extract: bool = False,
    ) -> pd.DataFrame:
        """
        Build the full labeled feature matrix.

        Args:
            limit: Max fire records (smoke test).
            skip_pseudo: Omit pseudo-absence rows entirely.
            skip_pseudo_extract: Place pseudo points but do not call extractors (join-only test).
        """
        fires = self.load_fires_table(limit=limit)
        fire_matrix = self.join_fire_features(fires)

        if skip_pseudo:
            logger.info("Skipping pseudo-absences (skip_pseudo=True)")
            return fire_matrix

        n_pseudo = int(round(len(fires) * self.pseudo_ratio))
        pseudo_points = self.generate_pseudo_points(fires, n_pseudo)
        logger.info("Generated %s pseudo-absence locations", len(pseudo_points))

        if skip_pseudo_extract or len(pseudo_points) == 0:
            return fire_matrix

        pseudo_matrix = self.extract_features_for_points(pseudo_points)
        pseudo_matrix["is_fire"] = 0

        combined = pd.concat([fire_matrix, pseudo_matrix], ignore_index=True)
        logger.info(
            "Feature matrix: %s fires + %s pseudo = %s rows",
            len(fire_matrix),
            len(pseudo_matrix),
            len(combined),
        )
        return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join fire features and optional pseudo-absences into feature_matrix.parquet"
    )
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Max fires (smoke test)")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output parquet path",
    )
    parser.add_argument(
        "--skip-pseudo",
        action="store_true",
        help="Only join fire rows (no pseudo-absences)",
    )
    parser.add_argument(
        "--skip-pseudo-extract",
        action="store_true",
        help="Generate pseudo points but skip slow per-point extraction",
    )
    args = parser.parse_args()

    builder = FeatureMatrixBuilder(config_path=args.config)
    builder.output_path = Path(args.output)
    matrix = builder.build_matrix(
        limit=args.limit,
        skip_pseudo=args.skip_pseudo,
        skip_pseudo_extract=args.skip_pseudo_extract,
    )

    builder.output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(builder.output_path, index=False)
    logger.info(
        "Saved feature matrix (%s rows, %s columns) to %s",
        len(matrix),
        len(matrix.columns),
        builder.output_path,
    )
    logger.info("is_fire counts: %s", matrix["is_fire"].value_counts().to_dict())


if __name__ == "__main__":
    main()
