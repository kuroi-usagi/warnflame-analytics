"""
Extract spatial context features: distance to roads (WUI proxy for infrastructure).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_CRS = "EPSG:5070"
WGS84 = "EPSG:4326"
DEFAULT_ROADS_PATH = "data/raw/california_roads.gpkg"


class SpatialFeatureExtractor:
    """Compute distance from fire centroids to nearest road segment."""

    def __init__(
        self,
        roads_path: str = DEFAULT_ROADS_PATH,
        config_path: str = "config/pipeline_config.yaml",
    ):
        config = load_config(config_path)
        roads_cfg = config.get("data", {}).get("roads", {})

        self.roads_path = Path(roads_path or roads_cfg.get("path", DEFAULT_ROADS_PATH))
        self.target_crs = roads_cfg.get("crs", TARGET_CRS)
        self._roads: Optional[gpd.GeoDataFrame] = None

        logger.info("Spatial extractor: roads=%s", self.roads_path)

    def load_roads(self) -> gpd.GeoDataFrame:
        if self._roads is not None:
            return self._roads

        if not self.roads_path.is_file():
            raise FileNotFoundError(
                f"Roads not found at {self.roads_path}. "
                "Run: python src/data/download_roads.py"
            )

        roads = gpd.read_file(self.roads_path)
        if roads.crs is None:
            roads = roads.set_crs(WGS84)
        if roads.crs.to_string() != self.target_crs:
            roads = roads.to_crs(self.target_crs)

        self._roads = roads
        logger.info("Loaded %s road segments", len(roads))
        return roads

    def distance_to_nearest_km(self, lon: float, lat: float) -> float:
        """Return distance in km from a WGS84 point to the nearest road."""
        point_gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)],
            crs=WGS84,
        ).to_crs(self.target_crs)

        roads = self.load_roads()
        nearest = gpd.sjoin_nearest(
            point_gdf,
            roads,
            how="left",
            distance_col="dist_m",
        )
        return float(nearest["dist_m"].min()) / 1000.0

    def extract_spatial_batch(
        self,
        fires: gpd.GeoDataFrame,
        batch_size: int = 500,
        limit: Optional[int] = None,
        resume_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Extract road-distance features for fire centroids.

        Uses GeoPandas spatial join for segment-accurate nearest distances.
        """
        work = fires.head(limit) if limit else fires
        completed: dict[int, dict[str, Any]] = {}

        if resume_path and resume_path.is_file():
            existing = pd.read_parquet(resume_path)
            for _, row in existing.iterrows():
                completed[int(row["OBJECTID"])] = row.to_dict()
            logger.info("Resuming spatial from %s (%s records)", resume_path, len(completed))

        pending_rows: list[dict[str, Any]] = []
        pending_points: list[Point] = []
        records: list[dict[str, Any]] = []

        roads = self.load_roads()

        def flush_pending() -> None:
            if not pending_rows:
                return

            points_gdf = gpd.GeoDataFrame(
                pending_rows,
                geometry=pending_points,
                crs=WGS84,
            ).to_crs(self.target_crs)

            nearest = gpd.sjoin_nearest(
                points_gdf,
                roads,
                how="left",
                distance_col="dist_m",
            )
            # sjoin_nearest returns ties when multiple roads share the same min distance
            nearest = (
                nearest.sort_values(["OBJECTID", "dist_m"])
                .drop_duplicates(subset=["OBJECTID"], keep="first")
            )

            for _, row in nearest.iterrows():
                dist_km = float(row["dist_m"]) / 1000.0
                record = {
                    "OBJECTID": int(row["OBJECTID"]),
                    "distance_to_roads_km": dist_km,
                    "infrastructure_distance_km": dist_km,
                }
                records.append(record)
                completed[int(row["OBJECTID"])] = record

            pending_rows.clear()
            pending_points.clear()

            if resume_path:
                pd.DataFrame(records).to_parquet(resume_path, index=False)

        for idx, (_, fire) in enumerate(work.iterrows()):
            object_id = int(fire["OBJECTID"])
            if object_id in completed:
                records.append(completed[object_id])
                continue

            if idx % batch_size == 0:
                logger.info("Spatial features %s/%s", idx + 1, len(work))

            lon = fire.get("centroid_lon")
            lat = fire.get("centroid_lat")
            if pd.isna(lon) or pd.isna(lat):
                centroid = fire.geometry.centroid
                lon, lat = centroid.x, centroid.y

            pending_rows.append({"OBJECTID": object_id})
            pending_points.append(Point(float(lon), float(lat)))

            if len(pending_rows) >= batch_size:
                flush_pending()

        flush_pending()
        return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract road distance features")
    parser.add_argument(
        "--input",
        type=str,
        default="data/interim/fires_with_centroids.gpkg",
        help="Input GeoPackage with fire centroids",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/fires_spatial_joined.parquet",
        help="Output Parquet path",
    )
    parser.add_argument("--roads", type=str, default=None, help="Roads GeoPackage path")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="Max fires (smoke test)")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint parquet for resume",
    )
    args = parser.parse_args()

    fires = gpd.read_file(args.input)
    logger.info("Loaded %s fires from %s", len(fires), args.input)

    extractor = SpatialFeatureExtractor(roads_path=args.roads or DEFAULT_ROADS_PATH)
    features = extractor.extract_spatial_batch(
        fires,
        batch_size=args.batch_size,
        limit=args.limit,
        resume_path=Path(args.resume) if args.resume else None,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.info(
        "Saved spatial features for %s/%s fires to %s",
        len(features),
        len(fires),
        output_path,
    )


if __name__ == "__main__":
    main()
