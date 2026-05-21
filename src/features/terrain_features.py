"""
Extract terrain features (elevation, slope, aspect) from USGS 3DEP DEM.

Supports a cached California DEM mosaic (fast batch sampling) or per-fire
patch requests via py3dep for smoke tests without a statewide download.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Literal, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

try:
    import py3dep
except ImportError:  # pragma: no cover - optional at import; required at runtime
    py3dep = None  # type: ignore[assignment]

logger = get_logger(__name__)

CALIFORNIA_BBOX = (-124.5, 32.5, -114.0, 42.0)  # west, south, east, north
TARGET_CRS = "EPSG:5070"
WGS84 = "EPSG:4326"

SLOPE_NODATA = 255
ASPECT_NODATA = 32767


def aspect_to_south_factor(aspect_degrees: float) -> float:
    """Map aspect (0-360, downslope direction) to south-exposure factor in [0, 1]."""
    radians = np.radians(aspect_degrees - 180.0)
    return float(max(0.0, np.cos(radians)))


def compute_slope_aspect(
    elevation_window: np.ndarray,
    pixel_size: float,
) -> tuple[float, float]:
    """Compute slope (degrees) and aspect (degrees) from a 3x3 elevation window."""
    if elevation_window.shape != (3, 3) or np.any(np.isnan(elevation_window)):
        return float("nan"), float("nan")

    dzdx = (elevation_window[1, 2] - elevation_window[1, 0]) / (2 * pixel_size)
    dzdy = (elevation_window[2, 1] - elevation_window[0, 1]) / (2 * pixel_size)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect_rad = np.arctan2(dzdy, -dzdx)
    slope_deg = float(np.degrees(slope_rad))
    aspect_deg = float(np.degrees(aspect_rad) % 360.0)
    return slope_deg, aspect_deg


def _terrain_feature_dict(
    elevation: float,
    slope: float,
    aspect: float,
) -> dict[str, float]:
    return {
        "elevation_meters": elevation,
        "slope_degrees": slope,
        "aspect_south_factor": aspect_to_south_factor(aspect)
        if not np.isnan(aspect)
        else float("nan"),
    }


def _patch_bbox(lon: float, lat: float, buffer_deg: float = 0.001) -> tuple[float, float, float, float]:
    return (lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)


def _clean_slope_aspect(slope: float, aspect: float) -> tuple[float, float]:
    if slope == SLOPE_NODATA or np.isnan(slope):
        slope = float("nan")
    if aspect == ASPECT_NODATA or np.isnan(aspect):
        aspect = float("nan")
    return float(slope), float(aspect)


class TerrainFeatureExtractor:
    """Sample elevation, slope, and aspect at fire centroids from 3DEP."""

    def __init__(
        self,
        dem_path: Optional[str] = None,
        resolution: Optional[int] = None,
        mode: Literal["mosaic", "patch"] = "mosaic",
        config_path: str = "config/pipeline_config.yaml",
    ):
        config = load_config(config_path)
        dem_cfg = config.get("data", {}).get("dem", {})

        self.resolution = resolution or dem_cfg.get("resolution", 10)
        self.target_crs = dem_cfg.get("crs", TARGET_CRS)
        self.mode = mode
        self.dem_path = Path(
            dem_path or f"cache/ca_dem_{self.resolution}m.tif",
        )

        self._dem: Optional[np.ndarray] = None
        self._transform = None
        self._crs = None

        logger.info(
            "Terrain extractor: mode=%s, resolution=%sm, dem=%s",
            self.mode,
            self.resolution,
            self.dem_path,
        )

    def ensure_dem(self, force_download: bool = False) -> None:
        """Download California DEM mosaic if not cached (mosaic mode)."""
        if self.dem_path.is_file() and not force_download:
            return

        if py3dep is None:
            raise ImportError(
                "py3dep is required for DEM download. Install with: pip install py3dep"
            )

        self.dem_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading California DEM at %sm (may take several minutes)...",
            self.resolution,
        )

        dem = py3dep.get_dem(CALIFORNIA_BBOX, resolution=self.resolution, crs=WGS84)
        dem.rio.to_raster(self.dem_path)
        logger.info("Saved DEM to %s", self.dem_path)

    def _load_dem(self) -> None:
        if self._dem is not None:
            return
        if not self.dem_path.is_file():
            self.ensure_dem()

        with rasterio.open(self.dem_path) as src:
            self._dem = src.read(1, masked=True)
            self._transform = src.transform
            self._crs = src.crs

    def sample_terrain_from_mosaic(self, lon: float, lat: float) -> dict[str, float]:
        """Sample elevation, slope, and aspect from a cached DEM using central differences."""
        self._load_dem()

        point = gpd.GeoSeries([Point(lon, lat)], crs=WGS84).to_crs(self._crs)
        x, y = point.iloc[0].x, point.iloc[0].y

        row, col = rowcol(self._transform, x, y)
        rows, cols = self._dem.shape

        if row < 1 or row >= rows - 1 or col < 1 or col >= cols - 1:
            return _terrain_feature_dict(float("nan"), float("nan"), float("nan"))

        window = self._dem[row - 1 : row + 2, col - 1 : col + 2].filled(np.nan)
        pixel_size = abs(self._transform.a)
        slope, aspect = compute_slope_aspect(window, pixel_size)
        elevation = (
            float(self._dem[row, col])
            if not self._dem.mask[row, col]
            else float("nan")
        )

        return _terrain_feature_dict(elevation, slope, aspect)

    def sample_terrain_from_patch(
        self,
        lon: float,
        lat: float,
        max_retries: int = 3,
    ) -> dict[str, float]:
        """Fetch a small 3DEP patch and sample DEM, slope, and aspect at the centroid."""
        if py3dep is None:
            raise ImportError(
                "py3dep is required for terrain extraction. Install with: pip install py3dep"
            )

        bbox = _patch_bbox(lon, lat)
        for attempt in range(max_retries):
            try:
                ds = py3dep.get_map(
                    ["DEM", "Slope Degrees", "Aspect Degrees"],
                    bbox,
                    resolution=self.resolution,
                    geo_crs=WGS84,
                    crs=self.target_crs,
                )
                point = gpd.GeoSeries([Point(lon, lat)], crs=WGS84).to_crs(
                    ds.rio.crs
                )
                px, py = point.iloc[0].x, point.iloc[0].y
                sample = ds.sel(x=px, y=py, method="nearest")

                elevation = float(sample["elevation"].values)
                slope, aspect = _clean_slope_aspect(
                    float(sample["slope_degrees"].values),
                    float(sample["aspect_degrees"].values),
                )
                if np.isnan(elevation):
                    return _terrain_feature_dict(float("nan"), float("nan"), float("nan"))
                return _terrain_feature_dict(elevation, slope, aspect)

            except Exception as exc:
                wait_time = 2**attempt
                logger.error(
                    "Terrain patch attempt %s/%s failed: %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)

        # Brief pause after exhausting retries (USGS often rate-limits / 502s)
        time.sleep(1.0)
        return _terrain_feature_dict(float("nan"), float("nan"), float("nan"))

    def sample_terrain_at_point(self, lon: float, lat: float) -> dict[str, float]:
        if self.mode == "patch":
            return self.sample_terrain_from_patch(lon, lat)
        return self.sample_terrain_from_mosaic(lon, lat)

    def extract_terrain_batch(
        self,
        fires: gpd.GeoDataFrame,
        batch_size: int = 500,
        limit: Optional[int] = None,
        resume_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Extract terrain features for fire centroids keyed by OBJECTID."""
        if self.mode == "mosaic":
            self.ensure_dem()

        work = fires.head(limit) if limit else fires
        completed: dict[int, dict[str, Any]] = {}

        if resume_path and resume_path.is_file():
            existing = pd.read_parquet(resume_path)
            for _, row in existing.iterrows():
                elev = row.get("elevation_meters", np.nan)
                if np.isfinite(elev):
                    completed[int(row["OBJECTID"])] = row.to_dict()
            logger.info(
                "Resuming terrain from %s (%s successful records)",
                resume_path,
                len(completed),
            )

        records: list[dict[str, Any]] = []

        for idx, (_, fire) in enumerate(work.iterrows()):
            object_id = int(fire["OBJECTID"])
            if object_id in completed:
                records.append(completed[object_id])
                continue

            if idx % batch_size == 0:
                logger.info("Terrain sampling %s/%s", idx + 1, len(work))

            lon = fire.get("centroid_lon")
            lat = fire.get("centroid_lat")
            if pd.isna(lon) or pd.isna(lat):
                centroid = fire.geometry.centroid
                lon, lat = centroid.x, centroid.y

            feats = self.sample_terrain_at_point(float(lon), float(lat))
            feats["OBJECTID"] = object_id
            records.append(feats)
            if np.isfinite(feats.get("elevation_meters", np.nan)):
                completed[object_id] = feats
                if self.mode == "patch":
                    time.sleep(0.25)
            else:
                logger.warning(
                    "No 3DEP data for OBJECTID %s at (%.4f, %.4f) — will retry on resume",
                    object_id,
                    lon,
                    lat,
                )

            if resume_path and (idx + 1) % batch_size == 0:
                pd.DataFrame(records).to_parquet(resume_path, index=False)

        return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract terrain features from 3DEP DEM")
    parser.add_argument(
        "--input",
        type=str,
        default="data/interim/fires_with_centroids.gpkg",
        help="Input GeoPackage with fire centroids",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/fires_terrain_joined.parquet",
        help="Output Parquet path",
    )
    parser.add_argument(
        "--mode",
        choices=["mosaic", "patch"],
        default="mosaic",
        help="mosaic: cached statewide DEM; patch: per-fire 3DEP request",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="Max fires (smoke test)")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint parquet for resume",
    )
    parser.add_argument(
        "--force-download-dem",
        action="store_true",
        help="Re-download mosaic DEM (mosaic mode only)",
    )
    args = parser.parse_args()

    fires = gpd.read_file(args.input)
    logger.info("Loaded %s fires from %s", len(fires), args.input)

    extractor = TerrainFeatureExtractor(mode=args.mode)
    if args.force_download_dem:
        extractor.ensure_dem(force_download=True)

    features = extractor.extract_terrain_batch(
        fires,
        batch_size=args.batch_size,
        limit=args.limit,
        resume_path=Path(args.resume) if args.resume else None,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.info(
        "Saved terrain features for %s/%s fires to %s",
        len(features),
        len(fires),
        output_path,
    )


if __name__ == "__main__":
    main()
