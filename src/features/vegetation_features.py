"""
Extract Sentinel-2 vegetation indices (NDVI, NDMI) at fire centroids.

Samples from cached median composites built via Planetary Computer
(see ``src/data/download_sentinel2.py``).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.data.validate_data import _parse_epoch_ms
from src.features.sentinel2_point import fetch_indices_at_point
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

WGS84 = "EPSG:4326"
DEFAULT_NDVI_PATH = "cache/sentinel2_ndvi_ca.tif"
DEFAULT_NDMI_PATH = "cache/sentinel2_ndmi_ca.tif"

# Neutral defaults when composites are missing (warnflame mid-range vegetation)
DEFAULT_NDVI = 0.5
DEFAULT_NDMI = 0.3


def ndvi_to_vegetation_density(ndvi: float) -> float:
    """Map NDVI in [-1, 1] to vegetation density in [0, 1] for warnflame integration."""
    return float(np.clip((ndvi + 1.0) / 2.0, 0.0, 1.0))


def _sample_raster(path: Path, lon: float, lat: float) -> float:
    """Sample one band at WGS84 lon/lat; return NaN when pixel is nodata or outside extent."""
    with rasterio.open(path) as src:
        sample = next(src.sample([(lon, lat)], masked=True))
        value = sample[0]
    if np.ma.is_masked(value):
        return float("nan")
    value = float(value)
    if not np.isfinite(value):
        return float("nan")
    return value


class VegetationFeatureExtractor:
    """Sample NDVI/NDMI from cached composite rasters at fire locations."""

    def __init__(
        self,
        ndvi_path: str = DEFAULT_NDVI_PATH,
        ndmi_path: Optional[str] = None,
        cache_dir: str = "cache",
        allow_fallback: bool = True,
        point_fallback: Optional[bool] = None,
        config_path: str = "config/pipeline_config.yaml",
    ):
        config = load_config(config_path)
        s2_cfg = config.get("data", {}).get("sentinel2", {})

        self.enabled = s2_cfg.get("enabled", True)
        self.cache_dir = Path(cache_dir)
        self.ndvi_path = Path(ndvi_path or s2_cfg.get("ndvi_path", DEFAULT_NDVI_PATH))
        self.ndmi_path = Path(
            ndmi_path or s2_cfg.get("ndmi_path", DEFAULT_NDMI_PATH)
        )
        self.allow_fallback = allow_fallback
        self.point_fallback = (
            s2_cfg.get("point_fallback_on_miss", True)
            if point_fallback is None
            else point_fallback
        )
        self.composite_months = tuple(s2_cfg.get("composite_months", [6, 7, 8, 9]))
        self.cloud_threshold = int(s2_cfg.get("cloud_threshold", 20))
        self.point_max_scenes = int(s2_cfg.get("point_max_scenes", 5))
        self._warned_missing_raster = False
        self._warned_nodata_sample = False

        logger.info(
            "Vegetation extractor: enabled=%s, ndvi=%s",
            self.enabled,
            self.ndvi_path,
        )

    def _paths_for_year(self, year: int) -> tuple[Path, Path]:
        """Prefer year-specific composites when present."""
        ndvi = self.cache_dir / f"sentinel2_ndvi_{year}.tif"
        ndmi = self.cache_dir / f"sentinel2_ndmi_{year}.tif"
        if ndvi.is_file() and ndmi.is_file():
            return ndvi, ndmi
        return self.ndvi_path, self.ndmi_path

    def extract_vegetation_at_point(
        self,
        lon: float,
        lat: float,
        year: Optional[int] = None,
        alarm_date: Optional[datetime] = None,
    ) -> dict[str, float]:
        if not self.enabled:
            return self._fallback_features("sentinel2 disabled in config")

        ndvi_path, ndmi_path = (
            self._paths_for_year(year) if year else (self.ndvi_path, self.ndmi_path)
        )

        if not ndvi_path.is_file():
            return self._fallback_features(f"NDVI raster missing at {ndvi_path}")

        ndvi_val = _sample_raster(ndvi_path, lon, lat)
        if np.isnan(ndvi_val):
            if self.point_fallback and (year is not None or alarm_date is not None):
                ndvi_val, ndmi_pc = self._fetch_at_point(lon, lat, year, alarm_date)
                if not np.isnan(ndvi_val):
                    ndmi_val = ndmi_pc if not np.isnan(ndmi_pc) else DEFAULT_NDMI
                    return {
                        "ndvi_mean": ndvi_val,
                        "ndmi_mean": ndmi_val,
                        "vegetation_density": ndvi_to_vegetation_density(ndvi_val),
                    }
            return self._nodata_features(lon, lat, ndvi_path)

        if ndmi_path.is_file():
            ndmi_val = _sample_raster(ndmi_path, lon, lat)
            if np.isnan(ndmi_val):
                if not self.allow_fallback:
                    raise ValueError(
                        f"No NDMI coverage at ({lon:.4f}, {lat:.4f}) in {ndmi_path}."
                    )
                ndmi_val = DEFAULT_NDMI
        else:
            ndmi_val = DEFAULT_NDMI

        return {
            "ndvi_mean": ndvi_val,
            "ndmi_mean": ndmi_val,
            "vegetation_density": ndvi_to_vegetation_density(ndvi_val),
        }

    def _fetch_at_point(
        self,
        lon: float,
        lat: float,
        year: Optional[int],
        alarm_date: Optional[datetime],
    ) -> tuple[float, float]:
        try:
            return fetch_indices_at_point(
                lon,
                lat,
                year=year,
                alarm_date=alarm_date,
                months=self.composite_months,
                cloud_threshold=self.cloud_threshold,
                max_scenes=self.point_max_scenes,
            )
        except ImportError:
            return float("nan"), float("nan")

    def _nodata_features(self, lon: float, lat: float, ndvi_path: Path) -> dict[str, float]:
        if not self.allow_fallback:
            raise ValueError(
                f"No Sentinel-2 coverage at ({lon:.4f}, {lat:.4f}) in {ndvi_path}. "
                "The composite is sparse (common with --quick). Re-download with "
                "python src/data/download_sentinel2.py --year <YEAR> "
                "(omit --quick; use --max-scenes 40 or higher)."
            )
        if not self._warned_nodata_sample:
            logger.warning(
                "No NDVI at (%.4f, %.4f) — raster nodata and point fetch failed; "
                "using default NDVI/NDMI",
                lon,
                lat,
            )
            self._warned_nodata_sample = True
        return {
            "ndvi_mean": DEFAULT_NDVI,
            "ndmi_mean": DEFAULT_NDMI,
            "vegetation_density": ndvi_to_vegetation_density(DEFAULT_NDVI),
        }

    def _fallback_features(self, reason: str) -> dict[str, float]:
        if not self.allow_fallback:
            raise FileNotFoundError(
                f"{reason}. Run: python src/data/download_sentinel2.py "
                "or pass --no-fallback only when rasters exist"
            )
        if not self._warned_missing_raster:
            logger.warning("%s — using default NDVI/NDMI for all fires", reason)
            self._warned_missing_raster = True
        return {
            "ndvi_mean": DEFAULT_NDVI,
            "ndmi_mean": DEFAULT_NDMI,
            "vegetation_density": ndvi_to_vegetation_density(DEFAULT_NDVI),
        }

    def extract_vegetation_batch(
        self,
        fires: gpd.GeoDataFrame,
        batch_size: int = 500,
        limit: Optional[int] = None,
        resume_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        work = fires.head(limit) if limit else fires
        completed: dict[int, dict[str, Any]] = {}

        if resume_path and resume_path.is_file():
            existing = pd.read_parquet(resume_path)
            for _, row in existing.iterrows():
                completed[int(row["OBJECTID"])] = row.to_dict()
            logger.info("Resuming vegetation from %s (%s records)", resume_path, len(completed))

        records: list[dict[str, Any]] = []

        for idx, (_, fire) in enumerate(work.iterrows()):
            object_id = int(fire["OBJECTID"])
            if object_id in completed:
                records.append(completed[object_id])
                continue

            if idx % batch_size == 0:
                logger.info("Vegetation sampling %s/%s", idx + 1, len(work))

            lon = fire.get("centroid_lon")
            lat = fire.get("centroid_lat")
            if pd.isna(lon) or pd.isna(lat):
                centroid = fire.geometry.centroid
                lon, lat = centroid.x, centroid.y

            year: Optional[int] = None
            alarm = _parse_epoch_ms(fire.get("ALARM_DATE"))
            if not pd.isna(alarm):
                year = int(alarm.year)

            alarm_dt: Optional[datetime] = None
            if not pd.isna(alarm):
                alarm_dt = alarm.to_pydatetime()

            feats = self.extract_vegetation_at_point(
                float(lon), float(lat), year=year, alarm_date=alarm_dt
            )
            feats["OBJECTID"] = object_id
            records.append(feats)
            completed[object_id] = feats

            if resume_path and (idx + 1) % batch_size == 0:
                pd.DataFrame(records).to_parquet(resume_path, index=False)

        return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Sentinel-2 vegetation features")
    parser.add_argument(
        "--input",
        type=str,
        default="data/interim/fires_with_centroids.gpkg",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/fires_vegetation_joined.parquet",
    )
    parser.add_argument("--ndvi-raster", default=None)
    parser.add_argument("--ndmi-raster", default=None)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="Max fires (smoke test)")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint parquet for resume",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail if composite rasters are missing",
    )
    parser.add_argument(
        "--no-point-fetch",
        action="store_true",
        help="Do not query Planetary Computer when raster has nodata at a fire",
    )
    args = parser.parse_args()

    fires = gpd.read_file(args.input)
    logger.info("Loaded %s fires from %s", len(fires), args.input)

    extractor = VegetationFeatureExtractor(
        ndvi_path=args.ndvi_raster or DEFAULT_NDVI_PATH,
        ndmi_path=args.ndmi_raster,
        allow_fallback=not args.no_fallback,
        point_fallback=not args.no_point_fetch,
    )
    features = extractor.extract_vegetation_batch(
        fires,
        batch_size=args.batch_size,
        limit=args.limit,
        resume_path=Path(args.resume) if args.resume else None,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.info(
        "Saved vegetation features for %s/%s fires to %s",
        len(features),
        len(fires),
        output_path,
    )


if __name__ == "__main__":
    main()
