"""
Extract pre-fire gridMET weather features for fire locations.

Uses pygridmet (HyRiver) for 4 km daily grids: temperature, precipitation,
wind, and fire danger indices (ERC, BI).
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import pandas as pd

from src.data.validate_data import _parse_epoch_ms
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

try:
    import pygridmet as gridmet
except ImportError:  # pragma: no cover - optional at import; required at runtime
    gridmet = None  # type: ignore[assignment]

logger = get_logger(__name__)

GRIDMET_MIN_DATE = datetime(1979, 1, 1)


def _normalize_gridmet_weather(weather: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """
    Map pygridmet column labels (e.g. 'tmmx (K)') to bare variable names.
    """
    rename: dict[str, str] = {}
    for col in weather.columns:
        base = col.split("(", 1)[0].strip()
        if base in variables:
            rename[col] = base
    out = weather.rename(columns=rename)
    return out[[c for c in variables if c in out.columns]]


class WeatherFeatureExtractor:
    """
    Extract pre-fire meteorological features from gridMET.

    For each fire, downloads daily variables for 7, 14, and 30-day windows
    before ignition and aggregates to max/mean (sum for precipitation).
    """

    DEFAULT_VARIABLES = ["tmmx", "tmmn", "pr", "vs", "erc", "bi"]
    DEFAULT_WINDOWS = [7, 14, 30]

    def __init__(
        self,
        variables: Optional[list[str]] = None,
        windows: Optional[list[int]] = None,
        cache_dir: str = "cache",
        required_completeness: float = 0.9,
        config_path: str = "config/pipeline_config.yaml",
    ):
        config = load_config(config_path)
        gridmet_cfg = config.get("data", {}).get("gridmet", {})
        weather_cfg = config.get("features", {}).get("weather", {})

        self.variables = variables or gridmet_cfg.get("variables", self.DEFAULT_VARIABLES)
        self.windows = windows or gridmet_cfg.get("windows", self.DEFAULT_WINDOWS)
        self.required_completeness = required_completeness or weather_cfg.get(
            "required_completeness", 0.9
        )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        expire_days = gridmet_cfg.get("cache_expire_days", 30)
        os.environ["HYRIVER_CACHE_NAME"] = str(self.cache_dir / "gridmet.db")
        os.environ["HYRIVER_CACHE_EXPIRE"] = str(expire_days)

        logger.info(
            "Initialized weather extractor: %s vars, windows %s",
            len(self.variables),
            self.windows,
        )

    def extract_weather_for_fire(
        self,
        centroid_lon: float,
        centroid_lat: float,
        ignition_date: datetime,
        max_retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Download daily gridMET data for the largest pre-fire window ending at ignition.

        Args:
            centroid_lon: Fire centroid longitude (WGS84).
            centroid_lat: Fire centroid latitude (WGS84).
            ignition_date: Fire alarm / ignition date.
            max_retries: Retry attempts on network failure.

        Returns:
            DataFrame indexed by date with weather columns, or None on failure.
        """
        if ignition_date < GRIDMET_MIN_DATE:
            logger.warning(
                "Ignition date %s before gridMET coverage (1979-01-01)",
                ignition_date,
            )
            return None

        max_window = max(self.windows)
        start_date = ignition_date - timedelta(days=max_window)
        end_date = ignition_date

        if gridmet is None:
            raise ImportError(
                "pygridmet is required for weather extraction. "
                "Install with: pip install pygridmet"
            )

        for attempt in range(max_retries):
            try:
                weather = gridmet.get_bycoords(
                    coords=(centroid_lon, centroid_lat),
                    dates=(
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                    ),
                    variables=self.variables,
                )

                expected_days = (end_date.date() - start_date.date()).days + 1
                actual_days = len(weather)

                if actual_days < self.required_completeness * expected_days:
                    logger.warning(
                        "Weather gap at (%.3f, %.3f): %s/%s days",
                        centroid_lon,
                        centroid_lat,
                        actual_days,
                        expected_days,
                    )
                    return None

                return _normalize_gridmet_weather(weather, self.variables)

            except Exception as exc:
                wait_time = 2**attempt
                logger.error(
                    "Weather extraction attempt %s/%s failed: %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)

        return None

    def compute_window_features(
        self,
        weather: pd.DataFrame,
        ignition_date: datetime,
        window_days: int,
    ) -> dict[str, float]:
        """Aggregate weather for a pre-fire window ending at ignition."""
        window_start = ignition_date - timedelta(days=window_days)
        if not isinstance(weather.index, pd.DatetimeIndex):
            weather = weather.copy()
            weather.index = pd.to_datetime(weather.index)

        window_weather = weather[
            (weather.index >= pd.Timestamp(window_start))
            & (weather.index <= pd.Timestamp(ignition_date))
        ]

        if len(window_weather) == 0:
            return {}

        features: dict[str, float] = {}
        for var in self.variables:
            if var not in window_weather.columns:
                continue

            features[f"{var}_max_{window_days}d"] = float(window_weather[var].max())
            features[f"{var}_mean_{window_days}d"] = float(window_weather[var].mean())
            if var == "pr":
                features[f"{var}_sum_{window_days}d"] = float(window_weather[var].sum())

        return features

    def extract_features_batch(
        self,
        fires: gpd.GeoDataFrame,
        batch_size: int = 100,
        limit: Optional[int] = None,
        resume_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Extract weather features for many fires.

        Args:
            fires: GeoDataFrame with centroid_lon, centroid_lat, ALARM_DATE, OBJECTID.
            batch_size: Progress logging interval.
            limit: Optional max fires (smoke tests).
            resume_path: Optional parquet checkpoint to skip completed OBJECTIDs.

        Returns:
            DataFrame of features keyed by OBJECTID.
        """
        work = fires.head(limit) if limit else fires
        completed: dict[int, dict[str, Any]] = {}

        if resume_path and resume_path.is_file():
            existing = pd.read_parquet(resume_path)
            for _, row in existing.iterrows():
                completed[int(row["OBJECTID"])] = row.to_dict()
            logger.info("Resuming from %s (%s records)", resume_path, len(completed))

        all_features: list[dict[str, Any]] = []

        for idx, (_, fire) in enumerate(work.iterrows()):
            object_id = int(fire["OBJECTID"])
            if object_id in completed:
                all_features.append(completed[object_id])
                continue

            if idx % batch_size == 0:
                logger.info("Processing fire %s/%s", idx + 1, len(work))

            lon = fire.get("centroid_lon")
            lat = fire.get("centroid_lat")
            if pd.isna(lon) or pd.isna(lat):
                centroid = fire.geometry.centroid
                lon, lat = centroid.x, centroid.y

            ignition = _parse_epoch_ms(fire.get("ALARM_DATE"))
            if pd.isna(ignition):
                continue
            ignition_dt = ignition.to_pydatetime().replace(tzinfo=None)

            weather = self.extract_weather_for_fire(lon, lat, ignition_dt)
            if weather is None:
                continue

            fire_features: dict[str, Any] = {"OBJECTID": object_id}
            for window in self.windows:
                fire_features.update(
                    self.compute_window_features(weather, ignition_dt, window)
                )

            all_features.append(fire_features)
            completed[object_id] = fire_features

            if resume_path and (idx + 1) % batch_size == 0:
                pd.DataFrame(all_features).to_parquet(resume_path, index=False)

        return pd.DataFrame(all_features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pre-fire gridMET weather features")
    parser.add_argument(
        "--input",
        type=str,
        default="data/interim/fires_with_centroids.gpkg",
        help="Input GeoPackage with fire centroids",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/fires_weather_joined.parquet",
        help="Output Parquet path",
    )
    parser.add_argument("--batch-size", type=int, default=100)
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

    extractor = WeatherFeatureExtractor()
    features = extractor.extract_features_batch(
        fires,
        batch_size=args.batch_size,
        limit=args.limit,
        resume_path=Path(args.resume) if args.resume else None,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.info(
        "Saved weather features for %s/%s fires to %s",
        len(features),
        len(fires),
        output_path,
    )


if __name__ == "__main__":
    main()
