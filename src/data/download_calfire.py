"""
Download CAL FIRE historical fire perimeter data via ArcGIS FeatureServer.

Outputs GeoPackage under data/raw/calfire_perimeters.gpkg by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CALFIREDownloader:
    """
    Download and filter California fire perimeter polygons from ArcGIS REST API.

    Default service: CAL FIRE FRAP California Historic Fire Perimeters (Firep24_1).
    Source: https://data.ca.gov/dataset/california-fire-perimeters-all
    """

    DEFAULT_BASE_URL = (
        "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
        "California_Historic_Fire_Perimeters/FeatureServer/0/query"
    )

    PAGE_SIZE = 2000

    # FRAP C_METHOD domain (integer codes on the FeatureServer)
    METHOD_GPS_GROUND = 1
    METHOD_GPS_AIR = 2
    METHOD_INFRARED = 3
    METHOD_OTHER_IMAGERY = 4
    DEFAULT_QUALITY_METHOD_CODES = [1, 2, 3, 4]

    METHOD_LABEL_TO_CODES: dict[str, list[int]] = {
        "GPS": [1, 2],
        "IMAGERY": [3, 4],
    }

    def __init__(
        self,
        output_dir: str = "data/raw",
        min_year: int = 2000,
        max_year: Optional[int] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize downloader.

        Args:
            output_dir: Directory for output GeoPackage.
            min_year: Minimum fire year (inclusive).
            max_year: Maximum fire year (inclusive); defaults to current year.
            base_url: ArcGIS FeatureServer ``/query`` endpoint override.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_year = min_year
        self.max_year = max_year if max_year is not None else datetime.now().year
        self.base_url = base_url or self.DEFAULT_BASE_URL

        logger.info(
            "Initialized CAL FIRE downloader: years %s-%s",
            self.min_year,
            self.max_year,
        )

    def _build_where_clause(self) -> str:
        return f"YEAR_ >= {self.min_year} AND YEAR_ <= {self.max_year}"

    def _resolve_method_codes(
        self,
        required_methods: Optional[list[str | int]],
    ) -> list[int]:
        """Map labels (GPS, IMAGERY) or integer codes to FRAP C_METHOD values."""
        if required_methods is None:
            return self.DEFAULT_QUALITY_METHOD_CODES.copy()

        codes: list[int] = []
        for method in required_methods:
            if isinstance(method, int):
                codes.append(method)
                continue

            label = str(method).upper()
            if label in self.METHOD_LABEL_TO_CODES:
                codes.extend(self.METHOD_LABEL_TO_CODES[label])
            elif label.isdigit():
                codes.append(int(label))

        return codes or self.DEFAULT_QUALITY_METHOD_CODES.copy()

    def _fetch_page(self, where: str, last_object_id: int = 0) -> gpd.GeoDataFrame:
        """
        Fetch one page ordered by OBJECTID (ArcGIS caps GeoJSON at 2000 features).
        """
        page_where = (
            f"({where}) AND OBJECTID > {last_object_id}"
            if last_object_id
            else where
        )
        params = {
            "where": page_where,
            "outFields": "*",
            "f": "geojson",
            "returnGeometry": "true",
            "outSR": "4326",
            "orderByFields": "OBJECTID",
            "resultRecordCount": self.PAGE_SIZE,
        }

        response = requests.get(self.base_url, params=params, timeout=300)
        response.raise_for_status()

        payload = response.text.strip()
        if not payload or '"features":[]' in payload.replace(" ", ""):
            return gpd.GeoDataFrame()

        return gpd.read_file(StringIO(payload))

    def download_fires(
        self,
        output_filename: str = "calfire_perimeters.gpkg",
    ) -> gpd.GeoDataFrame:
        """
        Download fire perimeters for the configured year range.

        Args:
            output_filename: GeoPackage filename under ``output_dir``.

        Returns:
            GeoDataFrame of fire perimeters.

        Raises:
            requests.HTTPError: If the API request fails.
            ValueError: If no features are returned.
        """
        where = self._build_where_clause()
        logger.info("Downloading CAL FIRE perimeters (%s)...", where)

        frames: list[gpd.GeoDataFrame] = []
        last_object_id = 0

        while True:
            batch = self._fetch_page(where, last_object_id=last_object_id)
            if batch.empty:
                break

            frames.append(batch)
            logger.info(
                "Fetched %s records (after OBJECTID %s)",
                len(batch),
                last_object_id,
            )

            if len(batch) < self.PAGE_SIZE:
                break

            if "OBJECTID" not in batch.columns:
                logger.warning("OBJECTID missing; cannot paginate further")
                break

            last_object_id = int(batch["OBJECTID"].max())

        if not frames:
            raise ValueError(
                f"No fires found for years {self.min_year}-{self.max_year}"
            )

        fires = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            crs=frames[0].crs,
        )

        logger.info("Downloaded %s fire perimeters", len(fires))

        output_path = self.output_dir / output_filename
        fires.to_file(output_path, driver="GPKG")
        logger.info("Saved to %s", output_path)

        return fires

    def filter_by_quality(
        self,
        fires: gpd.GeoDataFrame,
        min_acres: float = 10.0,
        max_acres: float = 1_000_000.0,
        required_methods: Optional[list[str | int]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Filter fires by size, collection method, and geometry validity.

        Args:
            fires: Input fire perimeter GeoDataFrame.
            min_acres: Minimum fire size in acres.
            max_acres: Maximum fire size in acres.
            required_methods: FRAP method codes (1-8) or labels (GPS, IMAGERY).
                Default keeps GPS + imagery methods (codes 1-4).

        Returns:
            Filtered GeoDataFrame.
        """
        n_before = len(fires)
        filtered = fires.copy()

        if "GIS_ACRES" in filtered.columns:
            filtered = filtered[
                (filtered["GIS_ACRES"] >= min_acres)
                & (filtered["GIS_ACRES"] <= max_acres)
            ]

        if "C_METHOD" in filtered.columns:
            method_codes = self._resolve_method_codes(required_methods)
            filtered = filtered[filtered["C_METHOD"].isin(method_codes)]

        if "geometry" in filtered.columns:
            filtered = filtered[filtered.geometry.notna() & filtered.geometry.is_valid]

        n_after = len(filtered)
        n_removed = n_before - n_after
        pct = 100 * n_removed / n_before if n_before else 0.0

        logger.info(
            "Filtered fires: %s -> %s (%s removed, %.1f%%)",
            n_before,
            n_after,
            n_removed,
            pct,
        )

        return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CAL FIRE fire perimeter data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2000,
        help="Minimum year (default: 2000)",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Maximum year (default: current year)",
    )
    parser.add_argument(
        "--skip-quality-filter",
        action="store_true",
        help="Skip quality filtering",
    )
    args = parser.parse_args()

    downloader = CALFIREDownloader(
        output_dir=args.output_dir,
        min_year=args.min_year,
        max_year=args.max_year,
    )

    fires = downloader.download_fires()

    if not args.skip_quality_filter:
        fires = downloader.filter_by_quality(fires)
        output_path = Path(args.output_dir) / "calfire_perimeters.gpkg"
        fires.to_file(output_path, driver="GPKG")
        logger.info("Wrote filtered perimeters to %s", output_path)

    logger.info("=== Download Summary ===")
    logger.info("Total fires: %s", len(fires))
    if "YEAR_" in fires.columns and len(fires):
        logger.info(
            "Year range: %s-%s",
            fires["YEAR_"].min(),
            fires["YEAR_"].max(),
        )
    if "GIS_ACRES" in fires.columns and len(fires):
        logger.info("Total area burned: %s acres", f"{fires['GIS_ACRES'].sum():,.0f}")
        logger.info("Mean fire size: %s acres", f"{fires['GIS_ACRES'].mean():,.1f}")


if __name__ == "__main__":
    main()
