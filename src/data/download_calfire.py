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

    Default service: CAL FIRE / FRAP California Fire Perimeters (All).
    """

    DEFAULT_BASE_URL = (
        "https://services1.arcgis.com/jUJYIo9rS62FWOcc/arcgis/rest/services/"
        "California_Fire_Perimeters_All/FeatureServer/0/query"
    )

    PAGE_SIZE = 2000

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

    def _fetch_page(self, where: str, offset: int) -> gpd.GeoDataFrame:
        params = {
            "where": where,
            "outFields": "*",
            "f": "geojson",
            "returnGeometry": "true",
            "outSR": "4326",
            "resultOffset": offset,
            "resultRecordCount": self.PAGE_SIZE,
        }

        response = requests.get(self.base_url, params=params, timeout=300)
        response.raise_for_status()

        payload = response.text.strip()
        if not payload or payload == '{"type":"FeatureCollection","features":[]}':
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
        offset = 0

        while True:
            batch = self._fetch_page(where, offset)
            if batch.empty:
                break

            frames.append(batch)
            logger.info("Fetched %s records (offset %s)", len(batch), offset)

            if len(batch) < self.PAGE_SIZE:
                break
            offset += len(batch)

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
        required_methods: Optional[list[str]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Filter fires by size, collection method, and geometry validity.

        Args:
            fires: Input fire perimeter GeoDataFrame.
            min_acres: Minimum fire size in acres.
            max_acres: Maximum fire size in acres.
            required_methods: Collection methods to keep (e.g. GPS, IMAGERY).

        Returns:
            Filtered GeoDataFrame.
        """
        if required_methods is None:
            required_methods = ["GPS", "IMAGERY"]

        n_before = len(fires)
        filtered = fires.copy()

        if "GIS_ACRES" in filtered.columns:
            filtered = filtered[
                (filtered["GIS_ACRES"] >= min_acres)
                & (filtered["GIS_ACRES"] <= max_acres)
            ]

        if "C_METHOD" in filtered.columns:
            filtered = filtered[filtered["C_METHOD"].isin(required_methods)]

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
