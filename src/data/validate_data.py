"""
Validate CAL FIRE perimeter records and prepare interim centroid table.

Filters invalid geometry, missing dates, unreasonable acreage, and points outside
California. Writes validated GeoPackage for downstream feature engineering.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# WGS84 bounding box for California (conservative)
CALIFORNIA_BOUNDS = box(-124.5, 32.5, -114.0, 42.0)
TARGET_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:5070"  # Albers — area-preserving for centroid calculation


@dataclass(frozen=True)
class ValidationThresholds:
    """Thresholds for per-record fire validation."""

    min_acres: float = 10.0
    max_acres: float = 1_000_000.0
    require_cont_date: bool = True
    california_bounds: Any = CALIFORNIA_BOUNDS


def _parse_epoch_ms(value: Any) -> pd.Timestamp | pd.NaT:
    """Parse ArcGIS epoch-millisecond dates (float) or datetime-like values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    if pd.isna(value):
        return pd.NaT

    try:
        if isinstance(value, (int, float)):
            return pd.to_datetime(value, unit="ms", utc=True)
        return pd.to_datetime(value, utc=True, errors="coerce")
    except (TypeError, ValueError):
        return pd.NaT


def _get_centroid_point(row: pd.Series) -> Point | None:
    geom = row.geometry
    if geom is None or geom.is_empty:
        return None
    try:
        return geom.centroid
    except Exception:
        return None


def validate_fire_record(
    fire_row: pd.Series,
    thresholds: Optional[ValidationThresholds] = None,
) -> bool:
    """
    Return True if a fire record passes all quality checks.

    Args:
        fire_row: Single row from a fire perimeter GeoDataFrame.
        thresholds: Optional validation thresholds (defaults from config-like values).

    Returns:
        True if all checks pass.
    """
    thresholds = thresholds or ValidationThresholds()
    checks = validation_checks(fire_row, thresholds)
    return all(checks.values())


def validation_checks(
    fire_row: pd.Series,
    thresholds: Optional[ValidationThresholds] = None,
) -> dict[str, bool]:
    """
    Run validation checks and return per-check boolean results.

    Useful for debugging exclusions in logs or reports.
    """
    thresholds = thresholds or ValidationThresholds()

    geom = fire_row.geometry if hasattr(fire_row, "geometry") else None
    centroid = _get_centroid_point(fire_row)

    alarm = _parse_epoch_ms(fire_row.get("ALARM_DATE"))
    cont = _parse_epoch_ms(fire_row.get("CONT_DATE"))

    acres = fire_row.get("GIS_ACRES")
    try:
        acres_ok = (
            pd.notna(acres)
            and thresholds.min_acres <= float(acres) <= thresholds.max_acres
        )
    except (TypeError, ValueError):
        acres_ok = False

    within_ca = False
    if centroid is not None and not centroid.is_empty:
        within_ca = thresholds.california_bounds.contains(centroid)

    checks = {
        "has_geometry": geom is not None and not geom.is_empty,
        "geometry_valid": geom is not None and geom.is_valid,
        "has_alarm_date": pd.notna(alarm),
        "reasonable_size": acres_ok,
        "within_california": within_ca,
    }

    if thresholds.require_cont_date:
        checks["has_cont_date"] = pd.notna(cont)
        if checks["has_alarm_date"] and checks["has_cont_date"]:
            checks["cont_after_alarm"] = cont >= alarm
        else:
            checks["cont_after_alarm"] = False

    return checks


def add_centroid_columns(fires: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add centroid longitude/latitude columns for weather and terrain extraction.

    Args:
        fires: GeoDataFrame in geographic CRS (EPSG:4326 recommended).

    Returns:
        Copy of ``fires`` with ``centroid_lon`` and ``centroid_lat`` columns.
    """
    if fires.crs is None:
        raise ValueError("Fire GeoDataFrame has no CRS; expected EPSG:4326")

    working = fires.to_crs(TARGET_CRS) if fires.crs.to_string() != TARGET_CRS else fires.copy()
    centroids = (
        working.to_crs(PROJECTED_CRS)
        .geometry.centroid.to_crs(TARGET_CRS)
    )

    working["centroid_lon"] = centroids.x
    working["centroid_lat"] = centroids.y

    return working


def validate_fires(
    fires: gpd.GeoDataFrame,
    thresholds: Optional[ValidationThresholds] = None,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Filter a fire perimeter GeoDataFrame to valid records only.

    Args:
        fires: Raw or quality-filtered fire perimeters.
        thresholds: Validation thresholds.

    Returns:
        Tuple of (valid GeoDataFrame, exclusion report DataFrame).
    """
    thresholds = thresholds or ValidationThresholds()

    if fires.crs is None:
        logger.warning("Input CRS missing; assuming %s", TARGET_CRS)
        fires = fires.set_crs(TARGET_CRS)
    elif fires.crs.to_string() != TARGET_CRS:
        fires = fires.to_crs(TARGET_CRS)

    n_before = len(fires)
    report_rows: list[dict[str, Any]] = []
    keep_mask: list[bool] = []

    for idx, row in fires.iterrows():
        checks = validation_checks(row, thresholds)
        passed = all(checks.values())
        keep_mask.append(passed)

        if not passed:
            failed = [name for name, ok in checks.items() if not ok]
            report_rows.append(
                {
                    "index": idx,
                    "OBJECTID": row.get("OBJECTID"),
                    "FIRE_NAME": row.get("FIRE_NAME"),
                    "failed_checks": ",".join(failed),
                }
            )

    valid = fires.loc[keep_mask].copy()
    valid = add_centroid_columns(valid)

    report = pd.DataFrame(report_rows)
    n_after = len(valid)
    n_removed = n_before - n_after
    pct = 100 * n_removed / n_before if n_before else 0.0

    logger.info(
        "Validated fires: %s -> %s (%s removed, %.1f%%)",
        n_before,
        n_after,
        n_removed,
        pct,
    )

    if not report.empty:
        logger.info(
            "Top failure reasons: %s",
            report["failed_checks"].value_counts().head(5).to_dict(),
        )

    return valid, report


def thresholds_from_config(config_path: str = "config/pipeline_config.yaml") -> ValidationThresholds:
    """Build validation thresholds from pipeline YAML."""
    config = load_config(config_path)
    calfire = config.get("data", {}).get("calfire", {})

    return ValidationThresholds(
        min_acres=float(calfire.get("min_acres", 10.0)),
        max_acres=float(calfire.get("max_acres", 1_000_000.0)),
        require_cont_date=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CAL FIRE perimeter records")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/calfire_perimeters.gpkg",
        help="Input GeoPackage path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/fires_with_centroids.gpkg",
        help="Output GeoPackage path",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional CSV path for excluded records",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Pipeline config YAML",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    logger.info("Loading fires from %s", input_path)
    fires = gpd.read_file(input_path)
    logger.info("Loaded %s records", len(fires))

    thresholds = thresholds_from_config(args.config)
    valid, report = validate_fires(fires, thresholds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid.to_file(output_path, driver="GPKG")
    logger.info("Wrote %s validated records to %s", len(valid), output_path)

    if args.report and not report.empty:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
        logger.info("Wrote exclusion report to %s", report_path)

    logger.info("=== Validation Summary ===")
    logger.info("Valid fires: %s", len(valid))
    if len(valid) and "YEAR_" in valid.columns:
        logger.info("Year range: %s-%s", valid["YEAR_"].min(), valid["YEAR_"].max())


if __name__ == "__main__":
    main()
