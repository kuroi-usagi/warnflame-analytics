"""Tests for src.data.validate_data."""

from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from src.data.validate_data import (
    ValidationThresholds,
    add_centroid_columns,
    validate_fire_record,
    validate_fires,
    validation_checks,
)

CA_POLY = Polygon(
    [
        (-121.0, 38.0),
        (-120.9, 38.0),
        (-120.9, 38.1),
        (-121.0, 38.1),
        (-121.0, 38.0),
    ]
)
OUTSIDE_CA_POLY = Polygon(
    [
        (-110.0, 40.0),
        (-109.9, 40.0),
        (-109.9, 40.1),
        (-110.0, 40.1),
        (-110.0, 40.0),
    ]
)


def _ms(dt: datetime) -> float:
    return dt.replace(tzinfo=timezone.utc).timestamp() * 1000


def _row(geometry, alarm_ms, cont_ms, acres=100.0, objectid=1):
    return pd.Series(
        {
            "OBJECTID": objectid,
            "FIRE_NAME": "Test Fire",
            "ALARM_DATE": alarm_ms,
            "CONT_DATE": cont_ms,
            "GIS_ACRES": acres,
            "geometry": geometry,
        }
    )


@pytest.fixture
def thresholds():
    return ValidationThresholds(min_acres=10.0, max_acres=1_000_000.0)


def test_validate_fire_record_passes(thresholds):
    alarm = _ms(datetime(2020, 8, 1))
    cont = _ms(datetime(2020, 8, 10))
    row = _row(CA_POLY, alarm, cont)

    assert validate_fire_record(row, thresholds) is True


def test_validate_fire_record_fails_outside_california(thresholds):
    alarm = _ms(datetime(2020, 8, 1))
    cont = _ms(datetime(2020, 8, 10))
    row = _row(OUTSIDE_CA_POLY, alarm, cont)

    checks = validation_checks(row, thresholds)
    assert checks["within_california"] is False
    assert validate_fire_record(row, thresholds) is False


def test_validate_fire_record_fails_small_acres(thresholds):
    alarm = _ms(datetime(2020, 8, 1))
    cont = _ms(datetime(2020, 8, 10))
    row = _row(CA_POLY, alarm, cont, acres=5.0)

    assert validation_checks(row, thresholds)["reasonable_size"] is False


def test_validate_fire_record_fails_missing_alarm(thresholds):
    row = _row(CA_POLY, float("nan"), _ms(datetime(2020, 8, 10)))
    checks = validation_checks(row, thresholds)

    assert checks["has_alarm_date"] is False


def test_validate_fires_filters_batch(thresholds):
    alarm = _ms(datetime(2020, 8, 1))
    cont = _ms(datetime(2020, 8, 10))

    gdf = gpd.GeoDataFrame(
        {
            "OBJECTID": [1, 2],
            "FIRE_NAME": ["Good", "Bad"],
            "ALARM_DATE": [alarm, alarm],
            "CONT_DATE": [cont, cont],
            "GIS_ACRES": [100.0, 100.0],
            "geometry": [CA_POLY, OUTSIDE_CA_POLY],
        },
        crs="EPSG:4326",
    )

    valid, report = validate_fires(gdf, thresholds)

    assert len(valid) == 1
    assert len(report) == 1
    assert "centroid_lon" in valid.columns
    assert "centroid_lat" in valid.columns


def test_add_centroid_columns():
    gdf = gpd.GeoDataFrame(
        {"geometry": [CA_POLY]},
        crs="EPSG:4326",
    )
    with_centroids = add_centroid_columns(gdf)

    assert "centroid_lon" in with_centroids.columns
    assert "centroid_lat" in with_centroids.columns
    assert isinstance(with_centroids.iloc[0]["centroid_lon"], float)
