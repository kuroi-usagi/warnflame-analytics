"""Tests for src.features.spatial_features."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from src.features.spatial_features import SpatialFeatureExtractor


@pytest.fixture
def roads_gpkg(tmp_path: Path) -> Path:
    """Horizontal road along y=0; point (500, 500) is 500 m away."""
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10_000, 0)])],
        crs="EPSG:5070",
    )
    path = tmp_path / "roads.gpkg"
    roads.to_file(path, driver="GPKG")
    return path


def test_distance_to_nearest_km(roads_gpkg):
    extractor = SpatialFeatureExtractor(roads_path=str(roads_gpkg))
    # Road runs (0,0)-(10000,0) in EPSG:5070; point 500 m north of midpoint
    lon, lat = (
        gpd.GeoSeries([Point(5000, 500)], crs="EPSG:5070")
        .to_crs("EPSG:4326")
        .iloc[0]
        .coords[0]
    )

    dist_km = extractor.distance_to_nearest_km(lon, lat)
    assert dist_km == pytest.approx(0.5, rel=0.05)


def test_extract_spatial_batch(roads_gpkg):
    roads = gpd.read_file(roads_gpkg)
    mid = roads.geometry.iloc[0].interpolate(0.5, normalized=True)
    mid_wgs84 = (
        gpd.GeoSeries([mid], crs="EPSG:5070").to_crs("EPSG:4326").iloc[0]
    )

    fires = gpd.GeoDataFrame(
        {
            "OBJECTID": [10, 11],
            "centroid_lon": [mid_wgs84.x, mid_wgs84.x],
            "centroid_lat": [mid_wgs84.y, mid_wgs84.y + 0.01],
            "geometry": [
                Point(mid_wgs84.x, mid_wgs84.y),
                Point(mid_wgs84.x, mid_wgs84.y + 0.01),
            ],
        },
        crs="EPSG:4326",
    )

    extractor = SpatialFeatureExtractor(roads_path=str(roads_gpkg))
    result = extractor.extract_spatial_batch(fires, batch_size=1)

    assert len(result) == 2
    assert "infrastructure_distance_km" in result.columns
    assert result.iloc[0]["distance_to_roads_km"] == pytest.approx(0.0, abs=0.01)
    assert result.iloc[1]["distance_to_roads_km"] > result.iloc[0]["distance_to_roads_km"]


def test_extract_spatial_batch_dedupes_equidistant_roads(tmp_path):
    """Parallel roads at equal distance should yield one row per fire."""
    roads = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (10_000, 0)]),
            LineString([(0, 10), (10_000, 10)]),
        ],
        crs="EPSG:5070",
    )
    roads_path = tmp_path / "parallel_roads.gpkg"
    roads.to_file(roads_path, driver="GPKG")

    lon, lat = (
        gpd.GeoSeries([Point(5000, 5)], crs="EPSG:5070")
        .to_crs("EPSG:4326")
        .iloc[0]
        .coords[0]
    )
    fires = gpd.GeoDataFrame(
        {
            "OBJECTID": [99],
            "centroid_lon": [lon],
            "centroid_lat": [lat],
            "geometry": [Point(lon, lat)],
        },
        crs="EPSG:4326",
    )

    extractor = SpatialFeatureExtractor(roads_path=str(roads_path))
    result = extractor.extract_spatial_batch(fires)
    assert len(result) == 1
    assert result.iloc[0]["distance_to_roads_km"] == pytest.approx(0.005, abs=0.001)


def test_load_roads_missing_file(tmp_path):
    extractor = SpatialFeatureExtractor(roads_path=str(tmp_path / "missing.gpkg"))
    with pytest.raises(FileNotFoundError, match="Run: python src/data/download_roads.py"):
        extractor.load_roads()
