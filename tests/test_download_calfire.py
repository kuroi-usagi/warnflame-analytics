"""Tests for src.data.download_calfire."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from src.data.download_calfire import CALFIREDownloader

SAMPLE_GEOJSON = """{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "OBJECTID": 1,
        "YEAR_": 2020,
        "GIS_ACRES": 150.0,
        "C_METHOD": 1
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-121.0, 38.0], [-120.9, 38.0], [-120.9, 38.1], [-121.0, 38.1], [-121.0, 38.0]]]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "OBJECTID": 2,
        "YEAR_": 1990,
        "GIS_ACRES": 5.0,
        "C_METHOD": 6
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-122.0, 37.0], [-121.9, 37.0], [-121.9, 37.1], [-122.0, 37.1], [-122.0, 37.0]]]
      }
    }
  ]
}"""


@pytest.fixture
def mock_geojson_response():
    response = MagicMock()
    response.text = SAMPLE_GEOJSON
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def downloader(tmp_path):
    return CALFIREDownloader(
        output_dir=tmp_path / "raw",
        min_year=2000,
        max_year=2024,
    )


def test_init_sets_year_range(downloader):
    assert downloader.min_year == 2000
    assert downloader.max_year == 2024
    assert downloader.output_dir.exists()


@patch("src.data.download_calfire.requests.get")
def test_download_fires_saves_geopackage(
    mock_get,
    downloader,
    mock_geojson_response,
    tmp_path,
):
    empty = MagicMock()
    empty.text = '{"type":"FeatureCollection","features":[]}'
    empty.raise_for_status = MagicMock()

    # First page returns data; second page (OBJECTID pagination) is empty
    mock_get.side_effect = [mock_geojson_response, empty]

    fires = downloader.download_fires(output_filename="test_fires.gpkg")

    assert len(fires) == 2
    output_path = tmp_path / "raw" / "test_fires.gpkg"
    assert output_path.exists()

    saved = gpd.read_file(output_path)
    assert len(saved) == 2


def test_filter_by_quality(downloader):
    fires = gpd.GeoDataFrame(
        {
            "OBJECTID": [1, 2, 3],
            "YEAR_": [2020, 2019, 2018],
            "GIS_ACRES": [150.0, 5.0, 50.0],
            "C_METHOD": [1, 6, 4],  # GPS Ground, Hand Drawn, Other Imagery
            "geometry": [
                Polygon([(-121, 38), (-120.9, 38), (-120.9, 38.1), (-121, 38.1)]),
                Polygon([(-122, 37), (-121.9, 37), (-121.9, 37.1), (-122, 37.1)]),
                Polygon([(-123, 36), (-122.9, 36), (-122.9, 36.1), (-123, 36.1)]),
            ],
        },
        crs="EPSG:4326",
    )

    filtered = downloader.filter_by_quality(fires)

    assert len(filtered) == 2
    assert set(filtered["C_METHOD"]) == {1, 4}
    assert all(filtered["GIS_ACRES"] >= 10.0)


@patch("src.data.download_calfire.requests.get")
def test_download_fires_raises_when_empty(mock_get, downloader):
    empty = MagicMock()
    empty.text = '{"type":"FeatureCollection","features":[]}'
    empty.raise_for_status = MagicMock()
    mock_get.return_value = empty

    with pytest.raises(ValueError, match="No fires found"):
        downloader.download_fires()
