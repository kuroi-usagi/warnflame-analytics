"""Tests for src.data.download_roads."""

import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from src.data.download_roads import download_roads


@pytest.fixture
def tiger_zip_bytes(tmp_path: Path) -> bytes:
    """Build a minimal shapefile ZIP mimicking TIGER extract layout."""
    roads = gpd.GeoDataFrame(
        {"RTTYP": ["M"]},
        geometry=[LineString([(-121.0, 38.0), (-120.9, 38.0)])],
        crs="EPSG:4326",
    )
    shp_dir = tmp_path / "shp"
    shp_dir.mkdir()
    roads.to_file(shp_dir / "tl_2023_06_prisecroads.shp")

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for path in shp_dir.iterdir():
            zf.write(path, arcname=path.name)
    return buffer.getvalue()


@patch("src.data.download_roads.requests.get")
def test_download_roads_saves_geopackage(mock_get, tmp_path, tiger_zip_bytes):
    response = MagicMock()
    response.content = tiger_zip_bytes
    response.raise_for_status = MagicMock()
    mock_get.return_value = response

    output = tmp_path / "roads.gpkg"
    roads = download_roads(
        output_path=str(output),
        url="https://example.com/roads.zip",
        target_crs="EPSG:5070",
        config_path="config/pipeline_config.yaml",
    )

    assert len(roads) == 1
    assert output.exists()
    saved = gpd.read_file(output)
    assert saved.crs.to_string() == "EPSG:5070"
