"""Tests for src.data.download_sentinel2."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from src.data.download_sentinel2 import (
    CALIFORNIA_BBOX,
    _select_scenes,
    _select_scenes_by_cloud,
    download_sentinel2_composite,
)


def _median_raster() -> xr.DataArray:
    """Minimal xarray raster for composite write path."""
    return xr.DataArray(
        np.array([[0.2]], dtype=np.float32),
        dims=["y", "x"],
    )


def test_select_scenes_limits_and_sorts_by_cloud_cover():
    items = [
        MagicMock(properties={"eo:cloud_cover": 30}),
        MagicMock(properties={"eo:cloud_cover": 5}),
        MagicMock(properties={"eo:cloud_cover": 15}),
    ]
    selected = _select_scenes_by_cloud(items, max_scenes=2)
    assert len(selected) == 2
    assert selected[0].properties["eo:cloud_cover"] == 5
    assert selected[1].properties["eo:cloud_cover"] == 15


def test_select_scenes_spatial_spreads_across_bbox():
    items = []
    for lon, cloud in [(-123.0, 5), (-123.0, 8), (-116.0, 6), (-116.0, 9)]:
        items.append(
            MagicMock(
                bbox=[lon - 0.5, 35.0, lon + 0.5, 36.0],
                properties={"eo:cloud_cover": cloud},
            )
        )
    selected = _select_scenes(items, max_scenes=2, bbox=CALIFORNIA_BBOX, strategy="spatial")
    assert len(selected) == 2
    lons = [(item.bbox[0] + item.bbox[2]) / 2 for item in selected]
    assert min(lons) < -120 and max(lons) > -118


def test_download_sentinel2_composite_writes_rasters(tmp_path):
    mock_item = MagicMock()
    mock_catalog = MagicMock()
    mock_catalog.search.return_value.items.return_value = [mock_item]
    mock_client_cls = MagicMock()
    mock_client_cls.open.return_value = mock_catalog

    band = _median_raster()
    stack = MagicMock()
    stack.sel.return_value = MagicMock(astype=MagicMock(return_value=band))
    stack.__sub__ = MagicMock(return_value=band)
    stack.__truediv__ = MagicMock(return_value=band)

    ndvi_out = tmp_path / "ndvi.tif"
    ndmi_out = tmp_path / "ndmi.tif"

    mock_pc = MagicMock()
    mock_pystac = MagicMock()
    mock_pystac.Client = mock_client_cls
    mock_stackstac = MagicMock()
    mock_stackstac.stack.return_value = stack

    with patch.dict(
        sys.modules,
        {
            "planetary_computer": mock_pc,
            "pystac_client": mock_pystac,
            "stackstac": mock_stackstac,
        },
    ), patch(
        "src.data.download_sentinel2._median_composite_from_items",
        return_value=(band, band),
    ):
        result = download_sentinel2_composite(
            ndvi_path=str(ndvi_out),
            ndmi_path=str(ndmi_out),
            year=2020,
            months=(6, 7),
            resolution=100,
            max_scenes=1,
            batch_size=1,
        )

    assert result == (ndvi_out, ndmi_out)
    assert ndvi_out.is_file()
    assert ndmi_out.is_file()


def test_download_sentinel2_import_error(tmp_path):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "planetary_computer":
            raise ImportError("no pc")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match="requirements-sentinel2"):
            download_sentinel2_composite(
                ndvi_path=str(tmp_path / "ndvi.tif"),
                ndmi_path=str(tmp_path / "ndmi.tif"),
            )
