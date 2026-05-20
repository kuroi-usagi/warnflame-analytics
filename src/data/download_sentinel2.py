"""
Build median Sentinel-2 NDVI/NDMI composites for California via Planetary Computer.

Requires optional deps: pip install -r requirements-sentinel2.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import rasterio.errors

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

CALIFORNIA_BBOX = (-124.5, 32.5, -114.0, 42.0)
DEFAULT_NDVI_PATH = "cache/sentinel2_ndvi_ca.tif"
DEFAULT_NDMI_PATH = "cache/sentinel2_ndmi_ca.tif"
DEFAULT_STACK_EPSG = 5070
DEFAULT_MAX_SCENES = 40
DEFAULT_BATCH_SIZE = 10
RASTER_NODATA = -9999.0


def _select_scenes_by_cloud(items: list, max_scenes: Optional[int]) -> list:
    """Keep globally lowest-cloud scenes (poor statewide coverage on large bboxes)."""
    if not max_scenes or len(items) <= max_scenes:
        return items
    return sorted(
        items,
        key=lambda item: item.properties.get("eo:cloud_cover", 100),
    )[:max_scenes]


def _select_scenes_spatially(
    items: list,
    max_scenes: Optional[int],
    bbox: tuple[float, float, float, float],
    grid_size: int = 6,
) -> list:
    """
    Pick scenes across a lat/lon grid so footprints spread over the bbox.

    Avoids choosing 40 scenes from the same corner of California.
    """
    if not max_scenes or len(items) <= max_scenes:
        return items

    west, south, east, north = bbox
    lon_step = (east - west) / grid_size
    lat_step = (north - south) / grid_size
    cells: dict[tuple[int, int], list] = {}

    for item in items:
        ib = item.bbox
        cx = (ib[0] + ib[2]) / 2
        cy = (ib[1] + ib[3]) / 2
        ix = min(grid_size - 1, max(0, int((cx - west) / lon_step)))
        iy = min(grid_size - 1, max(0, int((cy - south) / lat_step)))
        cells.setdefault((ix, iy), []).append(item)

    per_cell = max(1, max_scenes // max(len(cells), 1))
    selected: list = []
    seen: set[int] = set()

    for cell_items in cells.values():
        ranked = sorted(
            cell_items,
            key=lambda item: item.properties.get("eo:cloud_cover", 100),
        )
        for item in ranked[:per_cell]:
            item_id = id(item)
            if item_id not in seen:
                selected.append(item)
                seen.add(item_id)

    if len(selected) < max_scenes:
        for item in sorted(
            items,
            key=lambda item: item.properties.get("eo:cloud_cover", 100),
        ):
            if id(item) in seen:
                continue
            selected.append(item)
            seen.add(id(item))
            if len(selected) >= max_scenes:
                break

    return sorted(
        selected[:max_scenes],
        key=lambda item: item.properties.get("eo:cloud_cover", 100),
    )


def _select_scenes(
    items: list,
    max_scenes: Optional[int],
    bbox: tuple[float, float, float, float],
    strategy: str = "spatial",
    grid_size: int = 6,
) -> list:
    if strategy == "cloud":
        return _select_scenes_by_cloud(items, max_scenes)
    return _select_scenes_spatially(items, max_scenes, bbox, grid_size=grid_size)


def _median_composite_from_items(
    items: list[Any],
    bbox: tuple[float, float, float, float],
    resolution: int,
    stack_epsg: int,
    stackstac: Any,
    batch_size: int,
) -> tuple[Any, Any]:
    """
    Stack scenes in batches; skip failed batches; median across batch medians.
    """
    import xarray as xr

    ndvi_batches: list[Any] = []
    ndmi_batches: list[Any] = []
    n_batches = (len(items) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(items), batch_size):
        batch = items[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        logger.info(
            "Batch %s/%s: stacking %s scenes at %sm...",
            batch_num,
            n_batches,
            len(batch),
            resolution,
        )
        try:
            stack = stackstac.stack(
                batch,
                assets=["B04", "B08", "B11"],
                bounds_latlon=bbox,
                resolution=resolution,
                epsg=stack_epsg,
                chunksize=2048,
                errors_as_nodata=(rasterio.errors.RasterioIOError,),
            )
            red = stack.sel(band="B04").astype("float32")
            nir = stack.sel(band="B08").astype("float32")
            swir = stack.sel(band="B11").astype("float32")

            ndvi = (nir - red) / (nir + red)
            ndmi = (nir - swir) / (nir + swir)
            ndvi_batch = ndvi.median(dim="time", skipna=True)
            ndmi_batch = ndmi.median(dim="time", skipna=True)

            if hasattr(ndvi_batch, "compute"):
                ndvi_batch = ndvi_batch.compute()
                ndmi_batch = ndmi_batch.compute()

            ndvi_batches.append(ndvi_batch)
            ndmi_batches.append(ndmi_batch)
            logger.info("Batch %s/%s complete", batch_num, n_batches)

        except Exception as exc:
            logger.warning(
                "Batch %s/%s failed (%s) — skipping batch",
                batch_num,
                n_batches,
                exc,
            )

    if not ndvi_batches:
        raise RuntimeError(
            "All scene batches failed (network/read errors). "
            "Retry with --quick or fewer scenes: --max-scenes 15 --batch-size 5"
        )

    if len(ndvi_batches) == 1:
        return ndvi_batches[0], ndmi_batches[0]

    logger.info("Merging %s successful batches into statewide median...", len(ndvi_batches))
    ndvi_median = xr.concat(ndvi_batches, dim="batch").median(dim="batch", skipna=True)
    ndmi_median = xr.concat(ndmi_batches, dim="batch").median(dim="batch", skipna=True)
    return ndvi_median, ndmi_median


def download_sentinel2_composite(
    ndvi_path: str = DEFAULT_NDVI_PATH,
    ndmi_path: Optional[str] = None,
    year: int = 2020,
    months: Optional[tuple[int, ...]] = None,
    bbox: tuple[float, float, float, float] = CALIFORNIA_BBOX,
    cloud_threshold: int = 20,
    resolution: int = 250,
    max_scenes: Optional[int] = DEFAULT_MAX_SCENES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    stack_epsg: int = DEFAULT_STACK_EPSG,
    scene_selection: str = "spatial",
    config_path: str = "config/pipeline_config.yaml",
) -> tuple[Path, Path]:
    """
    Download Sentinel-2 L2A scenes and write median NDVI + NDMI GeoTIFFs.

    Processes scenes in small batches so one bad COG read does not abort the full run.
    """
    try:
        import planetary_computer as pc
        import pystac_client
        import stackstac
    except ImportError as exc:
        raise ImportError(
            "Sentinel-2 download requires planetary-computer, pystac-client, and stackstac. "
            "Install with: pip install -r requirements-sentinel2.txt"
        ) from exc

    config = load_config(config_path)
    s2_cfg = config.get("data", {}).get("sentinel2", {})

    months = months or tuple(s2_cfg.get("composite_months", [6, 7, 8, 9]))
    cloud_threshold = cloud_threshold or s2_cfg.get("cloud_threshold", 20)
    resolution = resolution or s2_cfg.get("resolution", 250)
    max_scenes = max_scenes if max_scenes is not None else s2_cfg.get("max_scenes", DEFAULT_MAX_SCENES)
    batch_size = batch_size or s2_cfg.get("batch_size", DEFAULT_BATCH_SIZE)
    stack_epsg = stack_epsg or s2_cfg.get("stack_epsg", DEFAULT_STACK_EPSG)
    scene_selection = scene_selection or s2_cfg.get("scene_selection", "spatial")

    ndvi_out = Path(ndvi_path or s2_cfg.get("ndvi_path", DEFAULT_NDVI_PATH))
    ndmi_out = Path(ndmi_path or s2_cfg.get("ndmi_path", DEFAULT_NDMI_PATH))
    ndvi_out.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Searching Sentinel-2 for %s, months %s, cloud_cover < %s",
        year,
        months,
        cloud_threshold,
    )

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    start = f"{year}-{min(months):02d}-01"
    end = f"{year}-{max(months):02d}-28"
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cloud_threshold}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for {year} months {months}")

    total_found = len(items)
    items = _select_scenes(items, max_scenes, bbox=bbox, strategy=scene_selection)
    if len(items) < total_found:
        logger.info(
            "Using %s/%s scenes (max_scenes=%s, selection=%s)",
            len(items),
            total_found,
            max_scenes,
            scene_selection,
        )

    logger.info(
        "Building composite: %s scenes, batch_size=%s, %sm, EPSG:%s",
        len(items),
        batch_size,
        resolution,
        stack_epsg,
    )

    ndvi_median, ndmi_median = _median_composite_from_items(
        items,
        bbox=bbox,
        resolution=resolution,
        stack_epsg=stack_epsg,
        stackstac=stackstac,
        batch_size=batch_size,
    )

    logger.info("Writing GeoTIFFs...")
    import rioxarray  # noqa: F401 — registers .rio accessor on xarray objects

    crs = f"EPSG:{stack_epsg}"
    if ndvi_median.rio.crs is None:
        ndvi_median = ndvi_median.rio.write_crs(crs)
    if ndmi_median.rio.crs is None:
        ndmi_median = ndmi_median.rio.write_crs(crs)

    import numpy as np

    ndvi_median = ndvi_median.where(np.isfinite(ndvi_median), RASTER_NODATA)
    ndmi_median = ndmi_median.where(np.isfinite(ndmi_median), RASTER_NODATA)
    ndvi_median = ndvi_median.rio.write_nodata(RASTER_NODATA)
    ndmi_median = ndmi_median.rio.write_nodata(RASTER_NODATA)

    ndvi_median.rio.to_raster(ndvi_out)
    ndmi_median.rio.to_raster(ndmi_out)
    logger.info("Saved NDVI to %s, NDMI to %s", ndvi_out, ndmi_out)
    return ndvi_out, ndmi_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 NDVI/NDMI median composite for California"
    )
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--ndvi-path", default=DEFAULT_NDVI_PATH)
    parser.add_argument("--ndmi-path", default=DEFAULT_NDMI_PATH)
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Stack resolution in meters (default from config, typically 250)",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Max scenes to stack (default from config; CA search finds thousands)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Scenes per stack batch (default 10; smaller = more resilient)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast test run: 15 scenes, batch 5, 500 m (~10–20 min)",
    )
    args = parser.parse_args()

    max_scenes = args.max_scenes
    batch_size = args.batch_size
    resolution = args.resolution or 250
    if args.quick:
        max_scenes = max_scenes or 15
        batch_size = batch_size or 5
        resolution = 500 if args.resolution is None else resolution
        logger.info(
            "Quick mode: max_scenes=%s, batch_size=%s, resolution=%sm",
            max_scenes,
            batch_size,
            resolution,
        )

    download_sentinel2_composite(
        ndvi_path=args.ndvi_path,
        ndmi_path=args.ndmi_path,
        year=args.year,
        resolution=resolution,
        max_scenes=max_scenes,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
