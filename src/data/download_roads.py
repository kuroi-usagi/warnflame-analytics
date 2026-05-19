"""
Download California primary roads (TIGER/Line) for spatial feature extraction.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import requests

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# TIGER folder is PRISECROADS (primary + secondary), not PRIMARYROADS.
DEFAULT_CA_PRIMARY_ROADS_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2024/PRISECROADS/tl_2024_06_prisecroads.zip"
)
FALLBACK_CA_PRIMARY_ROADS_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/PRISECROADS/tl_2023_06_prisecroads.zip"
)
DEFAULT_OUTPUT_PATH = "data/raw/california_roads.gpkg"
DEFAULT_TARGET_CRS = "EPSG:5070"


def download_roads(
    output_path: str = DEFAULT_OUTPUT_PATH,
    url: str = DEFAULT_CA_PRIMARY_ROADS_URL,
    target_crs: str = DEFAULT_TARGET_CRS,
    config_path: str = "config/pipeline_config.yaml",
) -> gpd.GeoDataFrame:
    """
    Download and save California primary roads as GeoPackage in a projected CRS.

    Args:
        output_path: Output GeoPackage path.
        url: TIGER/Line ZIP URL.
        target_crs: CRS for stored geometries (meters for distance queries).
        config_path: Optional YAML config override for defaults.

    Returns:
        GeoDataFrame of road geometries.
    """
    config = load_config(config_path)
    roads_cfg = config.get("data", {}).get("roads", {})

    output = Path(output_path or roads_cfg.get("path", DEFAULT_OUTPUT_PATH))
    url = url or roads_cfg.get("tiger_url", DEFAULT_CA_PRIMARY_ROADS_URL)
    target_crs = target_crs or roads_cfg.get("crs", DEFAULT_TARGET_CRS)

    output.parent.mkdir(parents=True, exist_ok=True)
    extract_dir = output.parent / "tiger_roads_tmp"

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading roads from %s", url)
    response = requests.get(url, timeout=600)
    if response.status_code == 404 and url != FALLBACK_CA_PRIMARY_ROADS_URL:
        logger.warning("Primary URL returned 404, trying fallback")
        url = FALLBACK_CA_PRIMARY_ROADS_URL
        response = requests.get(url, timeout=600)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        shp_name = next(n for n in zf.namelist() if n.endswith(".shp"))
        zf.extractall(extract_dir)

    shp_path = extract_dir / shp_name
    roads = gpd.read_file(shp_path).to_crs(target_crs)
    roads.to_file(output, driver="GPKG")
    logger.info("Saved %s road segments to %s", len(roads), output)

    shutil.rmtree(extract_dir, ignore_errors=True)
    return roads


def main() -> None:
    parser = argparse.ArgumentParser(description="Download California primary roads")
    parser.add_argument("--output", default=None, help="Output GeoPackage path")
    parser.add_argument("--url", default=None, help="TIGER/Line ZIP URL override")
    args = parser.parse_args()

    download_roads(
        output_path=args.output or DEFAULT_OUTPUT_PATH,
        url=args.url or DEFAULT_CA_PRIMARY_ROADS_URL,
    )


if __name__ == "__main__":
    main()
