# Data sources

External datasets used by the warnflame-analytics pipeline.

## CAL FIRE Fire Perimeters (Firep24_1)

| | |
|--|--|
| **Module** | `src/data/download_calfire.py` |
| **Output** | `data/raw/calfire_perimeters.gpkg` |
| **Coverage** | California wildfire perimeters, 2000–2024 (configurable) |
| **Filters** | GPS/imagery mapping, ≥10 acres |
| **Notes** | Live ArcGIS REST API; raw file is gitignored |

## gridMET (weather)

| | |
|--|--|
| **Module** | `src/features/weather_features.py` |
| **Library** | [pygridmet](https://github.com/hyriver/pygridmet) |
| **Output** | `data/interim/fires_weather_joined.parquet` |
| **Variables** | tmmx, tmmn, pr, vs, erc, bi |
| **Windows** | 7, 14, 30 days pre-ignition |
| **Resolution** | ~4 km daily |
| **Notes** | ~13 fires may fail at offshore centroids; retry with `--resume` |

## USGS 3DEP (terrain)

| | |
|--|--|
| **Module** | `src/features/terrain_features.py` |
| **Library** | [py3dep](https://github.com/hyriver/py3dep) |
| **Output** | `data/interim/fires_terrain_joined.parquet` |
| **Fields** | elevation_meters, slope_degrees, aspect_south_factor |
| **Modes** | `patch` (per-fire API, low memory) or `mosaic` (statewide DEM — high memory) |
| **Service** | `elevation.nationalmap.gov` — subject to 502 outages; use `--resume` |

## TIGER roads (spatial)

| | |
|--|--|
| **Module** | `src/data/download_roads.py`, `src/features/spatial_features.py` |
| **Source** | Census TIGER 2024 primary roads (California) |
| **Output** | `data/raw/california_roads.gpkg`, `data/interim/fires_spatial_joined.parquet` |
| **Fields** | distance_to_roads_km, infrastructure_distance_km |

## Sentinel-2 / Planetary Computer (vegetation)

| | |
|--|--|
| **Module** | `src/data/download_sentinel2.py`, `src/features/vegetation_features.py` |
| **Output** | `cache/sentinel2_ndvi_ca.tif`, `data/interim/fires_vegetation_joined.parquet` |
| **Fields** | ndvi_mean, ndmi_mean, vegetation_density |
| **Optional deps** | `pip install -r requirements-sentinel2.txt` |
| **Notes** | Pre-2015 fires use defaults when no S2 scenes exist |

## Processed training table

| | |
|--|--|
| **Module** | `src/features/build_features.py` |
| **Output** | `data/processed/feature_matrix.parquet` |
| **Labels** | `is_fire` (1 = fire perimeter, 0 = pseudo-absence) |

## Cache directories

| Path | Contents |
|------|----------|
| `cache/gridmet.db` | HyRiver gridMET HTTP cache |
| `cache/sentinel2_*.tif` | NDVI/NDMI composites |
| `cache/ca_dem_*.tif` | Optional statewide DEM (mosaic mode) |

All under `cache/` and `data/` — **not committed to git** (see `.gitignore`).
