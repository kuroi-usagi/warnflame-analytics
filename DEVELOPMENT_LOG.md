# System Development & Debugging Log

## 1. Active Objective & Current State

- **Current Goal:** Commit + push Chunk 9 (`build_features.py`); smoke test join + optional pseudo-absences.
- **Current System State:** Chunks 0–8 on `main` (`26edc26`). Chunk 9 implemented locally: feature matrix join + pseudo-absence sampling.

## 2. Comprehensive Implementation History

### [Attempt #1: Repository bootstrap]

- **Strategy:** PyCharm new project + `python -m venv` for local development environment.
- **Location:** `main.py`, `venv/`
- **Result:** PARTIAL
- **Failure Analysis:** N/A — environment only.
- **Lessons Learned:** Add root `.gitignore` before `git add .`.

### [Attempt #2: Phase 0 project skeleton]

- **Strategy:** `.gitignore`, directory tree, config YAMLs, `DEVELOPMENT_LOG.md`, initial commit.
- **Location:** Repository root, `config/`, `src/`, `docs/`, `data/`, `tests/`, `scripts/`
- **Result:** SUCCESS
- **Failure Analysis:** N/A
- **Lessons Learned:** Branch is `main` after first commit.

### [Attempt #3: GitHub push / SSH multi-account]

- **Strategy:** `git@github-tina:Jennifer-Werner/...` then `kuroi-usagi/...`
- **Location:** `git remote`, `~/.ssh/config`
- **Result:** SUCCESS (after remote URL fix)
- **Failure Analysis:**
  - `Permission denied to deploy key` — `github-personal` used `id_rsa` registered as deploy key on `Sunsvision/doppy-bot`.
  - `Repository not found` — remote pointed to `Jennifer-Werner` but `id_ed25519_jw` authenticates as `kuroi-usagi`.
- **Lessons Learned:** SSH host alias name ≠ GitHub account. Run `ssh -T git@github-tina` to see real user. Use `git@github-tina:kuroi-usagi/warnflame-analytics.git`.

### [Attempt #4: Structured logger]

- **Strategy:** `logging.config.dictConfig` from `config/logging_config.yaml`; `get_logger()` lazy-init.
- **Location:** `src/utils/logger.py`, `tests/test_logger.py`, `pytest.ini`
- **Result:** SUCCESS
- **Failure Analysis:** N/A
- **Lessons Learned:** Reset `_CONFIGURED` in tests to avoid handler leakage.

### [Attempt #5: CAL FIRE downloader]

- **Strategy:** ArcGIS FeatureServer pagination (2000/page); GeoPackage output; quality filter by acres + `C_METHOD`.
- **Location:** `src/data/download_calfire.py`, `tests/test_download_calfire.py`
- **Result:** SUCCESS (unit tests with mocked HTTP)
- **Failure Analysis:** N/A
- **Lessons Learned:** Default API: `California_Fire_Perimeters_All` on `services1.arcgis.com`. Live download not run in CI (network).

### [Attempt #6: Live CAL FIRE download HTTP 400]

- **Strategy:** Run `python src/data/download_calfire.py` against production API.
- **Location:** `src/data/download_calfire.py`
- **Result:** FAILURE → FIXED
- **Failure Analysis:** `400 Bad Request` — wrong FeatureServer URL (`jUJYIo9rS62FWOcc` / `California_Fire_Perimeters_All`). Correct public endpoint from data.ca.gov: `jUJYIo9tSA7EHvfZ` / `California_Historic_Fire_Perimeters`. `C_METHOD` is integer-coded (1=GPS Ground, 4=Other Imagery), not strings `GPS`/`IMAGERY`.
- **Lessons Learned:** Use OID pagination (`OBJECTID > last`) for GeoJSON; ~8371 fires for 2000–2024.

### [Attempt #7: Live download success]

- **Strategy:** Re-run downloader with fixed ArcGIS endpoint.
- **Location:** `data/raw/calfire_perimeters.gpkg`
- **Result:** SUCCESS — 8371 downloaded, 3768 after quality filter.
- **Failure Analysis:** N/A
- **Lessons Learned:** ~84MB GeoPackage; not tracked in git.

### [Attempt #8: validate_data.py]

- **Strategy:** Per-record checks (geometry, dates, acres, CA bounds); add `centroid_lon/lat`; write interim GPKG.
- **Location:** `src/data/validate_data.py`, `tests/test_validate_data.py`
- **Result:** SUCCESS — 3768 → 3722 valid (1.2% removed); exclusions mostly missing/invalid containment dates.
- **Failure Analysis:** N/A
- **Lessons Learned:** Compute centroids in EPSG:5070 then convert to WGS84. ArcGIS dates are epoch milliseconds (float).

### [Attempt #9: Bulk implementation rollback — Option A]

- **Strategy:** User chose Option A: keep only Chunk 5 (weather); delete uncommitted drafts for chunks 6–11.
- **Location:** Removed `terrain_features.py`, `spatial_features.py`, `vegetation_features.py`, `build_features.py`, `spatial_cv.py`, `train_model.py`, `download_roads.py`.
- **Result:** SUCCESS
- **Failure Analysis:** Prior agent session implemented multiple chunks at once without per-chunk commits — violates plan workflow.
- **Lessons Learned:** One chunk per commit; test and push before next chunk. Do not implement chunks 6+ until chunk 5 is on `main`.

### [Attempt #10: Chunk 5 weather_features.py]

- **Strategy:** `WeatherFeatureExtractor` with pygridmet, 7/14/30-day windows, resume checkpoint, mocked tests without requiring pygridmet at import.
- **Location:** `src/features/weather_features.py`, `tests/test_weather_features.py`
- **Result:** SUCCESS (15 tests pass)
- **Failure Analysis:** `patch("pygridmet.get_bycoords")` failed when pygridmet not installed — fixed with `patch.object(weather_module, "gridmet", mock)`.
- **Lessons Learned:** Full 3,722-fire run is slow; smoke test with `--limit 10` first.

### [Attempt #11: Chunk 5 smoke test — pip / parquet / column names]

- **Strategy:** Install `pygridmet` and `pyarrow`; run `--limit 10`; fix empty feature columns.
- **Location:** `src/features/weather_features.py`, `requirements.txt`, `tests/test_weather_features.py`
- **Result:** SUCCESS (10 fires, 40 columns in `fires_weather_joined.parquet`)
- **Failure Analysis:**
  - `pip install pygridmet   # if not installed` — shell treats `#` as package name; run `pip install pygridmet` on its own line.
  - `ImportError: pygridmet` — install never ran due to pip error above.
  - `ImportError` on `to_parquet` — missing `pyarrow`.
  - Parquet had only `OBJECTID` — pygridmet returns columns like `tmmx (K)`; code expected `tmmx`. Fixed with `_normalize_gridmet_weather()`.
- **Lessons Learned:** Do not put `#` comments on the same line as pip commands in copy-paste blocks.

### [Attempt #12: Chunk 6 terrain_features.py]

- **Strategy:** `TerrainFeatureExtractor` with mosaic mode (cached CA DEM via `py3dep.get_dem`, local central-difference slope/aspect) and patch mode (per-fire `get_map` with DEM + slope + aspect layers for smoke tests). Resume parquet checkpoint; mocked tests without py3dep at import.
- **Location:** `src/features/terrain_features.py`, `tests/test_terrain_features.py`
- **Result:** SUCCESS (21 tests pass; patch smoke 2 fires)
- **Failure Analysis:** N/A in dev; full CA mosaic download at 10 m is large — use `--mode patch` for smoke, mosaic for batch after one-time DEM cache.
- **Lessons Learned:** py3dep `get_map` returns `elevation`, `slope_degrees`, `aspect_degrees`; treat 255/32767 as nodata. Use `geo_crs=4326`, `crs=EPSG:5070`.

### [Attempt #13: Chunk 7 spatial_features + download_roads]

- **Strategy:** TIGER/Line CA primary roads download; `SpatialFeatureExtractor` with batched `gpd.sjoin_nearest` for segment-accurate `infrastructure_distance_km`; resume parquet; mocked ZIP download test.
- **Location:** `src/data/download_roads.py`, `src/features/spatial_features.py`, `tests/test_download_roads.py`, `tests/test_spatial_features.py`, `config/pipeline_config.yaml`
- **Result:** SUCCESS (25 tests pass)
- **Failure Analysis:** N/A
- **Lessons Learned:** Prefer `sjoin_nearest` over midpoint KD-tree for road distance. Download roads once before spatial extract.
- **Failure Analysis (roads download):** `TIGER2023/PRIMARYROADS/...` returns 404 — correct path is `PRISECROADS` (e.g. `TIGER2024/PRISECROADS/tl_2024_06_prisecroads.zip`).

### [Attempt #15: Chunk 9 build_features]

- **Strategy:** Left-join weather/terrain/spatial/vegetation parquets on `OBJECTID`; label `is_fire=1`; sample pseudo-absences in CA with min distance from fires; extract same features for negatives.
- **Location:** `src/features/build_features.py`, `config/pipeline_config.yaml`, `tests/test_build_features.py`
- **Result:** SUCCESS (4 unit tests; smoke `--limit 10 --skip-pseudo` → 10 rows × 52 columns)
- **Lessons Learned:** Use `--skip-pseudo` for fast join smoke tests; full pseudo run re-calls extractors (slow). Vegetation optional in `required_modalities`.

### [Attempt #14: Chunk 8 vegetation_features + download_sentinel2]

- **Strategy:** Planetary Computer median NDVI/NDMI composites; sample at centroids; `vegetation_density` for warnflame; fallback defaults when rasters missing; optional `requirements-sentinel2.txt`.
- **Location:** `src/data/download_sentinel2.py`, `src/features/vegetation_features.py`, tests, config
- **Result:** SUCCESS (unit tests with mock rasters)
- **Lessons Learned:** Full CA composite at 100 m is large — run download once; smoke test works with `--limit 10` + fallback until rasters exist.

## 3. Blocked Roads & Forbidden Patches

- Do not implement multiple chunks in one session without per-chunk commits.

- Do not use old URL `jUJYIo9rS62FWOcc/.../California_Fire_Perimeters_All` (returns 400).

- Do not commit `venv/`, large `data/raw/*.gpkg`, or `.idea/`.
- Do not use `github-personal` (`id_rsa`) for push — deploy key only.
- Do not use `Jennifer-Werner/` in remote unless key is registered on that account.
- Do not run `git add .` without checking for generated data files.

## 4. Pending Backlog & Next Logical Steps

- [x] Add `.gitignore`, `DEVELOPMENT_LOG.md`, directory skeleton
- [x] Initial commit + GitHub repo + push (`kuroi-usagi/warnflame-analytics`)
- [x] `src/utils/logger.py` + tests
- [x] `src/utils/config_loader.py`
- [x] `src/data/download_calfire.py` + tests
- [x] Live run: `python src/data/download_calfire.py --min-year 2000 --max-year 2024`
- [x] `src/data/validate_data.py`
- [x] `src/features/weather_features.py` (commit `9931916` on `main`)
- [x] Smoke: `python src/features/weather_features.py --limit 10` (after pygridmet + pyarrow + column fix)
- [x] Commit + push Chunk 5 fixes (`1caaf6f`)
- [ ] Full weather run: 3,722 fires with `--resume` (optional, slow)
- [x] Chunk 6: `terrain_features.py` on `main` (`d7b0602`)
- [x] Chunk 7: `spatial_features.py` + `download_roads.py` on `main` (`b143bb2`)
- [x] Chunk 8: `vegetation_features.py` + `download_sentinel2.py` on `main` (`26edc26`)
- [x] Chunk 9: `build_features.py` + tests (`ef370cc`)
- [x] Chunk 10: `spatial_cv.py` — K-means groups, leakage/separation report
- [ ] Smoke: full `feature_matrix.parquet` + `python src/models/spatial_cv.py`
- [ ] Chunks 11–14: train, export, viz, docs
