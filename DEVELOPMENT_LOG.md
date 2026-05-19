# System Development & Debugging Log

## 1. Active Objective & Current State

- **Current Goal:** Chunk 3+ — `validate_data.py`, then feature engineering (weather, terrain, spatial).
- **Current System State:** Phase 0 complete; GitHub remote `git@github-tina:kuroi-usagi/warnflame-analytics.git` pushed. Logger and CAL FIRE downloader implemented; all 6 unit tests passing.

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

## 3. Blocked Roads & Forbidden Patches

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
- [ ] Live run: `python src/data/download_calfire.py --min-year 2000 --max-year 2024`
- [ ] `src/data/validate_data.py`
- [ ] Weather, terrain, spatial features
- [ ] Model training, export weights, SHAP plots
