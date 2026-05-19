# WarnFlame Analytics — Architecture Specification

This document is the stable design reference for the warnflame-analytics pipeline.

**Source:** L-99 artifact specification (v1.0.0), integrated from project planning.

**Operational log:** See [DEVELOPMENT_LOG.md](../DEVELOPMENT_LOG.md) for implementation attempts, failures, and forbidden patches.

## Overview

Geospatial ML pipeline that:

1. Ingests CAL FIRE historical fire perimeters (Firep24_1)
2. Extracts pre-fire weather (gridMET), terrain (USGS 3DEP), and optional vegetation (Sentinel-2)
3. Trains Random Forest with spatial GroupKFold cross-validation
4. Exports normalized feature importances as `models/risk_weights.json` for the operational warnflame system

## Key modules (implementation order)

| Phase | Module | Purpose |
|-------|--------|---------|
| 1 | `src/utils/logger.py` | Structured logging |
| 2 | `src/utils/config_loader.py` | YAML config loading |
| 3 | `src/data/download_calfire.py` | CAL FIRE perimeter download |
| 4 | `src/data/validate_data.py` | Fire record quality checks |
| 5 | `src/features/weather_features.py` | gridMET pre-fire windows |
| 6 | `src/features/terrain_features.py` | Slope, aspect, elevation |
| 7 | `src/features/spatial_features.py` | Roads, WUI distance |
| 8 | `src/features/build_features.py` | Feature matrix assembly |
| 9 | `src/models/spatial_cv.py` | K-means blocks + GroupKFold |
| 10 | `src/models/train_model.py` | Random Forest training |
| 11 | `src/models/export_weights.py` | `risk_weights.json` export |
| 12 | `src/visualization/shap_plots.py` | SHAP explainability |

## Integration contract

Output file: `models/risk_weights.json`

Required fields (sum to 1.0 ± 0.01):

- `vegetation_density`
- `slope_degrees`
- `aspect_south_factor`
- `elevation_meters`
- `infrastructure_distance_km`
- `weather_erc_max` (optional in operational fallback)

## Full specification

The complete L-99 specification (research foundations, mathematical frameworks, file-by-file implementations, and integration with warnflame) was provided at project kickoff. Paste or append the full text here when archiving a frozen copy in the repository.

For day-to-day development, use this file plus `config/pipeline_config.yaml` and module docstrings as the source of truth.
