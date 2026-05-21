#!/usr/bin/env bash
# Warnflame analytics — end-to-end pipeline runner
#
# Usage:
#   ./scripts/run_full_pipeline.sh
#
# Skip slow steps (reuse cached interim data):
#   SKIP_DOWNLOAD=1 SKIP_WEATHER=1 SKIP_TERRAIN=1 SKIP_VEGETATION=1 ./scripts/run_full_pipeline.sh
#
# Fast ML path (fires-only matrix + synthetic negatives for training):
#   SKIP_DOWNLOAD=1 SKIP_WEATHER=1 SKIP_TERRAIN=1 SKIP_VEGETATION=1 \
#   BUILD_SKIP_PSEUDO=1 TRAIN_SYNTHETIC=1 ./scripts/run_full_pipeline.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

if [[ -d "venv/bin" ]]; then
  # shellcheck source=/dev/null
  source "venv/bin/activate"
fi

log() { echo "[pipeline] $*"; }

run() {
  log "$1"
  eval "$2"
}

# --- Optional downloads / features ---
if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  run "Download CAL FIRE perimeters" \
    "python src/data/download_calfire.py --min-year 2000 --max-year 2024"
fi

run "Validate fires and write centroids" \
  "python src/data/validate_data.py"

if [[ "${SKIP_WEATHER:-0}" != "1" ]]; then
  run "Extract gridMET weather features" \
    "python src/features/weather_features.py --resume data/interim/fires_weather_joined.parquet"
fi

if [[ "${SKIP_TERRAIN:-0}" != "1" ]]; then
  TERRAIN_MODE="${TERRAIN_MODE:-patch}"
  run "Extract 3DEP terrain (${TERRAIN_MODE})" \
    "PYTHONWARNINGS='ignore::FutureWarning' python src/features/terrain_features.py --mode ${TERRAIN_MODE} --batch-size 100 --resume data/interim/fires_terrain_joined.parquet --output data/interim/fires_terrain_joined.parquet"
fi

if [[ "${SKIP_ROADS:-0}" != "1" ]]; then
  run "Download California roads (TIGER)" \
    "python src/data/download_roads.py"
fi

if [[ "${SKIP_SPATIAL:-0}" != "1" ]]; then
  run "Extract spatial (road distance) features" \
    "python src/features/spatial_features.py --resume data/interim/fires_spatial_joined.parquet"
fi

if [[ "${SKIP_VEGETATION:-0}" != "1" ]]; then
  run "Extract Sentinel-2 vegetation features" \
    "PYTHONWARNINGS='ignore::FutureWarning' python src/features/vegetation_features.py --resume data/interim/fires_vegetation_joined.parquet --output data/interim/fires_vegetation_joined.parquet"
fi

# --- Feature matrix ---
BUILD_ARGS=""
if [[ "${BUILD_SKIP_PSEUDO:-0}" == "1" ]]; then
  BUILD_ARGS="--skip-pseudo"
fi
run "Build feature matrix" \
  "python src/features/build_features.py ${BUILD_ARGS}"

# --- Modeling ---
run "Assign spatial CV groups" \
  "python src/models/spatial_cv.py"

TRAIN_ARGS=""
if [[ "${TRAIN_SYNTHETIC:-0}" == "1" ]]; then
  TRAIN_ARGS="--synthetic-negatives"
fi
run "Train Random Forest" \
  "python src/models/train_model.py ${TRAIN_ARGS}"

run "Export risk_weights.json" \
  "python src/models/export_weights.py"

# --- Reporting ---
if [[ "${SKIP_PLOTS:-0}" != "1" ]]; then
  run "Performance plots" \
    "python src/visualization/performance_plots.py"
  run "SHAP summary plot" \
    "python src/visualization/shap_plots.py --n-samples ${SHAP_SAMPLES:-500}"
fi

log "Pipeline complete."
log "Outputs: data/processed/feature_matrix.parquet, models/risk_weights.json, reports/figures/"
