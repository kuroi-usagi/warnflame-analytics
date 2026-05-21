# Warnflame integration guide

This document describes how to use outputs from **warnflame-analytics** in the operational **warnflame** risk scoring system.

## Primary deliverable

**File:** `models/risk_weights.json`

**Schema (weights sum to 1.0 ± 0.01):**

| Key | Description |
|-----|-------------|
| `vegetation_density` | Normalized vegetation signal (0–1) |
| `slope_degrees` | Terrain slope importance |
| `aspect_south_factor` | South-facing aspect exposure |
| `elevation_meters` | Elevation importance |
| `infrastructure_distance_km` | Distance to roads / infrastructure |
| `weather_erc_max` | Optional fire-danger (ERC) weather factor |

Example:

```json
{
  "vegetation_density": 0.15,
  "slope_degrees": 0.20,
  "aspect_south_factor": 0.10,
  "elevation_meters": 0.25,
  "infrastructure_distance_km": 0.20,
  "weather_erc_max": 0.10
}
```

## Regenerating weights

```bash
source venv/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# After feature matrix + spatial CV + training:
python src/models/train_model.py --synthetic-negatives   # fast smoke path
# or
python src/models/train_model.py                         # requires pseudo-absences in matrix

python src/models/export_weights.py
```

Copy `models/risk_weights.json` into the warnflame deployment config path.

## Mapping to warnflame risk score

Operational warnflame combines factors into a 0–100 score. Use exported weights as **relative importance multipliers** on each normalized sub-score:

```text
risk_score = 100 * (
    w_veg   * vegetation_component +
    w_slope * slope_component +
    w_aspect * aspect_component +
    w_elev  * elevation_component +
    w_infra * infrastructure_component +
    w_erc   * weather_component   # optional
)
```

Ensure on-site components are scaled to `[0, 1]` before applying weights.

## Model artifact (optional)

**File:** `models/random_forest.joblib`

Contains:

- Fitted `RandomForestClassifier`
- `SimpleImputer` for missing weather/terrain values
- List of `feature_columns` used in training

Load for offline analysis or SHAP — not required for production scoring if using `risk_weights.json` only.

## Validation reports

| File | Purpose |
|------|---------|
| `models/training_metrics.json` | Spatial CV ROC-AUC, temporal 2024 holdout |
| `models/spatial_cv_results.json` | K-means group separation report |
| `reports/figures/*.png` | CV scores, weights bar chart, SHAP summary |

## Production-quality training

For weights that reflect real fire vs non-fire contrast (especially vegetation):

1. Build matrix **with** pseudo-absences (slow — gridMET for ~3,700 extra points):

   ```bash
   python src/features/build_features.py
   ```

2. Re-run spatial CV and training **without** `--synthetic-negatives`.

3. Re-export weights.

## Versioning

Record in warnflame:

- Git commit hash of warnflame-analytics
- Date of `risk_weights.json` generation
- Whether `synthetic_negatives_used` is true in `training_metrics.json`
