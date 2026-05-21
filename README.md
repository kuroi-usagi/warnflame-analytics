# WarnFlame Analytics Engine

Machine learning validation system for the [warnflame](https://github.com/Jennifer-Werner/warnflame) drone-based wildfire risk assessment platform.

Calibrates risk factor weights using historical California fire perimeters (CAL FIRE), gridMET weather, USGS terrain, and spatial cross-validation (Random Forest + GroupKFold).

## Status

Early development — see [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) for current work and [docs/ARCHITECTURE_SPEC.md](docs/ARCHITECTURE_SPEC.md) for the full system specification.

## Quick start

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

### Full pipeline (reuse cached features)

```bash
chmod +x scripts/run_full_pipeline.sh
SKIP_DOWNLOAD=1 SKIP_WEATHER=1 SKIP_TERRAIN=1 SKIP_VEGETATION=1 \
BUILD_SKIP_PSEUDO=1 TRAIN_SYNTHETIC=1 ./scripts/run_full_pipeline.sh
```

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) and [docs/INTEGRATION.md](docs/INTEGRATION.md).

## Repository layout

```
config/          Pipeline and logging YAML
src/             Analytics pipeline modules
data/            Raw, interim, and processed data (not in git)
models/          Trained models and risk_weights.json
docs/            Architecture spec and integration guides
tests/           Unit tests
```

## Development workflow

1. Read `DEVELOPMENT_LOG.md` before coding or debugging.
2. Implement one logical unit (function/class group) per commit.
3. Update `DEVELOPMENT_LOG.md` after each attempt.
4. Push to `main` frequently.

## License

MIT
