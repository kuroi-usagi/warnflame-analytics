"""YAML configuration loader."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path: Path to YAML file (e.g. config/pipeline_config.yaml).

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
