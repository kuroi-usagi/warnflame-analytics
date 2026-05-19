"""Structured logging for the warnflame-analytics pipeline."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

_CONFIGURED = False
_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "logging_config.yaml"
)


def _ensure_log_directory(config_path: Path) -> None:
    """Create the log file parent directory if a file handler is configured."""
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    handlers = config.get("handlers") or {}
    for handler in handlers.values():
        if isinstance(handler, dict) and "filename" in handler:
            Path(handler["filename"]).parent.mkdir(parents=True, exist_ok=True)


def setup_logging(config_path: str | Path | None = None) -> None:
    """
    Configure root logging from YAML (``config/logging_config.yaml`` by default).

    Args:
        config_path: Optional path to a logging dictConfig YAML file.
    """
    global _CONFIGURED

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Logging config not found: {path}")

    _ensure_log_directory(path)

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    logging.config.dictConfig(config)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a module logger, configuring logging on first use.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
