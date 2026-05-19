"""Tests for src.utils.logger."""

import logging

import pytest
import yaml

from src.utils import logger as logger_module


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Isolate tests: re-run setup_logging on each test."""
    logger_module._CONFIGURED = False
    logging.root.handlers.clear()
    yield
    logger_module._CONFIGURED = False
    logging.root.handlers.clear()


def test_get_logger_returns_named_logger(tmp_path):
    log_file = tmp_path / "test.log"
    config_path = tmp_path / "logging.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "standard": {
                        "format": "%(name)s - %(levelname)s - %(message)s",
                    }
                },
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "level": "DEBUG",
                        "formatter": "standard",
                        "stream": "ext://sys.stdout",
                    },
                    "file": {
                        "class": "logging.FileHandler",
                        "level": "DEBUG",
                        "formatter": "standard",
                        "filename": str(log_file),
                        "encoding": "utf-8",
                    },
                },
                "root": {"level": "DEBUG", "handlers": ["console", "file"]},
            }
        ),
        encoding="utf-8",
    )

    logger_module.setup_logging(config_path)
    test_logger = logger_module.get_logger("tests.logger")

    assert test_logger.name == "tests.logger"
    test_logger.info("pipeline started")
    assert log_file.read_text(encoding="utf-8").count("pipeline started") == 1


def test_get_logger_uses_default_config():
    logger_module.setup_logging()
    test_logger = logger_module.get_logger("tests.default")

    assert test_logger.name == "tests.default"
    assert logging.getLogger().handlers
