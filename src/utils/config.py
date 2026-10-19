"""Configuration loader for the credit default prediction project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters. Returns an empty dict
        if the file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found at %s. Using default parameters.", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    logger.debug("Loaded configuration from %s", path)
    return config


def get_config_value(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """Safely traverse a nested dictionary using dot-separated keys.

    Args:
        config: Configuration dictionary.
        path: Dot-separated path string (e.g. ``'data.split.test_size'``).
        default: Default value returned if key path is missing.

    Returns:
        Resolved configuration value or ``default``.
    """
    keys = path.split(".")
    current: Any = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
