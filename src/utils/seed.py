"""Seed utility for reproducible experiments across Python, NumPy, and system environments."""

from __future__ import annotations

import os
import random

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seed across Python's random module, NumPy, and environment.

    Args:
        seed: Random integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug("Configured global random seed to %d", seed)
