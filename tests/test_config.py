"""Tests for configuration loading and random seed utilities."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from src.utils.config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from src.utils.seed import set_seed


class TestConfig:
    """Tests for load_config and get_config_value."""

    def test_load_default_config_file(self):
        """configs/config.yaml should exist and have required top-level sections."""
        config = load_config(DEFAULT_CONFIG_PATH)
        assert isinstance(config, dict)
        assert len(config) > 0
        assert "seed" in config
        assert "data" in config
        assert "models" in config
        assert "evaluation" in config

    def test_load_missing_file_returns_empty_dict(self, tmp_path: Path):
        """Loading a non-existent config path gracefully returns an empty dict."""
        missing = tmp_path / "does_not_exist.yaml"
        config = load_config(missing)
        assert config == {}

    def test_get_config_value_nested_lookup(self):
        """get_config_value properly traverses nested dictionary keys."""
        sample = {"data": {"split": {"test_size": 0.2, "val_size": 0.1}}}
        assert get_config_value(sample, "data.split.test_size") == 0.2
        assert get_config_value(sample, "data.split.val_size") == 0.1

    def test_get_config_value_returns_default_when_missing(self):
        """Returns the fallback default value when a key does not exist."""
        sample = {"a": {"b": 1}}
        assert get_config_value(sample, "a.c", default=99) == 99
        assert get_config_value(sample, "x.y.z", default="fallback") == "fallback"


class TestSeed:
    """Tests for set_seed reproducibility."""

    def test_set_seed_reproducible_numpy(self):
        """Setting the same seed produces identical numpy random sequences."""
        set_seed(123)
        seq1 = np.random.rand(5)

        set_seed(123)
        seq2 = np.random.rand(5)

        np.testing.assert_array_equal(seq1, seq2)

    def test_set_seed_reproducible_python_random(self):
        """Setting the same seed produces identical Python random sequences."""
        set_seed(456)
        val1 = [random.random() for _ in range(5)]

        set_seed(456)
        val2 = [random.random() for _ in range(5)]

        assert val1 == val2
