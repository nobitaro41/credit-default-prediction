"""Tests for src.data.split module.

These tests use small synthetic DataFrames so they run fast and deterministically,
without loading the full 150K-row training set.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.split import (
    DEFAULT_SEED,
    split_dataset,
    stratified_train_val_test_split,
)

TARGET_COL = "SeriousDlqin2yrs"


def _make_df(n_rows: int = 200, positive_ratio: float = 0.1) -> pd.DataFrame:
    """Build a synthetic DataFrame with a binary target and a few features."""
    rng = np.random.default_rng(7)
    n_positive = int(round(n_rows * positive_ratio))
    y = np.array([0] * (n_rows - n_positive) + [1] * n_positive)
    rng.shuffle(y)
    return pd.DataFrame(
        {
            TARGET_COL: y,
            "feature_a": rng.normal(size=n_rows),
            "feature_b": rng.uniform(0, 1, size=n_rows),
        }
    )


def _assert_disjoint_union(x_splits, y_splits, original_index):
    """The partitions are disjoint and cover the original index."""
    all_idx = set()
    for x_split in x_splits:
        live = set(x_split.index)
        assert not (live & all_idx), "Split partitions must be disjoint."
        all_idx |= live
    assert all_idx == set(original_index)


class TestStratifiedSplit:
    """Tests for stratified_train_val_test_split."""

    def test_returns_six_objects(self):
        """The splitter returns 6 objects (3 X, 3 y) of matching lengths."""
        df = _make_df()
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        result = stratified_train_val_test_split(X, y)
        assert len(result) == 6
        X_train, X_val, X_test, y_train, y_val, y_test = result
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

    def test_approx_ratios(self):
        """Split sizes are close to the requested train/val/test fractions."""
        df = _make_df()
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        X_train, X_val, X_test, *_ = stratified_train_val_test_split(X, y, test_size=0.15, val_size=0.15)
        n = len(df)
        assert len(X_test) / n == pytest.approx(0.15, abs=0.02)
        assert len(X_val) / n == pytest.approx(0.15, abs=0.02)
        assert len(X_train) / n == pytest.approx(0.70, abs=0.02)

    def test_strat_preserves_positive_rate(self):
        """Each split keeps roughly the target positive rate."""
        df = _make_df(n_rows=1000, positive_ratio=0.1)
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        _, _, _, y_train, y_val, y_test = stratified_train_val_test_split(X, y)
        for y_split in (y_train, y_val, y_test):
            assert y_split.mean() == pytest.approx(0.1, abs=0.02)

    def test_disjoint_and_exhaustive(self):
        """Partitions are disjoint and cover the whole original index."""
        df = _make_df()
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(X, y)
        _assert_disjoint_union(
            (X_train, X_val, X_test), (y_train, y_val, y_test), df.index
        )

    def test_reproducible_with_same_seed(self):
        """Same random_state yields the exact same row assignment."""
        df = _make_df()
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        a = stratified_train_val_test_split(X, y, random_state=42)
        b = stratified_train_val_test_split(X, y, random_state=42)
        for xa, xb in zip(a[:3], b[:3]):
            assert list(xa.index) == list(xb.index)

    def test_raises_on_large_holdout(self):
        """val + test >= 1.0 must raise."""
        df = _make_df()
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        with pytest.raises(ValueError, match="must be < 1.0"):
            stratified_train_val_test_split(X, y, test_size=0.6, val_size=0.6)


class TestSplitDataset:
    """Tests for split_dataset wrapper."""

    def test_separates_target_from_features(self):
        """X splits exclude the target column."""
        df = _make_df()
        X_train, _, _, _, _, _ = split_dataset(df)
        assert TARGET_COL not in X_train.columns

    def test_missing_target_raises(self):
        """A DataFrame without the target column raises ValueError."""
        df = _make_df().drop(columns=[TARGET_COL])
        with pytest.raises(ValueError, match="Target column"):
            split_dataset(df)

    def test_default_seed_constant(self):
        """The default seed is a stable, reproducible constant."""
        assert DEFAULT_SEED == 42
