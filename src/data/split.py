"""Train/validation/test splitting utilities for the credit default prediction project.

Splitting is performed ONLY on the labeled training data (``cs-training.csv``);
the unlabeled ``cs-test.csv`` is reserved for final predictions. Splits are
stratified to preserve the class balance (~6.7% defaulters), and are a pure
row-partition performed before any fitting, so they introduce no leakage.

Split statistics are deterministic for a given ``random_state`` (see
``DEFAULT_SEED``). The seed and the split ratios are module constants for now;
they can be promoted to ``configs/config.yaml`` once that file is populated.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import get_config_value, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default split ratios and seed loaded from configs/config.yaml with fallbacks.
_cfg = load_config()
DEFAULT_TEST_SIZE = float(get_config_value(_cfg, "data.split.test_size", 0.15))
DEFAULT_VAL_SIZE = float(get_config_value(_cfg, "data.split.val_size", 0.15))
DEFAULT_SEED = int(get_config_value(_cfg, "seed", 42))


def stratified_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split features/target into stratified train/validation/test partitions.

    A two-stage ``train_test_split`` is used so that both the validation and
    test sets are stratified on ``y``, preserving the class distribution.

    Args:
        X: Feature DataFrame. Rows are partitioned; columns are untouched.
        y: Target Series aligned with ``X``.
        test_size: Fraction of rows held out for the test set.
        val_size: Fraction of rows held out for the validation set.
        random_state: Seed for reproducible splits.

    Returns:
        Tuple ``(X_train, X_val, X_test, y_train, y_val, y_test)``. The samples
        are disjoint and their union covers ``X`` exactly.

    Raises:
        ValueError: If ``val_size + test_size`` is not a strict fraction < 1.
    """
    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) must be < 1.0"
        )

    # Split off the test set first (train/val vs test), stratified on y.
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Re-split the remaining train/val block to carve out the validation set.
    # Scale val_size relative to the reduced pool so the final fractions hold.
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_ratio, stratify=y_tv, random_state=random_state
    )

    logger.info(
        "Split sizes -> train: %d (%.1f%%) | val: %d (%.1f%%) | test: %d (%.1f%%)",
        len(X_train), 100.0 * len(X_train) / len(X),
        len(X_val), 100.0 * len(X_val) / len(X),
        len(X_test), 100.0 * len(X_test) / len(X),
    )
    for name, y_split in (("train", y_train), ("val", y_val), ("test", y_test)):
        positive_rate = 100.0 * y_split.mean()
        logger.info("%s set positive rate: %.2f%%", name, positive_rate)

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_dataset(
    df: pd.DataFrame,
    target_col: str = "SeriousDlqin2yrs",
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split a labeled DataFrame into stratified train/validation/test partitions.

    This is a convenience wrapper around ``stratified_train_val_test_split``
    that separates the target column from the features first.

    Args:
        df: Labeled DataFrame (training data, including the target column).
        target_col: Name of the target column.
        test_size: Fraction of rows held out for the test set.
        val_size: Fraction of rows held out for the validation set.
        random_state: Seed for reproducible splits.

    Returns:
        Tuple ``(X_train, X_val, X_test, y_train, y_val, y_test)``.

    Raises:
        ValueError: If ``target_col`` is not present in ``df``.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return stratified_train_val_test_split(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )
