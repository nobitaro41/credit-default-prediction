"""Deterministic data cleaning utilities for the credit default prediction project.

This module performs rule-based, parameter-free cleaning that can be applied
identically to the train/validation/test splits BEFORE the feature pipeline,
without leaking information: every rule here uses a fixed threshold, never a
statistic computed from the data.

Imputation of missing values (``MonthlyIncome``, ``NumberOfDependents``) is
intentionally NOT done here. It belongs to the feature pipeline that is fit
only on the training set (see ``src.features.build_features``).
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns reporting how many months a borrower was past due. The raw data
# encodes missing/erroneous values as 96/98; anything >= this threshold is a
# data-entry error, not a real delinquency count.
PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

# Fixed business caps used to winsorise extreme outliers. These are chosen to
# cover the plausible range of the raw data; they are NOT derived from it, so
# the same caps can be applied to future / out-of-time data.
MONTHLY_INCOME_CAP = 200_000.0  # Monthly income above this is implausible.
DEBT_RATIO_CAP = 6_000.0        # DebtRatio explodes when MonthlyIncome is 0; cap bounds it.


def drop_invalid_rows(df: pd.DataFrame, max_past_due: int = 90, min_age: int = 1) -> pd.DataFrame:
    """Drop rows with data-entry errors (past-due sentinels and non-positive age).

    Args:
        df: Input DataFrame containing the raw feature columns.
        max_past_due: Rows with any past-due column >= this value are dropped
            (raw data uses 96/98 as missing/erroneous sentinels).
        min_age: Rows with age < this value are dropped (age == 0 is invalid).

    Returns:
        DataFrame with invalid rows removed. The index is carried over unchanged.
    """
    before = len(df)

    past_due_mask = df[PAST_DUE_COLUMNS].ge(max_past_due).any(axis=1)
    age_mask = df["age"] < min_age

    if past_due_mask.any() or age_mask.any():
        # Count disjoint categories so the breakdown sums to the total dropped.
        past_only = int((past_due_mask & ~age_mask).sum())
        age_only = int((age_mask & ~past_due_mask).sum())
        both = int((past_due_mask & age_mask).sum())
        logger.info(
            f"Dropping {int((past_due_mask | age_mask).sum())} invalid rows "
            f"(past-due sentinels: {past_only}, non-positive age: {age_only}, both: {both})."
        )
        df = df.loc[~(past_due_mask | age_mask)]
    else:
        logger.info("No invalid rows to drop.")

    logger.info(f"Row count: {before} -> {len(df)}")
    return df


def clip_outliers(
    df: pd.DataFrame,
    monthly_income_cap: float = MONTHLY_INCOME_CAP,
    debt_ratio_cap: float = DEBT_RATIO_CAP,
) -> pd.DataFrame:
    """Winsorise the most extreme outliers of MonthlyIncome and DebtRatio.

    The caps are fixed constants (see module docstring), so this transform is
    deterministic and safe to apply identically across all splits.

    Args:
        df: Input DataFrame.
        monthly_income_cap: Upper bound for MonthlyIncome.
        debt_ratio_cap: Upper bound for DebtRatio.

    Returns:
        DataFrame with the two columns clipped to their upper bounds.
    """
    for column, cap in (("MonthlyIncome", monthly_income_cap), ("DebtRatio", debt_ratio_cap)):
        if column not in df.columns:
            continue
        n_clipped = int(df[column].gt(cap).sum())
        df = df.copy() if n_clipped else df
        if n_clipped:
            df[column] = df[column].clip(upper=cap)
            logger.info(f"Clipped {n_clipped} rows of {column} to cap {cap}.")
        else:
            logger.info(f"No {column} values exceeded the cap {cap}.")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data end-to-end and return a tidy DataFrame.

    Applies ``drop_invalid_rows`` then ``clip_outliers``, and resets the row
    index so downstream steps get a clean, contiguous index. Missing values are
    preserved as-is (imputation happens later, in the fitted feature pipeline).

    Args:
        df: Raw DataFrame, typically the output of ``load_training_data``.

    Returns:
        Cleaned DataFrame with a fresh 0..N-1 index.
    """
    logger.info(f"Cleaning data: shape {df.shape}")
    df = drop_invalid_rows(df)
    df = clip_outliers(df)
    df = df.reset_index(drop=True)
    logger.info(f"Cleaning done: shape {df.shape}")
    return df
