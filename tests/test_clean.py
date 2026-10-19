"""Tests for src.data.clean module.

These tests use small synthetic DataFrames so they run fast and deterministically,
without loading the full 150K-row training set.
"""

import pandas as pd

from src.data.clean import (
    DEBT_RATIO_CAP,
    MONTHLY_INCOME_CAP,
    PAST_DUE_COLUMNS,
    clean_data,
    clip_outliers,
    drop_invalid_rows,
)


def _make_df() -> pd.DataFrame:
    """Build a small raw-like DataFrame for the tests.

    Rows 0-1 are valid (row 0 carries missing values, row 1 carries outliers),
    rows 2-5 carry the invalid patterns the cleaner must remove.
    """
    return pd.DataFrame(
        {
            "SeriousDlqin2yrs": [0, 1, 0, 1, 0, 0],
            "RevolvingUtilizationOfUnsecuredLines": [0.1, 0.5, 0.2, 0.3, 0.4, 0.6],
            "age": [35, 40, 45, 50, 0, 55],  # row 4 has invalid age
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 98, 2, 1, 3],  # row 2 sentinel
            "DebtRatio": [0.5, 9999.0, 0.8, 9999.0, 0.3, 0.7],  # row 1 outlier
            "MonthlyIncome": [None, 300000.0, 6000.0, 7000.0, 5000.0, 8000.0],  # NaN + outlier
            "NumberOfOpenCreditLinesAndLoans": [2, 3, 4, 5, 6, 7],
            "NumberOfTimes90DaysLate": [0, 1, 0, 98, 0, 1],  # row 3 sentinel
            "NumberRealEstateLoansOrLines": [1, 1, 0, 2, 1, 0],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 1, 0, 0, 0, 90],  # row 5 sentinel
            "NumberOfDependents": [None, 0.0, 2.0, 1.0, 1.0, 3.0],  # 1 NaN
        }
    )


class TestDropInvalidRows:
    """Tests for drop_invalid_rows."""

    def test_drops_past_due_sentinels(self):
        """Rows with a past-due value >= 90 are dropped."""
        df = _make_df()
        cleaned = drop_invalid_rows(df)
        assert len(cleaned) == 2  # only the 2 valid rows survive
        assert not cleaned[PAST_DUE_COLUMNS].ge(90).any().any()

    def test_drops_non_positive_age(self):
        """Rows with age < min_age are dropped."""
        df = _make_df()
        cleaned = drop_invalid_rows(df)
        assert (cleaned["age"] >= 1).all()

    def test_no_change_on_clean_data(self):
        """A DataFrame with no invalid rows is returned length-preserving."""
        df = _make_df()
        valid = df[~df[PAST_DUE_COLUMNS].ge(90).any(axis=1) & (df["age"] >= 1)]
        cleaned = drop_invalid_rows(valid)
        assert list(cleaned.index) == list(valid.index)


class TestClipOutliers:
    """Tests for clip_outliers."""

    def test_clips_monthly_income(self):
        """MonthlyIncome above the cap is clipped to the cap."""
        df = _make_df()
        cleaned = clip_outliers(df)
        assert cleaned["MonthlyIncome"].max() <= MONTHLY_INCOME_CAP

    def test_clips_debt_ratio(self):
        """DebtRatio above the cap is clipped to the cap."""
        df = _make_df()
        cleaned = clip_outliers(df)
        assert cleaned["DebtRatio"].max() <= DEBT_RATIO_CAP

    def test_rows_are_either_clipped_or_unchanged(self):
        """Clipped rows are capped exactly; other rows keep their values."""
        df = _make_df()
        cleaned = clip_outliers(df)
        # Row 1 is the outlier: both values land exactly on the caps.
        assert cleaned.loc[1, "MonthlyIncome"] == MONTHLY_INCOME_CAP
        assert cleaned.loc[1, "DebtRatio"] == DEBT_RATIO_CAP
        # Non-outlier row keeps its original value.
        assert cleaned.loc[0, "DebtRatio"] == 0.5


class TestCleanData:
    """Tests for clean_data."""

    def test_returns_dataframe(self):
        """clean_data returns a DataFrame."""
        assert isinstance(clean_data(_make_df()), pd.DataFrame)

    def test_resets_index(self):
        """The output index is a clean contiguous range."""
        cleaned = clean_data(_make_df())
        assert list(cleaned.index) == list(range(len(cleaned)))

    def test_no_sentinel_or_invalid_age(self):
        """No past-due sentinel and no non-positive age remain."""
        cleaned = clean_data(_make_df())
        assert not cleaned[PAST_DUE_COLUMNS].ge(90).any().any()
        assert (cleaned["age"] >= 1).all()

    def test_clips_outliers_end_to_end(self):
        """Outliers are capped after the full cleaning pass."""
        cleaned = clean_data(_make_df())
        assert cleaned["MonthlyIncome"].max() <= MONTHLY_INCOME_CAP
        assert cleaned["DebtRatio"].max() <= DEBT_RATIO_CAP

    def test_preserves_missing_values(self):
        """clean_data does NOT impute NaN (imputation is fitted in the pipeline)."""
        cleaned = clean_data(_make_df())
        assert cleaned["MonthlyIncome"].isna().any()
        assert cleaned["NumberOfDependents"].isna().any()

    def test_handles_test_data_without_target(self):
        """clean_data works on unlabelled test data (no target column)."""
        df = _make_df().drop(columns=["SeriousDlqin2yrs"])
        cleaned = clean_data(df)
        assert "SeriousDlqin2yrs" not in cleaned.columns
        assert list(cleaned.index) == list(range(len(cleaned)))

    def test_preserves_columns(self):
        """clean_data drops rows only, never columns."""
        df = _make_df()
        cleaned = clean_data(df)
        assert list(cleaned.columns) == list(df.columns)
