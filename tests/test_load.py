"""Tests for src.data.load module."""

import pandas as pd
import pytest

from src.data.load import (
    EXPECTED_COLUMNS,
    EXPECTED_DTYPES,
    get_feature_target_split,
    load_test_data,
    load_training_data,
    validate_schema,
)


class TestLoadTrainingData:
    """Tests for load_training_data function."""

    def test_load_training_data_shape(self):
        """Test that training data loads with correct shape."""
        df = load_training_data()
        assert df.shape == (150000, 11)

    def test_load_training_data_columns(self):
        """Test that training data has all expected columns."""
        df = load_training_data()
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_load_training_data_target_binary(self):
        """Test that target column only contains 0 and 1."""
        df = load_training_data()
        target_values = df["SeriousDlqin2yrs"].unique()
        assert set(target_values).issubset({0, 1})

    def test_load_training_data_target_distribution(self):
        """Test target class distribution (imbalanced ~6.7% positive)."""
        df = load_training_data()
        target_counts = df["SeriousDlqin2yrs"].value_counts()
        assert target_counts[0] > target_counts[1]  # Majority class is 0
        assert target_counts[1] == 10026  # Exact count from data


class TestLoadTestData:
    """Tests for load_test_data function."""

    def test_load_test_data_shape(self):
        """Test that test data loads with correct shape (no target)."""
        df = load_test_data()
        assert df.shape == (101503, 10)

    def test_load_test_data_no_target(self):
        """Test that test data does not have target column."""
        df = load_test_data()
        assert "SeriousDlqin2yrs" not in df.columns

    def test_load_test_data_columns(self):
        """Test that test data has expected feature columns."""
        df = load_test_data()
        expected = [c for c in EXPECTED_COLUMNS if c != "SeriousDlqin2yrs"]
        assert list(df.columns) == expected


class TestValidateSchema:
    """Tests for validate_schema function."""

    def test_validate_schema_passes_on_valid_data(self):
        """Test that validate_schema passes on valid training data."""
        df = load_training_data()
        # Should not raise
        validate_schema(df)

    def test_validate_schema_fails_on_missing_column(self):
        """Test that validate_schema fails when required column is missing."""
        df = load_training_data()
        df_missing = df.drop(columns=["age"])
        with pytest.raises(ValueError, match="Required column missing: age"):
            validate_schema(df_missing)

    def test_validate_schema_fails_on_invalid_target(self):
        """Test that validate_schema fails when target has invalid values."""
        df = load_training_data()
        df_invalid = df.copy()
        df_invalid.loc[0, "SeriousDlqin2yrs"] = 2  # Invalid value
        with pytest.raises(ValueError, match="Target column has unexpected values"):
            validate_schema(df_invalid)


class TestGetFeatureTargetSplit:
    """Tests for get_feature_target_split function."""

    def test_split_returns_correct_shapes(self):
        """Test that split returns correct X and y shapes."""
        df = load_training_data()
        X, y = get_feature_target_split(df)
        assert X.shape == (150000, 10)
        assert y.shape == (150000,)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_split_excludes_target_from_features(self):
        """Test that target column is not in features."""
        df = load_training_data()
        X, y = get_feature_target_split(df)
        assert "SeriousDlqin2yrs" not in X.columns
        assert y.name == "SeriousDlqin2yrs"

    def test_split_fails_on_missing_target(self):
        """Test that split fails when target column is missing."""
        df = load_test_data()  # No target column
        with pytest.raises(ValueError, match="Target column 'SeriousDlqin2yrs' not found"):
            get_feature_target_split(df)

    def test_split_custom_target_column(self):
        """Test split with custom target column name."""
        df = load_training_data()
        df_renamed = df.rename(columns={"SeriousDlqin2yrs": "target"})
        X, y = get_feature_target_split(df_renamed, target_col="target")
        assert X.shape == (150000, 10)
        assert y.name == "target"


class TestConstants:
    """Tests for module constants."""

    def test_expected_columns_complete(self):
        """Test that EXPECTED_COLUMNS has all required columns."""
        required = {
            "SeriousDlqin2yrs",
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        }
        assert set(EXPECTED_COLUMNS) == required

    def test_expected_dtypes_complete(self):
        """Test that EXPECTED_DTYPES has entries for all columns."""
        assert set(EXPECTED_DTYPES.keys()) == set(EXPECTED_COLUMNS)
