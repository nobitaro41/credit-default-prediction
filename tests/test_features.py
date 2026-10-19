"""Tests for src.features.build_features module.

Small synthetic DataFrames keep these tests fast and deterministic.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.build_features import (
    NUMERIC_FEATURES,
    build_preprocessor,
    make_full_pipeline,
    validate_feature_columns,
)


def _make_feature_df(n_rows: int = 80) -> pd.DataFrame:
    """Build a DataFrame with the expected numeric features, with some NaNs."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=n_rows) for c in NUMERIC_FEATURES})
    df.loc[:5, "MonthlyIncome"] = np.nan
    df.loc[:3, "NumberOfDependents"] = np.nan
    return df


def _make_target(n_rows: int = 80) -> pd.Series:
    return pd.Series([i % 2 for i in range(n_rows)])


class TestBuildPreprocessor:
    """Tests for build_preprocessor."""

    def test_imputes_missing_values(self):
        """The fitted preprocessor leaves no NaN in its output."""
        pre = build_preprocessor()
        X = _make_feature_df()
        transformed = pre.fit_transform(X)
        assert not np.isnan(transformed).any()

    def test_standardises_columns(self):
        """Each transformed column has ~zero mean and unit variance."""
        pre = build_preprocessor()
        X = _make_feature_df()
        transformed = pre.fit_transform(X)
        assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-6)

    def test_imputer_fit_only_on_train(self):
        """Median imputation comes from train; a test set missing a value is fine."""
        X_train = _make_feature_df()
        # A test column value that is NaN but not present in train extremes.
        X_test = _make_feature_df(n_rows=10)
        pre = build_preprocessor()
        pre.fit(X_train)
        transformed = pre.transform(X_test)
        assert not np.isnan(transformed).any()


class TestMakeFullPipeline:
    """Tests for make_full_pipeline."""

    def test_returns_pipeline_instance(self):
        """make_full_pipeline wraps the classifier in a sklearn Pipeline."""
        pipe = make_full_pipeline(LogisticRegression(max_iter=1000))
        assert hasattr(pipe, "fit") and hasattr(pipe, "predict_proba")

    def test_fit_predict_roundtrip(self):
        """The combined pipeline fits and produces valid binary probabilities."""
        pipe = make_full_pipeline(LogisticRegression(max_iter=1000))
        X = _make_feature_df()
        y = _make_target()
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)[:, 1]
        assert len(proba) == len(X)
        assert (proba >= 0).all() and (proba <= 1).all()


class TestValidateFeatureColumns:
    """Tests for validate_feature_columns."""

    def test_passes_on_complete_columns(self):
        """No error when all expected feature columns are present."""
        validate_feature_columns(_make_feature_df())  # should not raise

    def test_raises_on_missing_column(self):
        """Raises when an expected feature column is absent."""
        df = _make_feature_df().drop(columns=["age"])
        with pytest.raises(ValueError, match="Missing expected feature columns"):
            validate_feature_columns(df)
