"""Tests for src.models.predict module.

A small model is trained on synthetic data and written to a temporary path so the
loading, threshold loading and batch scoring flows are exercised end-to-end.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import NUMERIC_FEATURES
from src.models.predict import (
    DEFAULT_THRESHOLD,
    LABEL_COL,
    PROBA_COL,
    Predictor,
    load_model,
    load_threshold,
    predict_batch,
)
from src.models.train import save_metrics, save_model, train_model


def _make_feature_df(n_rows: int = 120, seed: int = 5):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.normal(size=n_rows) for c in NUMERIC_FEATURES})
    y = (X["age"] + X["DebtRatio"] > 0).astype(int)
    return X, y


@pytest.fixture()
def saved_model(tmp_path):
    """Train a real model and persist model + metrics to a temp directory."""
    X, y = _make_feature_df()
    pipeline = train_model("logistic_regression", X, y)
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    save_model(pipeline, model_path)
    save_metrics({"threshold": 0.42, "pr_auc": 0.9}, metrics_path)
    return X, model_path, metrics_path


class TestLoadModel:
    """Tests for load_model."""

    def test_loads_saved_pipeline(self, saved_model):
        _, model_path, _ = saved_model
        assert load_model(model_path) is not None

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_model(tmp_path / "nope.joblib")


class TestLoadThreshold:
    """Tests for load_threshold."""

    def test_returns_stored_threshold(self, saved_model):
        _, _, metrics_path = saved_model
        assert load_threshold(metrics_path) == 0.42

    def test_returns_default_when_missing(self, tmp_path):
        assert load_threshold(tmp_path / "missing.json") == DEFAULT_THRESHOLD


class TestPredictor:
    """Tests for Predictor."""

    def test_from_disk_loads_model_and_threshold(self, saved_model):
        _, model_path, metrics_path = saved_model
        predictor = Predictor.from_disk(model_path=model_path, metrics_path=metrics_path)
        assert predictor.threshold == 0.42

    def test_predict_proba_in_unit_interval(self, saved_model):
        X, model_path, metrics_path = saved_model
        predictor = Predictor.from_disk(model_path=model_path, metrics_path=metrics_path)
        proba = predictor.predict_proba(X)
        assert len(proba) == len(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_returns_binary(self, saved_model):
        X, model_path, metrics_path = saved_model
        predictor = Predictor.from_disk(model_path=model_path, metrics_path=metrics_path)
        pred = predictor.predict(X)
        assert set(np.unique(pred)).issubset({0, 1})

    def test_missing_feature_column_raises(self, saved_model):
        _, model_path, metrics_path = saved_model
        predictor = Predictor.from_disk(model_path=model_path, metrics_path=metrics_path)
        X_bad = _make_feature_df()[0].drop(columns=["age"])
        with pytest.raises(ValueError, match="Missing expected feature columns"):
            predictor.predict_proba(X_bad)


class TestPredictBatch:
    """Tests for predict_batch."""

    def test_scores_csv_and_writes_output(self, saved_model, tmp_path):
        X, model_path, _ = saved_model
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        X.to_csv(input_path)  # keeps the default integer index

        result = predict_batch(input_path, output_path, model_path=model_path, threshold=0.5)

        assert output_path.exists()
        assert set(np.unique(result[LABEL_COL])).issubset({0, 1})
        assert (result[PROBA_COL] >= 0).all() and (result[PROBA_COL] <= 1).all()
        # The original feature columns are preserved alongside the predictions.
        assert set(NUMERIC_FEATURES).issubset(result.columns)

    def test_drops_target_column_before_scoring(self, saved_model, tmp_path):
        X, model_path, _ = saved_model
        df_with_target = X.copy()
        df_with_target["SeriousDlqin2yrs"] = pd.Series([0] * len(X), index=X.index)
        input_path = tmp_path / "with_target.csv"
        output_path = tmp_path / "out.csv"
        df_with_target.to_csv(input_path)

        result = predict_batch(input_path, output_path, model_path=model_path, threshold=0.5)
        assert "SeriousDlqin2yrs" not in result.columns
        assert PROBA_COL in result.columns
