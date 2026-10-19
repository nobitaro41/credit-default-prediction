"""Tests for src.models.tune module.

Tiny synthetic data and a couple of trials keep the Optuna runs fast; tuning is
never allowed to touch the test set, so these tests exercise the search on the
training data only.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import NUMERIC_FEATURES
from src.models.tune import (
    build_tuned_pipeline,
    get_search_space,
    objective,
    tune_model,
)

FAST_KWARGS = dict(n_trials=2, cv_folds=2)


def _make_data(n_rows: int = 160, seed: int = 3):
    """Deterministic, separable features with a balanced binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.normal(size=n_rows) for c in NUMERIC_FEATURES})
    y = (X["age"] + X["DebtRatio"] > 0).astype(int)
    return X, y


class TestGetSearchSpace:
    """Tests for get_search_space."""

    def test_logistic_regression_space(self):
        """LR search space samples C and solver."""
        space = get_search_space(_TrialStub(), "logistic_regression")
        assert {"C", "solver"} <= set(space)

    def test_xgboost_space_has_core_grid(self):
        """XGBoost space includes depth/learning-rate/regularisation."""
        space = get_search_space(_TrialStub(), "xgboost")
        assert {"max_depth", "learning_rate", "reg_lambda"} <= set(space)

    def test_unknown_model_raises(self):
        """An unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_search_space(_TrialStub(), "does_not_exist")


class _TrialStub:
    """A minimal stand-in for optuna.Trial using fixed suggestions."""

    def suggest_float(self, name, low, high, log=False):
        return float(low)

    def suggest_int(self, name, low, high, step=1):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


class TestObjective:
    """Tests for the Optuna objective."""

    def test_returns_valid_pr_auc(self):
        """The objective returns a PR-AUC in [0, 1]."""
        X, y = _make_data()
        value = objective(_TrialStub(), "logistic_regression", X, y, cv_folds=2, random_state=0)
        assert 0.0 <= value <= 1.0


class TestTuneModel:
    """Tests for tune_model."""

    def test_returns_params_value_and_pipeline(self):
        """tune_model returns best_params, best PR-AUC, and a fitted pipeline."""
        X, y = _make_data()
        best_params, best_value, pipeline = tune_model(
            "logistic_regression", X, y, **FAST_KWARGS, random_state=0
        )
        assert isinstance(best_params, dict)
        assert 0.0 <= best_value <= 1.0
        assert hasattr(pipeline, "predict_proba")
        proba = pipeline.predict_proba(X)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_best_params_are_estimator_kwargs(self):
        """The returned best params feed directly into the estimator."""
        X, y = _make_data()
        best_params, _, _ = tune_model(
            "logistic_regression", X, y, **FAST_KWARGS, random_state=0
        )
        assert "C" in best_params

    def test_unknown_model_raises(self):
        """tune_model rejects unknown model names."""
        X, y = _make_data()
        with pytest.raises(ValueError, match="Unknown model"):
            tune_model("does_not_exist", X, y, **FAST_KWARGS)


class TestBuildTunedPipeline:
    """Tests for build_tuned_pipeline."""

    def test_fits_with_provided_params(self):
        """build_tuned_pipeline fits a pipeline using the given params."""
        X, y = _make_data()
        pipeline = build_tuned_pipeline(
            "logistic_regression", X, y,
            {"C": 0.1, "solver": "lbfgs"}, random_state=0,
        )
        preds = pipeline.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})
