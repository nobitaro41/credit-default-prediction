"""Tests for the src.models module (train.py and evaluate.py).

Small synthetic DataFrames keep these tests fast; MLflow is disabled everywhere
to avoid side effects and to keep the tests self-contained.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from src.features.build_features import NUMERIC_FEATURES
from src.models.evaluate import (
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    SUMMARY_METRICS,
    ModelEvaluator,
)
from src.models.train import (
    MODEL_REGISTRY,
    PRIMARY_METRIC,
    run_training,
    train_all,
    train_model,
)


def _make_data(n_rows: int = 300, seed: int = 0):
    """Build deterministic, separable features + binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.normal(size=n_rows) for c in NUMERIC_FEATURES})
    y = (X["age"] + X["DebtRatio"] > 0).astype(int)
    return X, y


def _expected_cost(y_true, y_pred, fn_cost=DEFAULT_FN_COST, fp_cost=DEFAULT_FP_COST):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn_cost * fn + fp_cost * fp


class TestTrainModel:
    """Tests for train_model."""

    def test_returns_fitted_pipeline(self):
        """train_model fits and returns a pipeline that can predict."""
        X, y = _make_data()
        pipe = train_model("logistic_regression", X, y)
        proba = pipe.predict_proba(X)[:, 1]
        assert len(proba) == len(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_unknown_model_raises(self):
        """An unknown model name raises ValueError."""
        X, y = _make_data()
        with pytest.raises(ValueError, match="Unknown model"):
            train_model("not_a_model", X, y)

    def test_registry_contains_baseline_and_gbms(self):
        """The registry covers baseline + the strong models from spec."""
        assert "logistic_regression" in MODEL_REGISTRY
        for name in ("random_forest", "xgboost", "lightgbm"):
            assert name in MODEL_REGISTRY


class TestTrainAll:
    """Tests for train_all."""

    def test_trains_all_requested_models(self):
        """train_all returns one pipeline + metrics per requested model."""
        X, y = _make_data()
        pipes, metrics, best = train_all(X, y, X, y, mlflow_tracking=False)
        assert set(pipes.keys()) == set(MODEL_REGISTRY)
        assert set(metrics.keys()) == set(MODEL_REGISTRY)
        assert best in metrics

    def test_best_model_has_max_pr_auc(self):
        """The best model is the one with the highest PR-AUC on the validation set."""
        X, y = _make_data()
        _, metrics, best = train_all(X, y, X, y, mlflow_tracking=False)
        assert metrics[best][PRIMARY_METRIC] == max(
            m[PRIMARY_METRIC] for m in metrics.values()
        )

    def test_subset_of_models(self):
        """model_names restricts which models are trained."""
        X, y = _make_data()
        pipes, _, _ = train_all(X, y, X, y, model_names=["logistic_regression"], mlflow_tracking=False)
        assert list(pipes.keys()) == ["logistic_regression"]


class TestRunTraining:
    """Tests for run_training orchestration."""

    def test_saves_model_and_metrics(self, tmp_path):
        """run_training writes model.joblib and metrics.json with the best model."""
        X, y = _make_data()
        model_path = tmp_path / "model.joblib"
        metrics_path = tmp_path / "metrics.json"
        best, metrics = run_training(
            X, y, X, y, X, y,
            model_names=["logistic_regression"],
            mlflow_tracking=False,
            model_path=model_path,
            metrics_path=metrics_path,
        )
        assert best == "logistic_regression"
        assert metrics["best_model"] == best
        assert 0.0 <= metrics["threshold"] <= 1.0
        assert model_path.exists() and metrics_path.exists()

    def test_final_metrics_have_scalar_summary(self, tmp_path):
        """The saved metrics include the standard scalar summary metrics."""
        X, y = _make_data()
        _, metrics = run_training(
            X, y, X, y, X, y,
            model_names=["logistic_regression"],
            mlflow_tracking=False,
            model_path=tmp_path / "model.joblib",
            metrics_path=tmp_path / "metrics.json",
        )
        for key in SUMMARY_METRICS:
            assert key in metrics
            assert isinstance(metrics[key], float)


class TestModelEvaluator:
    """Tests for ModelEvaluator."""

    @pytest.fixture()
    def fitted(self):
        X, y = _make_data()
        pipe = train_model("logistic_regression", X, y)
        return ModelEvaluator(pipe), X, y

    def test_predict_returns_binary(self, fitted):
        """Predictions are 0/1 integers within the expected range."""
        evaluator, X, _ = fitted
        pred = evaluator.predict(X)
        assert set(np.unique(pred)).issubset({0, 1})
        assert len(pred) == len(X)

    def test_evaluate_returns_required_keys(self, fitted):
        """evaluate returns all summary metrics plus confusion matrix."""
        evaluator, X, y = fitted
        metrics = evaluator.evaluate(X, y)
        for key in SUMMARY_METRICS:
            assert key in metrics
        assert len(metrics["confusion_matrix"]) == 2  # 2x2 confusion matrix

    def test_optimal_threshold_lowers_expected_cost(self, fitted):
        """The cost-optimal threshold does not cost more than threshold=0.5."""
        evaluator, X, y = fitted
        evaluator.threshold = 0.5  # baseline
        cost_at_default = _expected_cost(y, evaluator.predict(X))
        best = evaluator.optimal_threshold(X, y)  # sets self.threshold to the best
        assert 0.0 <= best <= 1.0
        cost_at_optimal = _expected_cost(y, evaluator.predict(X))
        assert cost_at_optimal <= cost_at_default
