"""Model training and MLflow experiment tracking.

Trains a family of classifiers — logistic regression baseline, random forest,
XGBoost, LightGBM — each wrapped in the same ``preprocessor + classifier``
pipeline (see ``src.features.build_features``). Runs are logged to MLflow,
models are ranked on the validation set by PR-AUC (the metric of choice for an
imbalanced target), and the best one is saved together with its test metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.build_features import make_full_pipeline
from src.models.evaluate import SUMMARY_METRICS, ModelEvaluator
from src.utils.config import get_config_value, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cfg = load_config()
DEFAULT_SEED = int(get_config_value(_cfg, "seed", 42))

# Metric used to rank models on the validation set (imbalanced target -> PR-AUC).
PRIMARY_METRIC = str(get_config_value(_cfg, "models.primary_metric", "pr_auc"))

# Registry of trainable classifiers. Each entry holds the estimator class and its
# fixed hyper-parameters; ``random_state`` is always injected per run for
# reproducibility, and extra overrides can be passed to ``train_model``.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "class": LogisticRegression,
        "params": {"max_iter": 2000},
    },
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {"n_estimators": 200, "n_jobs": -1},
    },
    "xgboost": {
        "class": XGBClassifier,
        "params": {"n_estimators": 200, "learning_rate": 0.1, "n_jobs": -1},
    },
    "lightgbm": {
        "class": LGBMClassifier,
        "params": {"n_estimators": 200, "learning_rate": 0.1, "n_jobs": -1, "verbosity": -1},
    },
}


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = DEFAULT_SEED,
    estimator_kwargs: Optional[dict[str, Any]] = None,
) -> Pipeline:
    """Fit a single model (preprocessor + classifier) on the training set.

    The imputer and scaler are fit on ``X_train`` only, so there is no leakage.

    Args:
        model_name: Key into ``MODEL_REGISTRY``.
        X_train: Training features.
        y_train: Training labels.
        random_state: Seed for the classifier, for reproducibility.
        estimator_kwargs: Optional per-run estimator parameter overrides.

    Returns:
        Fitted ``Pipeline`` (preprocessor + classifier).

    Raises:
        ValueError: If ``model_name`` is not in ``MODEL_REGISTRY``.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}"
        )

    entry = MODEL_REGISTRY[model_name]
    params = dict(entry["params"])
    params["random_state"] = random_state
    if estimator_kwargs:
        params.update(estimator_kwargs)

    classifier = entry["class"](**params)
    pipeline = make_full_pipeline(classifier)
    pipeline.fit(X_train, y_train)

    logger.info("Trained '%s' on %d rows.", model_name, len(X_train))
    return pipeline


def train_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_names: Optional[list[str]] = None,
    random_state: int = DEFAULT_SEED,
    mlflow_tracking: bool = True,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]], str]:
    """Train every requested model, evaluate on the validation set, rank them.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (used for ranking, not fitting).
        y_val: Validation labels.
        model_names: Subset of ``MODEL_REGISTRY``; defaults to all models.
        random_state: Seed shared by every model (reproducibility).
        mlflow_tracking: If True, each training run is logged to MLflow.

    Returns:
        Tuple of ``(pipelines, metrics)`` where ``pipelines`` maps model name to
        fitted pipeline and ``metrics`` maps model name to its validation
        metrics dict, plus the name of the best model by ``PRIMARY_METRIC``.
    """
    if model_names is None:
        model_names = list(MODEL_REGISTRY)

    pipelines: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, Any]] = {}

    for name in model_names:
        pipeline = train_model(name, X_train, y_train, random_state=random_state)
        evaluator = ModelEvaluator(pipeline)
        run_metrics = evaluator.evaluate(X_val, y_val)
        pipelines[name] = pipeline
        metrics[name] = run_metrics
        if mlflow_tracking:
            _log_mlflow_run(name, pipeline, run_metrics, random_state)

    best_name = max(metrics, key=lambda n: metrics[n][PRIMARY_METRIC])
    logger.info("Best model on validation by %s: '%s' (%.4f).",
                PRIMARY_METRIC, best_name, metrics[best_name][PRIMARY_METRIC])
    return pipelines, metrics, best_name


def run_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_names: Optional[list[str]] = None,
    random_state: int = DEFAULT_SEED,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    mlflow_tracking: bool = True,
    model_path: Path | str = "models/model.joblib",
    metrics_path: Path | str = "models/metrics.json",
) -> tuple[str, dict[str, Any]]:
    """Full orchestration: train all models, pick the best, save model + metrics.

    Models are ranked on the validation set by ``PRIMARY_METRIC``. The best one
    is then evaluated on the test set (once, at the end) at the cost-optimal
    threshold, and both the pipeline and the metrics are persisted so the README
    and the serving layer (``src.models.predict``) can consume them.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data (used for model selection only).
        X_test, y_test: Test data (used for the final report only).
        model_names: Subset of ``MODEL_REGISTRY``; defaults to all models.
        random_state: Seed shared by every model.
        fn_cost / fp_cost: Business costs used for threshold selection.
        mlflow_tracking: Passed through to ``train_all``; disable in tests.
        model_path: Where the best fitted pipeline is saved.
        metrics_path: Where the test metrics JSON is saved.

    Returns:
        Tuple of ``(best_model_name, test_metrics_dict)``.
    """
    pipelines, val_metrics, best_name = train_all(
        X_train, y_train, X_val, y_val,
        model_names=model_names, random_state=random_state,
        mlflow_tracking=mlflow_tracking,
    )
    best_pipeline = pipelines[best_name]

    # Final evaluation happens only once, on the held-out test set.
    evaluator = ModelEvaluator(best_pipeline)
    evaluator.optimal_threshold(X_val, y_val, fn_cost=fn_cost, fp_cost=fp_cost)
    test_metrics = evaluator.evaluate(X_test, y_test)
    test_metrics["best_model"] = best_name
    test_metrics["fn_cost"] = fn_cost
    test_metrics["fp_cost"] = fp_cost

    save_model(best_pipeline, model_path)
    save_metrics(test_metrics, metrics_path)
    logger.info("Saved best model '%s' to %s and metrics to %s.",
                best_name, model_path, metrics_path)
    return best_name, test_metrics


def save_model(pipeline: Pipeline, path: Path | str) -> None:
    """Persist a fitted pipeline with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Model saved to %s", path)


def save_metrics(metrics: dict[str, Any], path: Path | str) -> None:
    """Write a metrics dict (JSON-serialisable values only) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float, str, bool)) or v is None
    }
    path.write_text(json.dumps(serialisable, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metrics saved to %s", path)


def _log_mlflow_run(
    model_name: str,
    pipeline: Pipeline,
    metrics: dict[str, Any],
    random_state: int,
) -> None:
    """Log one training run to MLflow; failures degrade to a warning, not a crash.

    The pipeline is stored as a joblib artifact rather than through
    ``mlflow.sklearn.log_model``: newer MLflow versions hand sklearn models to
    skops, which refuses many sklearn objects as "untrusted types" (a moving
    target). A plain artifact keeps the model inside the run and is version-robust;
    the serving layer loads it with ``joblib`` anyway.
    """
    import tempfile

    try:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("random_state", random_state)
            mlflow.log_metrics({k: metrics[k] for k in SUMMARY_METRICS})
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                joblib.dump(pipeline, tmp.name)
                mlflow.log_artifact(tmp.name, artifact_path="model")
    except Exception:  # pragma: no cover - defensive: MLflow should not kill training
        logger.exception("MLflow logging failed for '%s'.", model_name)
