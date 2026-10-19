"""Hyper-parameter tuning with Optuna and stratified cross-validation.

Tuning is confined to the training set (the validation/test sets are never
touched during the search), because folding the training set gives an unbiased
estimate of each trial while the test set stays sealed for final reporting. The
objective is the mean out-of-fold PR-AUC, matching the way the models are ranked
in ``src.models.train``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.features.build_features import make_full_pipeline
from src.models.train import DEFAULT_SEED, MODEL_REGISTRY
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_search_space(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Suggest a hyper-parameter combination for one trial of ``model_name``.

    Args:
        trial: The Optuna trial to sample from.
        model_name: Key into ``MODEL_REGISTRY``.

    Returns:
        Dict of estimator keyword arguments sampled by the trial.

    Raises:
        ValueError: If ``model_name`` is unknown.
    """
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    raise ValueError(
        f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}"
    )


def objective(
    trial: optuna.Trial,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    random_state: int,
) -> float:
    """Optuna objective: mean out-of-fold PR-AUC of one hyper-parameter combo.

    Args:
        trial: The trial whose parameters are tested.
        model_name: Key into ``MODEL_REGISTRY``.
        X_train, y_train: Training data (the only data the search may use).
        cv_folds: Number of stratified folds.
        random_state: Seed for the estimator and the fold shuffling.

    Returns:
        Mean PR-AUC across folds (higher is better).
    """
    entry = MODEL_REGISTRY[model_name]
    params = {**entry["params"], **get_search_space(trial, model_name)}
    params["random_state"] = random_state

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        if y_val.nunique() < 2:  # PR-AUC is undefined on a single-class fold
            continue
        pipe = make_full_pipeline(entry["class"](**params))
        pipe.fit(X_fold, y_fold)
        y_proba = pipe.predict_proba(X_val)[:, 1]
        fold_scores.append(average_precision_score(y_val, y_proba))

    mean_pr_auc = float(np.mean(fold_scores)) if fold_scores else 0.0
    return mean_pr_auc


def build_tuned_pipeline(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, Any],
    random_state: int = DEFAULT_SEED,
) -> Pipeline:
    """Fit the full pipeline on ``X_train`` using chosen ``best_params``.

    The imputer/scaler in the pipeline are re-fit on the training set here, so
    nothing from the cross-validation folds leaks into the final model.

    Args:
        model_name: Key into ``MODEL_REGISTRY``.
        X_train, y_train: Training data.
        best_params: Hyper-parameters returned by a tuned Optuna study.
        random_state: Seed for the estimator.

    Returns:
        Fitted ``Pipeline`` (preprocessor + classifier) using ``best_params``.
    """
    entry = MODEL_REGISTRY[model_name]
    params = {**entry["params"], **dict(best_params), "random_state": random_state}
    pipeline = make_full_pipeline(entry["class"](**params))
    pipeline.fit(X_train, y_train)
    logger.info("Built tuned '%s' pipeline on %d rows.", model_name, len(X_train))
    return pipeline


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 20,
    cv_folds: int = 4,
    random_state: int = DEFAULT_SEED,
    timeout: Optional[float] = None,
    show_progress_bar: bool = False,
) -> tuple[dict[str, Any], float, Pipeline]:
    """Run an Optuna study over ``model_name`` and fit its best configuration.

    Args:
        model_name: Key into ``MODEL_REGISTRY``.
        X_train, y_train: Training data used for (and only for) the search.
        n_trials: Number of hyper-parameter combinations to try.
        cv_folds: Number of stratified folds per trial.
        random_state: Seed shared by the study, the folds and the estimators.
        timeout: Optional time budget in seconds for the whole study.
        show_progress_bar: Whether Optuna should render a progress bar.

    Returns:
        Tuple ``(best_params, best_mean_pr_auc, best_pipeline)`` where
        ``best_pipeline`` is the full pipeline fit on ``X_train`` with the best
        parameters found by the study.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")

    study = optuna.create_study(direction="maximize", study_name=f"tune-{model_name}")
    study.optimize(
        lambda trial: objective(trial, model_name, X_train, y_train, cv_folds, random_state),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
    )

    best_params = study.best_params
    best_value = float(study.best_value)
    logger.info(
        "Tuning '%s' finished: best PR-AUC %.4f with params %s",
        model_name, best_value, best_params,
    )

    best_pipeline = build_tuned_pipeline(model_name, X_train, y_train, best_params, random_state)
    return best_params, best_value, best_pipeline
