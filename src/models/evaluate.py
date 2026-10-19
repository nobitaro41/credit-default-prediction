"""Model evaluation metrics and business-cost threshold selection.

The target is imbalanced (~6.7% defaulters), so the evaluation never relies on
accuracy alone: it reports ROC-AUC, PR-AUC, F1, recall, precision and the Brier
score (calibration). The decision threshold is chosen by minimising the expected
business cost — missing a defaulter (false negative) is costlier than wrongly
rejecting a good customer (false positive).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config import get_config_value, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Relative cost of each misclassification loaded from configs/config.yaml with fallbacks.
_cfg = load_config()
DEFAULT_FN_COST = float(get_config_value(_cfg, "evaluation.fn_cost", 5.0))
DEFAULT_FP_COST = float(get_config_value(_cfg, "evaluation.fp_cost", 1.0))

# Metrics that are finite scalars and can be logged to MLflow / written to JSON.
SUMMARY_METRICS = ("roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy", "brier")


def _expected_cost(y_true: pd.Series, y_pred: np.ndarray, fn_cost: float, fp_cost: float) -> float:
    """Weighted cost of the confusion matrix at a given threshold."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn_cost * fn + fp_cost * fp


class ModelEvaluator:
    """Evaluate a fitted classifier and choose the cost-optimal threshold."""

    def __init__(self, model: Any, threshold: float = 0.5) -> None:
        """Wrap a fitted model for evaluation.

        Args:
            model: Fitted estimator or Pipeline exposing ``predict`` and (ideally)
                ``predict_proba``.
            threshold: Decision threshold used by ``predict`` (default 0.5).
        """
        self.model = model
        self.threshold = threshold

    def positive_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(default = 1) for each row in ``X``."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict 0/1 labels using ``self.threshold``."""
        if hasattr(self.model, "predict_proba"):
            return (self.positive_probabilities(X) >= self.threshold).astype(int)
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Compute the full metric set for ``X``/``y`` at ``self.threshold``.

        Args:
            X: Feature DataFrame.
            y: True binary labels aligned with ``X``.

        Returns:
            Dict with ``roc_auc``, ``pr_auc``, ``f1``, ``recall``, ``precision``,
            ``accuracy``, ``brier`` and ``confusion_matrix`` (list of lists).
        """
        y_proba = self.positive_probabilities(X)
        y_pred = self.predict(X)
        metrics = {
            "roc_auc": roc_auc_score(y, y_proba),
            "pr_auc": average_precision_score(y, y_proba),
            "f1": f1_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "accuracy": accuracy_score(y, y_pred),
            "brier": brier_score_loss(y, y_proba),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "threshold": self.threshold,
        }
        logger.info(
            "Metrics at threshold %.3f -> ROC-AUC %.4f | PR-AUC %.4f | F1 %.4f | recall %.4f | "
            "precision %.4f | Brier %.4f",
            self.threshold, metrics["roc_auc"], metrics["pr_auc"], metrics["f1"],
            metrics["recall"], metrics["precision"], metrics["brier"],
        )
        return metrics

    def optimal_threshold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fn_cost: float = DEFAULT_FN_COST,
        fp_cost: float = DEFAULT_FP_COST,
    ) -> float:
        """Return the threshold minimising the expected misclassification cost.

        The threshold is searched over the unique predicted probabilities, so the
        returned value is a real decision boundary of the model.

        Args:
            X: Feature DataFrame (validation or test set).
            y: True binary labels aligned with ``X``.
            fn_cost: Cost of a false negative (missed defaulter).
            fp_cost: Cost of a false positive (rejected good customer).

        Returns:
            Float threshold in [0, 1] with the lowest expected cost, which also
            becomes ``self.threshold``.
        """
        y_proba = self.positive_probabilities(X)
        candidates = np.unique(np.concatenate([y_proba, [0.0, 1.0]]))
        costs = np.array(
            [_expected_cost(y, y_proba >= t, fn_cost, fp_cost) for t in candidates]
        )
        best_idx = int(costs.argmin())
        self.threshold = float(candidates[best_idx])
        logger.info(
            "Optimal threshold %.3f (expected cost %.1f at FN=%.1f / FP=%.1f).",
            self.threshold, costs[best_idx], fn_cost, fp_cost,
        )
        return self.threshold
