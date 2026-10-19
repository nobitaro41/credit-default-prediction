"""SHAP explanations for a single customer, shared by the API and the UI.

The deployed best model is a tree ensemble (``random_forest``), so local
attributions come from ``shap.TreeExplainer`` on the pipeline's classifier. The
preprocessor is a per-column ``ColumnTransformer`` (median imputation + standard
scaling), so the transformed feature order matches ``NUMERIC_FEATURES`` one-to-one
and every SHAP value maps back to a raw feature name.

Non-tree estimators raise ``NotImplementedError`` at build time; callers degrade
gracefully (the API returns 501) rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.build_features import NUMERIC_FEATURES
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureContribution:
    """SHAP value of one feature for a single customer."""

    feature: str
    value: float  # positive pushes default risk up, negative pushes it down


@dataclass(frozen=True)
class Explanation:
    """Full local explanation: base rate plus one contribution per feature."""

    base_value: float
    contributions: tuple[FeatureContribution, ...]  # sorted by |value|, descending


class Explainer:
    """Wrap a fitted pipeline so a single row can be explained with SHAP."""

    def __init__(self, pipeline: Pipeline) -> None:
        """Build a ``TreeExplainer`` around the pipeline's classifier.

        Args:
            pipeline: Fitted ``preprocessor + classifier`` pipeline.

        Raises:
            NotImplementedError: If the classifier is not a tree ensemble.
        """
        self._pipeline = pipeline
        self._classifier = pipeline.named_steps["classifier"]
        self._preprocessor = pipeline.named_steps["preprocessor"]
        self._explainer = self._build(pipeline)

    @staticmethod
    def _build(pipeline: Pipeline):
        import shap  # imported lazily so model imports stay light at startup

        classifier = pipeline.named_steps["classifier"]
        try:
            return shap.TreeExplainer(classifier)
        except Exception as exc:  # non-tree estimators fail while building the trees
            raise NotImplementedError(
                f"Explanations require a tree ensemble model, got "
                f"{type(classifier).__name__} ({type(exc).__name__})."
            ) from exc

    @staticmethod
    def _positive_class_value(value) -> float:
        """Pick the default (class 1) entry from a scalar/list/array base value."""
        arr = np.asarray(value).reshape(-1)
        return float(arr[1] if arr.size > 1 else arr[0])

    def explain_row(self, X: pd.DataFrame, index: int = 0) -> Explanation:
        """Explain the customer at position ``index`` of ``X``.

        Args:
            X: Feature DataFrame with the raw feature columns.
            index: Row to explain (defaults to the first/only row).

        Returns:
            An ``Explanation`` whose contributions are sorted by |SHAP value|.

        Raises:
            ValueError: If the transformed feature count does not match
                ``NUMERIC_FEATURES`` (the pipeline does not alter the layout).
        """
        row = X.iloc[[index]].copy()
        transformed = self._preprocessor.transform(row)  # shape (1, n_features)

        values = self._explainer.shap_values(transformed)
        # Binary models can return different formats:
        # - list of two arrays (one per class) -> take class 1
        # - 3D array (n_samples, n_features, 2) -> take class 1
        # - 2D array (n_samples, n_features) -> use directly
        arr = np.asarray(values)
        if arr.ndim == 3 and arr.shape[2] == 2:
            # Shape (1, n_features, 2) -> take class 1
            arr = arr[:, :, 1]
        elif isinstance(values, list) and len(values) == 2:
            # List of two arrays -> take class 1
            arr = np.asarray(values[1])
        # Now arr should be 2D: (1, n_features)
        row_values = arr.reshape(-1)

        if row_values.size != len(NUMERIC_FEATURES):
            raise ValueError(
                f"Expected {len(NUMERIC_FEATURES)} SHAP values, got {row_values.size}."
            )

        base_value = self._positive_class_value(self._explainer.expected_value)
        contributions = [
            FeatureContribution(feature=feature, value=float(row_values[position]))
            for position, feature in enumerate(NUMERIC_FEATURES)
        ]
        contributions.sort(key=lambda c: abs(c.value), reverse=True)
        return Explanation(base_value=base_value, contributions=tuple(contributions))
