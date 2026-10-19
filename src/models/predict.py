"""Load a trained model and produce predictions, shared by the API and batch runs.

The model saved by ``src.models.train`` is a complete scikit-learn ``Pipeline``
(preprocessor + classifier) that expects a pandas DataFrame with the raw feature
columns. Every prediction entry point therefore validates the columns first, then
calls ``predict_proba`` — this single code path is used by the FastAPI endpoint
and by batch jobs over new CSV files (e.g. the unlabeled ``cs-test.csv``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.build_features import validate_feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_METRICS_PATH = "models/metrics.json"
DEFAULT_THRESHOLD = 0.5

# Column names written by predict_batch.
PROBA_COL = "default_probability"
LABEL_COL = "prediction"

TARGET_COL = "SeriousDlqin2yrs"


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a fitted pipeline saved with ``src.models.train.save_model``.

    Args:
        model_path: Path to the joblib file.

    Returns:
        The fitted ``Pipeline``.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    logger.info("Loaded model from %s (%s).", model_path, type(model).__name__)
    return model


def load_threshold(
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
    default: float = DEFAULT_THRESHOLD,
) -> float:
    """Read the decision threshold stored in the metrics file.

    Args:
        metrics_path: Path to the JSON written by ``save_metrics``.
        default: Threshold to fall back to when the file is missing.

    Returns:
        The stored (cost-optimal) threshold, or ``default``.
    """
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        logger.warning("%s not found; using default threshold %.3f.", metrics_path, default)
        return default
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = float(metrics.get("threshold", default))
    logger.info("Loaded decision threshold %.3f from %s.", threshold, metrics_path)
    return threshold


class Predictor:
    """Prediction interface shared by the API and the batch jobs."""

    def __init__(self, model: Pipeline, threshold: float = DEFAULT_THRESHOLD) -> None:
        """Wrap a fitted pipeline with a decision threshold.

        Args:
            model: Fitted ``Pipeline`` producing ``predict_proba``.
            threshold: Probability above which a row is labelled defaulter=1.
        """
        self.model = model
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(default = 1) per row, validating the feature columns first.

        Args:
            X: Feature DataFrame with the expected raw feature columns.

        Returns:
            Float array of positive-class probabilities (same length as ``X``).

        Raises:
            ValueError: If expected feature columns are missing from ``X``.
        """
        validate_feature_columns(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return 0/1 labels at ``self.threshold``."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    @classmethod
    def from_disk(
        cls,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        metrics_path: Path | str = DEFAULT_METRICS_PATH,
        threshold: Optional[float] = None,
    ) -> "Predictor":
        """Build a Predictor from the saved artifacts (or an explicit threshold).

        Args:
            model_path: Path to the joblib pipeline.
            metrics_path: Path to the metrics JSON (for the threshold).
            threshold: Explicit threshold; overrides the one stored in metrics.

        Returns:
            A ready-to-use ``Predictor``.
        """
        model = load_model(model_path)
        if threshold is None:
            threshold = load_threshold(metrics_path)
        return cls(model, threshold=threshold)


def predict_batch(
    input_path: Path | str,
    output_path: Path | str,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
    threshold: Optional[float] = None,
    index_col: int = 0,
) -> pd.DataFrame:
    """Score every row of a CSV and write the predictions to disk.

    The input CSV must contain the raw feature columns (a target column, if
    present, is dropped before scoring). The output CSV keeps the input rows and
    adds ``default_probability`` and ``prediction`` columns.

    Args:
        input_path: Path to the CSV to score (e.g. ``cs-test.csv``).
        output_path: Where to write the scored CSV.
        model_path: Path to the fitted pipeline.
        metrics_path: Path to the metrics JSON (for the decision threshold).
        threshold: Explicit decision threshold; overrides the stored one.
        index_col: CSV column to use as the row index (0 = the Unnamed id).

    Returns:
        The scored DataFrame (also written to ``output_path``).
    """
    df = pd.read_csv(input_path, index_col=index_col)
    logger.info("Loaded %d rows from %s.", len(df), input_path)

    # Test data may carry an all-NaN target column; it must not be passed to the
    # pipeline as a feature.
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    predictor = Predictor.from_disk(
        model_path=model_path, metrics_path=metrics_path, threshold=threshold
    )
    output = df.copy()
    output[PROBA_COL] = predictor.predict_proba(df)
    output[LABEL_COL] = predictor.predict(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, encoding="utf-8")
    logger.info(
        "Wrote %d predictions to %s (%.1f%% predicted defaulters).",
        len(output), output_path, 100.0 * output[LABEL_COL].mean(),
    )
    return output
