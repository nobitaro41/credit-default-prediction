"""Feature engineering and the preprocessing pipeline.

The pipeline follows the no-leakage rule: every transformer (median imputer,
standard scaler) is fit only on the training set, and lives inside a scikit-learn
``Pipeline`` that is applied identically to train/validation/test later. Missing
value imputation is done here (not in ``src.data.clean``) so it is estimated from
the training distribution only.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load import EXPECTED_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COL = "SeriousDlqin2yrs"

# Every raw feature is numeric, so a single numeric transformer covers the whole
# feature space (this also keeps the ColumnTransformer easy to extend with
# categorical encoders later, if needed).
NUMERIC_FEATURES = [c for c in EXPECTED_COLUMNS if c != TARGET_COL]


def build_preprocessor() -> TransformerMixin:
    """Build the transformer that imputes and scales all numeric features.

    Returns:
        A scikit-learn ``ColumnTransformer`` doing median imputation then
        standard scaling over ``NUMERIC_FEATURES``. It is stateless until fit.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    logger.info(
        "Num preprocessor: median imputation + standard scaling over %d features.",
        len(NUMERIC_FEATURES),
    )
    return ColumnTransformer(transformers=[("num", numeric_transformer, NUMERIC_FEATURES)])


def make_full_pipeline(classifier: object) -> Pipeline:
    """Combine the preprocessor with a classifier into a single fit-able pipeline.

    Args:
        classifier: Any scikit-learn-compatible estimator implementing ``fit``
            and ``predict`` (e.g. ``LogisticRegression``, ``XGBClassifier``).

    Returns:
        A ``Pipeline`` of ``[("preprocessor", ...), ("classifier", ...)]``. Fitting
        this pipeline on the training set fits the imputer/scaler on train only.
    """
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", classifier)])


def validate_feature_columns(X: pd.DataFrame) -> None:
    """Ensure the input DataFrame contains all expected feature columns.

    Args:
        X: Feature DataFrame (no target column).

    Raises:
        ValueError: If any expected feature column is missing from ``X``.
    """
    missing = set(NUMERIC_FEATURES) - set(X.columns)
    if missing:
        raise ValueError(f"Missing expected feature columns: {sorted(missing)}")
    logger.info("Feature columns validated (%d expected).", len(NUMERIC_FEATURES))
