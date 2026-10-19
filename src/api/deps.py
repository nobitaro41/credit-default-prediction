"""FastAPI dependencies — load heavyweight artefacts once per process.

The fitted pipeline and the SHAP explainer are cached with ``lru_cache`` so the
first request pays the load cost and every later one reuses the objects. The
model/metrics paths default to the repository layout and can be overridden with
the ``CDP_MODEL_PATH`` / ``CDP_METRICS_PATH`` environment variables (used to
point the Docker image at mounted artifacts).
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends

from src.models.explain import Explainer
from src.models.predict import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, Predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = os.getenv("CDP_MODEL_PATH", DEFAULT_MODEL_PATH)
METRICS_PATH = os.getenv("CDP_METRICS_PATH", DEFAULT_METRICS_PATH)


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Load (once) the fitted pipeline plus the cost-optimal decision threshold."""
    logger.info("Loading model from %s (metrics: %s).", MODEL_PATH, METRICS_PATH)
    return Predictor.from_disk(model_path=MODEL_PATH, metrics_path=METRICS_PATH)


@lru_cache(maxsize=1)
def get_explainer(predictor: Predictor = Depends(get_predictor)) -> Explainer | None:
    """Return a cached SHAP explainer for the loaded model, or ``None``.

    ``None`` means the model cannot be explained (non-tree estimator); the
    endpoint translates that into a 501 response instead of failing the health
    of the whole service.
    """
    try:
        return Explainer(predictor.model)
    except NotImplementedError as exc:
        logger.warning("Explanations unavailable: %s", exc)
        return None
