"""FastAPI service exposing the credit default model.

Endpoints:
- ``GET  /health``        liveness and a summary of the loaded model
- ``GET  /``              service summary (links to the interactive docs)
- ``POST /predict``       score a single borrower
- ``POST /predict/batch`` score a batch of borrowers
- ``POST /explain``       SHAP explanation for a single borrower

The decision threshold defaults to the cost-optimal value stored with the model
and can be overridden per request with the ``threshold`` query parameter.
"""

from __future__ import annotations

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.deps import get_explainer, get_predictor
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    PredictionInput,
    PredictResponse,
)
from src.features.build_features import NUMERIC_FEATURES
from src.models.explain import Explainer
from src.models.predict import Predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

APP_VERSION = "0.1.0"

app = FastAPI(
    title="Credit Default Prediction API",
    description=(
        "Binary classifier predicting whether a borrower defaults within two years "
        "(Give Me Some Credit dataset). Returns the default probability, the binary "
        "label at a cost-optimal threshold, and an optional SHAP explanation."
    ),
    version=APP_VERSION,
)

# The Streamlit UI runs on another port, so the browser makes cross-origin calls
# to this service. Open CORS is fine for this demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    """Minimal landing point pointing to the interactive docs."""
    return {"service": "Credit Default Prediction API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, summary="Service health & model info")
def health(predictor: Predictor = Depends(get_predictor)) -> HealthResponse:
    """Return liveness plus a summary of the deployed model."""
    estimator = predictor.model.named_steps["classifier"].__class__.__name__
    return HealthResponse(
        status="ok",
        model_loaded=True,
        estimator=estimator,
        threshold=predictor.threshold,
        n_features=len(NUMERIC_FEATURES),
        version=APP_VERSION,
    )


@app.post("/predict", response_model=PredictResponse, summary="Score a single borrower")
def predict(
    record: PredictionInput,
    threshold: float | None = Query(None, ge=0.0, le=1.0, description="Optional override of the stored decision threshold."),
    predictor: Predictor = Depends(get_predictor),
) -> PredictResponse:
    """Score one record and label it at the given (or stored) threshold."""
    proba = float(predictor.predict_proba(record.to_dataframe())[0])
    used_threshold = predictor.threshold if threshold is None else threshold
    label = int(proba >= used_threshold)
    return PredictResponse(
        default_probability=proba,
        prediction=label,
        threshold=used_threshold,
        risk_level="high" if label else "low",
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, summary="Score a batch of borrowers")
def predict_batch(
    batch: BatchPredictRequest,
    threshold: float | None = Query(None, ge=0.0, le=1.0, description="Optional override of the stored decision threshold."),
    predictor: Predictor = Depends(get_predictor),
) -> BatchPredictResponse:
    """Score up to 1000 records in a single call (one response per input row)."""
    features = batch.records[0].to_dataframe()
    for record in batch.records[1:]:
        features = pd.concat([features, record.to_dataframe()], ignore_index=True)
    probas = predictor.predict_proba(features)
    used_threshold = predictor.threshold if threshold is None else threshold
    return BatchPredictResponse(
        predictions=[
            PredictResponse(
                default_probability=float(proba),
                prediction=int(proba >= used_threshold),
                threshold=used_threshold,
                risk_level="high" if proba >= used_threshold else "low",
            )
            for proba in probas
        ]
    )


@app.post("/explain", response_model=ExplainResponse, summary="SHAP explanation for a borrower")
def explain(
    record: PredictionInput,
    predictor: Predictor = Depends(get_predictor),
    explainer: Explainer | None = Depends(get_explainer),
) -> ExplainResponse:
    """Explain the model's decision: base default risk plus per-feature SHAP values."""
    if explainer is None:
        raise HTTPException(
            status_code=501,
            detail="Explanations require a tree ensemble model; serve a tree model "
            "(e.g. random_forest) to enable this endpoint.",
        )
    proba = float(predictor.predict_proba(record.to_dataframe())[0])
    explanation = explainer.explain_row(record.to_dataframe())
    return ExplainResponse(
        default_probability=proba,
        base_value=explanation.base_value,
        features=[
            FeatureContribution(feature=contribution.feature, shap_value=contribution.value)
            for contribution in explanation.contributions
        ],
    )
