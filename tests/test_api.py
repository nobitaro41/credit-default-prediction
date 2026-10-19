"""Tests for the FastAPI service (src.api: schemas, deps, endpoints).

A small real model is trained on synthetic data and injected through
``app.dependency_overrides`` so no heavy model file or external service is
needed. A tree model (random_forest) covers every endpoint including SHAP;
a linear baseline covers the graceful "not explainable" fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_predictor
from src.api.main import app
from src.features.build_features import NUMERIC_FEATURES
from src.models.predict import Predictor
from src.models.train import train_model


def _make_data(n_rows: int = 120, seed: int = 5):
    """Deterministic, separable features + binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.normal(size=n_rows) for c in NUMERIC_FEATURES})
    y = (X["age"] + X["DebtRatio"] > 0).astype(int)
    return X, y


def _valid_payload(overrides: dict | None = None) -> dict:
    """A valid raw-feature payload using the dataset's exact column names."""
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 35,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.3,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2.0,
    }
    if overrides:
        payload.update(overrides)
    return payload


@pytest.fixture(scope="module")
def tree_predictor() -> Predictor:
    """A fitted tree pipeline so every endpoint (including SHAP) works."""
    X, y = _make_data()
    pipeline = train_model("random_forest", X, y, random_state=1)
    return Predictor(pipeline, threshold=0.5)


@pytest.fixture()
def client(tree_predictor):
    """TestClient with the tree model injected via dependency override."""
    app.dependency_overrides[get_predictor] = lambda: tree_predictor
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestHealth:
    """GET /health"""

    def test_returns_ok_and_model_summary(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["threshold"] == 0.5
        assert body["n_features"] == len(NUMERIC_FEATURES)


class TestPredict:
    """POST /predict"""

    def test_single_valid_record(self, client):
        resp = client.post("/predict", json=_valid_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["default_probability"] <= 1.0
        assert body["prediction"] in (0, 1)
        assert body["threshold"] == 0.5
        assert body["risk_level"] in ("low", "high")

    def test_accepts_snake_case_field_names(self, client):
        """populate_by_name means snake_case keys resolve to the same aliases."""
        payload = _valid_payload()
        payload["monthly_income"] = payload.pop("MonthlyIncome")
        assert client.post("/predict", json=payload).status_code == 200

    def test_threshold_query_override(self, client):
        resp = client.post("/predict", json=_valid_payload(), params={"threshold": 0.9})
        assert resp.status_code == 200
        assert resp.json()["threshold"] == 0.9

    def test_out_of_range_threshold_is_rejected(self, client):
        resp = client.post("/predict", json=_valid_payload(), params={"threshold": 1.5})
        assert resp.status_code == 422

    def test_missing_required_column_is_rejected(self, client):
        payload = _valid_payload()
        del payload["age"]
        assert client.post("/predict", json=payload).status_code == 422

    def test_negative_age_is_rejected(self, client):
        assert client.post("/predict", json=_valid_payload({"age": -1})).status_code == 422

    def test_missing_optional_column_is_imputed(self, client):
        """"MonthlyIncome"/"NumberOfDependents" may be absent (or null)."""
        payload = _valid_payload({"MonthlyIncome": None})
        assert client.post("/predict", json=payload).status_code == 200


class TestPredictBatch:
    """POST /predict/batch"""

    def test_scores_multiple_records(self, client):
        resp = client.post(
            "/predict/batch",
            json={"records": [_valid_payload(), _valid_payload({"age": 50})]},
        )
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 2

    def test_empty_batch_is_rejected(self, client):
        assert client.post("/predict/batch", json={"records": []}).status_code == 422


class TestExplain:
    """POST /explain"""

    def test_single_customer_breakdown(self, client):
        resp = client.post("/explain", json=_valid_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["default_probability"] <= 1.0
        assert isinstance(body["base_value"], float)
        features = body["features"]
        assert len(features) == len(NUMERIC_FEATURES)
        assert {f["feature"] for f in features} == set(NUMERIC_FEATURES)


class TestExplainUnsupported:
    """A non-tree model cannot be explained; the endpoint degrades to 501."""

    def test_linear_model_returns_501(self):
        X, y = _make_data()
        pipeline = train_model("logistic_regression", X, y, random_state=1)
        linear_predictor = Predictor(pipeline, threshold=0.5)
        app.dependency_overrides[get_predictor] = lambda: linear_predictor
        try:
            with TestClient(app) as test_client:
                assert test_client.post("/explain", json=_valid_payload()).status_code == 501
                assert test_client.post("/predict", json=_valid_payload()).status_code == 200
        finally:
            app.dependency_overrides.clear()


def test_root_points_to_docs(client):
    assert client.get("/").status_code == 200
