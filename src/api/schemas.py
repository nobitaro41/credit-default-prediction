"""Pydantic request/response schemas for the credit default prediction API.

The request schema mirrors the raw feature columns of the Give Me Some Credit
dataset, so API payloads use the exact column names the training pipeline was
built on. Only ``MonthlyIncome`` and ``NumberOfDependents`` are optional: missing
values are imputed by the pipeline from the training distribution. Responses
carry the predicted default probability, the binary label (at a cost-optimal
threshold), and — for the explanation endpoint — per-feature SHAP values.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.features.build_features import NUMERIC_FEATURES

# Field names are snake_case Python identifiers; the raw column names (which
# contain dashes, e.g. "NumberOfTime30-59DaysPastDueNotWorse") are kept as JSON
# aliases so clients send the same keys as the dataset.
_FLOAT_FIELD_KW = {"ge": 0.0}


class PredictionInput(BaseModel):
    """One borrower record — the ten raw features the model expects."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    revolving_utilization_of_unsecured_lines: float = Field(
        alias="RevolvingUtilizationOfUnsecuredLines",
        **_FLOAT_FIELD_KW,
        description="Total credit card balances divided by total credit limits.",
    )
    age: int = Field(
        ge=18,
        le=120,
        description="Borrower age in years (adults only).",
    )
    number_of_times_30_59_days_past_due: int = Field(
        alias="NumberOfTime30-59DaysPastDueNotWorse",
        ge=0,
        description="Times 30-59 days past due (but not worse) in the last 2 years.",
    )
    debt_ratio: float = Field(
        alias="DebtRatio",
        **_FLOAT_FIELD_KW,
        description="Monthly debt payments / monthly gross income.",
    )
    monthly_income: float | None = Field(
        alias="MonthlyIncome",
        default=None,
        **_FLOAT_FIELD_KW,
        description="Monthly income (missing values are imputed by the model).",
    )
    number_of_open_credit_lines_and_loans: int = Field(
        alias="NumberOfOpenCreditLinesAndLoans",
        ge=0,
        description="Number of open credit lines and loans.",
    )
    number_of_times_90_days_late: int = Field(
        alias="NumberOfTimes90DaysLate",
        ge=0,
        description="Times 90+ days past due in the last 2 years.",
    )
    number_real_estate_loans_or_lines: int = Field(
        alias="NumberRealEstateLoansOrLines",
        ge=0,
        description="Number of real estate loans or lines of credit.",
    )
    number_of_times_60_89_days_past_due: int = Field(
        alias="NumberOfTime60-89DaysPastDueNotWorse",
        ge=0,
        description="Times 60-89 days past due (but not worse) in the last 2 years.",
    )
    number_of_dependents: float | None = Field(
        alias="NumberOfDependents",
        default=None,
        **_FLOAT_FIELD_KW,
        description="Number of dependents in the family (excluding the borrower).",
    )

    def to_dataframe(self) -> pd.DataFrame:
        """Render the record as a one-row DataFrame in raw feature order.

        Returns:
            A DataFrame whose columns are exactly ``NUMERIC_FEATURES`` in the
            order the preprocessor expects; ``None`` becomes NaN (imputed later).
        """
        raw = self.model_dump(by_alias=True)
        return pd.DataFrame([raw], columns=NUMERIC_FEATURES)


class PredictResponse(BaseModel):
    """Prediction for a single customer."""

    default_probability: float = Field(ge=0.0, le=1.0)
    prediction: Literal[0, 1]
    threshold: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "high"]


class BatchPredictRequest(BaseModel):
    """Body of the batch endpoint: up to 1000 records in one call."""

    records: list[PredictionInput] = Field(min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    """Prediction for every record of a batch call, in input order."""

    predictions: list[PredictResponse]


class FeatureContribution(BaseModel):
    """SHAP value of a single feature for one customer."""

    feature: str
    shap_value: float  # positive pushes risk up, negative pushes it down


class ExplainResponse(BaseModel):
    """Base default risk plus a per-feature SHAP breakdown."""

    default_probability: float = Field(ge=0.0, le=1.0)
    base_value: float
    features: list[FeatureContribution]


class HealthResponse(BaseModel):
    """Liveness probe with a summary of the loaded model."""

    status: Literal["ok"]
    model_loaded: bool
    estimator: str  # classifier class name, e.g. "RandomForestClassifier"
    threshold: float = Field(ge=0.0, le=1.0)
    n_features: int
    version: str
