"""Streamlit frontend for the Credit Default Prediction project.

This app provides an interactive UI to:
- Input borrower characteristics
- Get default probability and risk classification from the FastAPI backend
- Visualize SHAP feature contributions explaining the prediction
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Default API base URL (can be overridden via environment variable)
DEFAULT_API_URL = os.getenv("CDP_API_URL", "http://localhost:8000")

# Feature descriptions matching the dataset
FEATURE_CONFIG = [
    {
        "name": "RevolvingUtilizationOfUnsecuredLines",
        "label": "Revolving Utilization of Unsecured Lines",
        "help": "Total credit card balances divided by total credit limits. High values indicate heavy credit usage.",
        "min": 0.0,
        "max": 5.0,
        "default": 0.5,
        "step": 0.01,
        "format": "%.2f",
    },
    {
        "name": "age",
        "label": "Age",
        "help": "Borrower age in years (adults only).",
        "min": 18,
        "max": 100,
        "default": 45,
        "step": 1,
        "format": "%d",
    },
    {
        "name": "NumberOfTime30-59DaysPastDueNotWorse",
        "label": "Number of Times 30-59 Days Past Due",
        "help": "Times 30-59 days past due (but not worse) in the last 2 years. Values 96/98 indicate missing data.",
        "min": 0,
        "max": 98,
        "default": 0,
        "step": 1,
        "format": "%d",
    },
    {
        "name": "DebtRatio",
        "label": "Debt Ratio",
        "help": "Monthly debt payments / monthly gross income. High values indicate financial strain.",
        "min": 0.0,
        "max": 10.0,
        "default": 0.3,
        "step": 0.01,
        "format": "%.2f",
    },
    {
        "name": "MonthlyIncome",
        "label": "Monthly Income",
        "help": "Monthly gross income in USD. Leave empty (or 0) to use the model's imputed value.",
        "min": 0.0,
        "max": 500000.0,
        "default": 5000.0,
        "step": 100.0,
        "format": "%.0f",
        "optional": True,
    },
    {
        "name": "NumberOfOpenCreditLinesAndLoans",
        "label": "Number of Open Credit Lines and Loans",
        "help": "Total number of open credit lines and loans.",
        "min": 0,
        "max": 50,
        "default": 5,
        "step": 1,
        "format": "%d",
    },
    {
        "name": "NumberOfTimes90DaysLate",
        "label": "Number of Times 90+ Days Late",
        "help": "Times 90+ days past due in the last 2 years. Values 96/98 indicate missing data.",
        "min": 0,
        "max": 98,
        "default": 0,
        "step": 1,
        "format": "%d",
    },
    {
        "name": "NumberRealEstateLoansOrLines",
        "label": "Number of Real Estate Loans or Lines",
        "help": "Number of real estate loans or lines of credit.",
        "min": 0,
        "max": 20,
        "default": 1,
        "step": 1,
        "format": "%d",
    },
    {
        "name": "NumberOfTime60-89DaysPastDueNotWorse",
        "label": "Number of Times 60-89 Days Past Due",
        "help": "Times 60-89 days past due (but not worse) in the last 2 years. Values 96/98 indicate missing data.",
        "min": 0,
        "max": 98,
        "default": 0,
        "step": 1,
        "format": "%d",
    },
    {
        "name": "NumberOfDependents",
        "label": "Number of Dependents",
        "help": "Number of dependents in the family (excluding the borrower). Leave empty for model imputation.",
        "min": 0.0,
        "max": 20.0,
        "default": 0.0,
        "step": 1.0,
        "format": "%.0f",
        "optional": True,
    },
]


def render_sidebar() -> dict[str, Any]:
    """Render the sidebar with input fields and return the collected values."""
    st.sidebar.header("📋 Borrower Information")
    st.sidebar.caption("Enter the borrower's characteristics to predict default risk.")

    inputs = {}
    for feat in FEATURE_CONFIG:
        name = feat["name"]
        label = feat["label"]
        help_text = feat["help"]
        min_val = feat["min"]
        max_val = feat["max"]
        default = feat["default"]
        step = feat["step"]
        fmt = feat["format"]
        is_optional = feat.get("optional", False)

        if is_optional:
            # For optional fields, use a checkbox to enable/disable
            enabled = st.sidebar.checkbox(f"Provide {label}", value=True, key=f"enable_{name}")
            if enabled:
                if isinstance(default, int) or (isinstance(default, float) and default == int(default)):
                    val = st.sidebar.number_input(
                        label,
                        min_value=min_val,
                        max_value=max_val,
                        value=int(default),
                        step=step,
                        format=fmt,
                        help=help_text,
                        key=name,
                    )
                else:
                    val = st.sidebar.number_input(
                        label,
                        min_value=min_val,
                        max_value=max_val,
                        value=default,
                        step=step,
                        format=fmt,
                        help=help_text,
                        key=name,
                    )
                inputs[name] = val
            else:
                inputs[name] = None
        else:
            if isinstance(default, int) or (isinstance(default, float) and default == int(default)):
                val = st.sidebar.number_input(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=int(default),
                    step=step,
                    format=fmt,
                    help=help_text,
                    key=name,
                )
            else:
                val = st.sidebar.number_input(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default,
                    step=step,
                    format=fmt,
                    help=help_text,
                    key=name,
                )
            inputs[name] = val

    st.sidebar.divider()

    # API configuration
    st.sidebar.header("⚙️ API Settings")
    api_url = st.sidebar.text_input(
        "API Base URL",
        value=DEFAULT_API_URL,
        help="Base URL of the FastAPI backend (e.g., http://localhost:8000)",
    )

    # Threshold override
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Override the model's stored cost-optimal threshold. Higher = fewer predicted defaulters.",
    )

    predict_button = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)

    return {
        "inputs": inputs,
        "api_url": api_url,
        "threshold": threshold,
        "predict_button": predict_button,
    }


def call_predict_api(api_url: str, inputs: dict[str, Any], threshold: float) -> dict[str, Any] | None:
    """Call the FastAPI /predict endpoint."""
    # Prepare payload - convert to the exact field names the API expects
    # The API expects camelCase aliases matching the dataset column names
    payload = {}
    for feat in FEATURE_CONFIG:
        name = feat["name"]
        val = inputs.get(name)
        if val is not None:
            # Map python snake_case to the API's alias (the raw column name)
            # The API uses aliases like "RevolvingUtilizationOfUnsecuredLines"
            alias = feat["name"]  # This is already the raw column name
            payload[alias] = val
        elif feat.get("optional", False):
            # Optional fields can be omitted or set to null
            payload[feat["name"]] = None

    try:
        response = requests.post(
            f"{api_url}/predict",
            params={"threshold": threshold} if threshold != 0.5 else None,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at `{api_url}`. Make sure the FastAPI server is running.")
        st.info("Run `make serve` in another terminal to start the API server.")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. The API took too long to respond.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


def call_explain_api(api_url: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Call the FastAPI /explain endpoint for SHAP values."""
    payload = {}
    for feat in FEATURE_CONFIG:
        name = feat["name"]
        val = inputs.get(name)
        if val is not None:
            payload[feat["name"]] = val
        elif feat.get("optional", False):
            payload[feat["name"]] = None

    try:
        response = requests.post(
            f"{api_url}/explain",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 501:
            st.warning("⚠️ Explanations unavailable: the deployed model is not a tree ensemble.")
        else:
            st.error(f"❌ Explanation API error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception:
        return None


def render_prediction_result(result: dict[str, Any]) -> None:
    """Display the prediction result with gauge and risk level."""
    proba = result["default_probability"]
    prediction = result["prediction"]
    threshold = result["threshold"]
    risk_level = result["risk_level"]

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Gauge chart for probability
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=proba * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Default Probability (%)", "font": {"size": 20}},
                delta={"reference": threshold * 100, "relative": False, "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                    "bar": {"color": "darkred" if proba >= threshold else "darkgreen"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, threshold * 100], "color": "#d4edda"},
                        {"range": [threshold * 100, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": threshold * 100,
                    },
                },
            )
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Risk level badge
    if risk_level == "high":
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem; background-color: #f8d7da;
                        border-radius: 0.5rem; border: 1px solid #f5c6cb; margin-top: 1rem;">
                <h2 style="color: #721c24; margin: 0;">⚠️ HIGH RISK</h2>
                <p style="color: #721c24; margin: 0.5rem 0 0 0;">
                    Predicted to default (probability: {proba:.1%} ≥ threshold: {threshold:.1%})
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem; background-color: #d4edda;
                        border-radius: 0.5rem; border: 1px solid #c3e6cb; margin-top: 1rem;">
                <h2 style="color: #155724; margin: 0;">✅ LOW RISK</h2>
                <p style="color: #155724; margin: 0.5rem 0 0 0;">
                    Predicted to repay (probability: {proba:.1%} < threshold: {threshold:.1%})
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Metrics row
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Default Probability", f"{proba:.2%}")
    mcol2.metric("Prediction", "Default (1)" if prediction == 1 else "No Default (0)")
    mcol3.metric("Decision Threshold", f"{threshold:.2%}")
    mcol4.metric("Risk Level", risk_level.upper())


def render_shap_explanation(explanation: dict[str, Any]) -> None:
    """Render SHAP explanation as a horizontal bar chart."""
    features = explanation["features"]
    base_value = explanation["base_value"]

    if not features:
        st.info("No SHAP values available.")
        return

    # Prepare data for plotting
    df_shap = pd.DataFrame([
        {"feature": f["feature"], "shap_value": f["shap_value"]}
        for f in features
    ])

    # Sort by absolute SHAP value for better visualization
    df_shap["abs_shap"] = df_shap["shap_value"].abs()
    df_shap = df_shap.sort_values("abs_shap", ascending=True)

    # Create horizontal bar chart
    colors = ["#dc3545" if v > 0 else "#28a745" for v in df_shap["shap_value"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df_shap["feature"],
            x=df_shap["shap_value"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in df_shap["shap_value"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>SHAP value: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"SHAP Feature Contributions (Base value: {base_value:.4f})",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="SHAP Value (positive = increases default risk)",
        yaxis_title="Feature",
        height=max(400, len(df_shap) * 35),
        margin=dict(l=200, r=50, t=60, b=40),
        showlegend=False,
    )

    # Add a vertical line at x=0
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    st.plotly_chart(fig, use_container_width=True)

    # Feature importance table
    with st.expander("📊 View SHAP Values Table"):
        df_display = df_shap[["feature", "shap_value"]].copy()
        df_display.columns = ["Feature", "SHAP Value"]
        df_display["Impact"] = df_display["SHAP Value"].apply(
            lambda x: "🔴 Increases Risk" if x > 0 else "🟢 Decreases Risk"
        )
        df_display["SHAP Value"] = df_display["SHAP Value"].apply(lambda x: f"{x:+.6f}")
        st.dataframe(df_display, use_container_width=True, hide_index=True)


def render_feature_summary(inputs: dict[str, Any]) -> None:
    """Render a summary of the input features."""
    with st.expander("📋 Input Summary", expanded=False):
        # Create a nice display of all inputs
        summary_data = []
        for feat in FEATURE_CONFIG:
            name = feat["name"]
            label = feat["label"]
            val = inputs.get(name)
            if val is not None:
                if isinstance(val, float):
                    display_val = f"{val:,.2f}"
                else:
                    display_val = f"{val:,}"
            else:
                display_val = "*(imputed by model)*"
            summary_data.append({"Feature": label, "Value": display_val})

        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)


def check_api_health(api_url: str) -> dict[str, Any] | None:
    """Check if the API is healthy and return model info."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def main():
    """Main Streamlit application."""
    # Custom CSS for better styling
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stButton > button {
            width: 100%;
        }
        .metric-container {
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.title("💳 Credit Default Prediction")
    st.markdown(
        """
        This application predicts whether a borrower will experience a **serious delinquency
        (90+ days past due) within 2 years**, using a machine learning model trained on the
        *Give Me Some Credit* dataset (150,000 borrowers).
        """
    )

    # Check API health on startup
    sidebar_data = render_sidebar()
    api_url = sidebar_data["api_url"]

    # Health check indicator
    health = check_api_health(api_url)
    if health:
        st.sidebar.success(f"✅ API Connected ({health.get('estimator', 'Unknown')})")
    else:
        st.sidebar.error("❌ API Not Connected")
        st.sidebar.caption(f"Trying: {api_url}")

    # Main content area
    if sidebar_data["predict_button"]:
        with st.spinner("🔮 Getting prediction from model..."):
            result = call_predict_api(api_url, sidebar_data["inputs"], sidebar_data["threshold"])

        if result:
            st.subheader("🎯 Prediction Result")
            render_prediction_result(result)

            # Get SHAP explanation
            with st.spinner("🔍 Computing SHAP explanations..."):
                explanation = call_explain_api(api_url, sidebar_data["inputs"])

            if explanation:
                st.subheader("🔍 Model Explanation (SHAP)")
                st.caption(
                    "SHAP values show how each feature pushes the prediction away from the "
                    "base rate. **Red bars increase default risk**, **green bars decrease it**."
                )
                render_shap_explanation(explanation)
            else:
                st.info(
                    "SHAP explanations are only available for tree-based models "
                    "(Random Forest, XGBoost, LightGBM). The current model may not support this."
                )

            # Show input summary
            render_feature_summary(sidebar_data["inputs"])

    else:
        # Welcome screen when no prediction has been made
        st.info("👈 Enter borrower information in the sidebar and click **Predict** to get started.")

        # Show model info if available
        if health:
            with st.expander("ℹ️ Model Information", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Model Type", health.get("estimator", "Unknown"))
                col2.metric("Decision Threshold", f"{health.get('threshold', 0):.2%}")
                col3.metric("Features", health.get("n_features", "Unknown"))

        # Sample borrower profiles
        st.subheader("📝 Try Sample Profiles")
        st.caption("Click a button below to fill the form with a sample borrower profile.")

        sample_profiles = {
            "Low Risk": {
                "RevolvingUtilizationOfUnsecuredLines": 0.15,
                "age": 45,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.2,
                "MonthlyIncome": 8000,
                "NumberOfOpenCreditLinesAndLoans": 3,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 1.0,
            },
            "Medium Risk": {
                "RevolvingUtilizationOfUnsecuredLines": 0.65,
                "age": 35,
                "NumberOfTime30-59DaysPastDueNotWorse": 1,
                "DebtRatio": 0.45,
                "MonthlyIncome": 3500,
                "NumberOfOpenCreditLinesAndLoans": 8,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2.0,
            },
            "High Risk": {
                "RevolvingUtilizationOfUnsecuredLines": 1.2,
                "age": 28,
                "NumberOfTime30-59DaysPastDueNotWorse": 3,
                "DebtRatio": 0.85,
                "MonthlyIncome": 2000,
                "NumberOfOpenCreditLinesAndLoans": 12,
                "NumberOfTimes90DaysLate": 2,
                "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 1,
                "NumberOfDependents": 3.0,
            },
        }

        cols = st.columns(len(sample_profiles))
        for idx, (profile_name, profile_data) in enumerate(sample_profiles.items()):
            with cols[idx]:
                if st.button(profile_name, use_container_width=True, key=f"sample_{profile_name}"):
                    # Store in session state to pre-fill the form
                    for feat in FEATURE_CONFIG:
                        name = feat["name"]
                        if name in profile_data:
                            st.session_state[name] = profile_data[name]
                    st.rerun()

        # Footer
        st.divider()
        st.markdown(
            """
            ---
            **About this project:** This is a portfolio project demonstrating a complete ML pipeline
            for credit default prediction, including data preprocessing, model training (Logistic Regression,
            Random Forest, XGBoost, LightGBM), hyperparameter tuning with Optuna, model explanation with SHAP,
            and deployment with FastAPI + Streamlit + Docker.

            **Dataset:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle)
            """
        )


if __name__ == "__main__":
    main()
