"""Generate visual figures for model evaluation and interpretability.

Generates:
1. ROC Curve (reports/figures/roc_curve.png)
2. Precision-Recall Curve (reports/figures/pr_curve.png)
3. Confusion Matrix at optimal threshold (reports/figures/confusion_matrix.png)
4. Calibration / Reliability Curve (reports/figures/calibration_curve.png)
5. SHAP Global Feature Importance Summary (reports/figures/shap_summary.png)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.features.build_features import NUMERIC_FEATURES
from src.utils.config import get_config_value, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_roc_curve(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    output_path: Path | str,
    threshold: float = 0.2,
) -> None:
    """Plot and save the Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    # Find point closest to the chosen operating threshold
    idx = np.argmin(np.abs(thresholds - threshold))

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")
    plt.scatter(
        [fpr[idx]],
        [tpr[idx]],
        color="#d62728",
        s=80,
        zorder=5,
        label=f"Operating Point (Threshold = {threshold:.2f})",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved ROC curve to %s", output_path)


def generate_pr_curve(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    output_path: Path | str,
    threshold: float = 0.2,
) -> None:
    """Plot and save the Precision-Recall (PR) curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline = np.mean(y_true)

    # Operating point on PR curve
    # thresholds array has length len(precision) - 1
    idx = min(np.argmin(np.abs(thresholds - threshold)), len(recall) - 1)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#2ca02c", lw=2.5, label=f"PR Curve (PR-AUC = {pr_auc:.4f})")
    plt.axhline(
        y=baseline,
        color="#7f7f7f",
        lw=1.5,
        linestyle="--",
        label=f"No-Skill Baseline ({baseline * 100:.2f}%)",
    )
    plt.scatter(
        [recall[idx]],
        [precision[idx]],
        color="#d62728",
        s=80,
        zorder=5,
        label=f"Operating Point (Threshold = {threshold:.2f})",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall (PR) Curve — Imbalanced Target", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved PR curve to %s", output_path)


def generate_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    output_path: Path | str,
    threshold: float = 0.2,
) -> None:
    """Plot and save the confusion matrix evaluated at the decision threshold."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    labels = [
        [f"TN\n{tn:,}\n({tn / total:.1%})", f"FP (Reject Good)\n{fp:,}\n({fp / total:.1%})"],
        [f"FN (Miss Defaulter)\n{fn:,}\n({fn / total:.1%})", f"TP\n{tp:,}\n({tp / total:.1%})"],
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # Show text in each cell
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > (cm.max() / 2.0) else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=11, fontweight="medium")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Non-Default (0)", "Predicted: Default (1)"], fontsize=10)
    ax.set_yticklabels(["Actual: Non-Default (0)", "Actual: Default (1)"], fontsize=10)
    ax.set_title(
        f"Confusion Matrix @ Cost-Optimal Threshold ({threshold:.2f})\n"
        f"Total Samples = {total:,} | Positive Prevalence = {np.mean(y_true):.1%}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix to %s", output_path)


def generate_calibration_curve(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    output_path: Path | str,
) -> None:
    """Plot and save the calibration / reliability curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    brier = brier_score_loss(y_true, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="s", lw=2, color="#9467bd", label=f"Model (Brier = {brier:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", label="Perfectly Calibrated")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives (Empirical Risk)", fontsize=11)
    plt.title("Calibration / Reliability Diagram", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved calibration curve to %s", output_path)


def generate_shap_summary(
    pipeline,
    X: pd.DataFrame,
    output_path: Path | str,
    max_samples: int = 500,
) -> None:
    """Compute and save a global SHAP feature importance summary plot."""
    import shap

    logger.info("Computing SHAP values on %d samples...", min(len(X), max_samples))
    sample = X.sample(n=min(len(X), max_samples), random_state=42)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform features with the fitted preprocessor
    transformed = preprocessor.transform(sample)

    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed)

    # Format values for binary classification (take class 1)
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[2] == 2:
        shap_matrix = arr[:, :, 1]
    elif isinstance(values, list) and len(values) == 2:
        shap_matrix = np.asarray(values[1])
    else:
        shap_matrix = arr

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_matrix,
        features=sample[NUMERIC_FEATURES],
        feature_names=NUMERIC_FEATURES,
        show=False,
        plot_type="dot",
        max_display=10,
    )
    plt.title("SHAP Global Feature Importance (Beeswarm Plot)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot to %s", output_path)


def main() -> int:
    """Generate all evaluation and interpretability figures."""
    cfg = load_config("configs/config.yaml")

    data_dir = Path(get_config_value(cfg, "data.processed_dir", "data/processed"))
    test_csv = data_dir / "test.csv"
    if not test_csv.exists():
        logger.error("Test data split not found at %s. Please run make train first.", test_csv)
        return 1

    model_path = Path(get_config_value(cfg, "artifacts.model_path", "models/model.joblib"))
    metrics_path = Path(get_config_value(cfg, "artifacts.metrics_path", "models/metrics.json"))
    figures_dir = Path(get_config_value(cfg, "artifacts.figures_dir", "reports/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading test data from %s", test_csv)
    test_df = pd.read_csv(test_csv)
    target_col = get_config_value(cfg, "data.target_column", "SeriousDlqin2yrs")

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    threshold = 0.20
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            threshold = float(metrics.get("threshold", 0.20))
    logger.info("Using decision threshold: %.3f", threshold)

    logger.info("Generating predictions on test set (%d samples)...", len(X_test))
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # 1. ROC Curve
    generate_roc_curve(y_test, y_proba, figures_dir / "roc_curve.png", threshold=threshold)

    # 2. Precision-Recall Curve
    generate_pr_curve(y_test, y_proba, figures_dir / "pr_curve.png", threshold=threshold)

    # 3. Confusion Matrix
    generate_confusion_matrix(y_test, y_pred, figures_dir / "confusion_matrix.png", threshold=threshold)

    # 4. Calibration Curve
    generate_calibration_curve(y_test, y_proba, figures_dir / "calibration_curve.png")

    # 5. SHAP Global Feature Importance
    try:
        generate_shap_summary(model, X_test, figures_dir / "shap_summary.png")
    except Exception as exc:
        logger.warning("Failed to generate SHAP plot: %s", exc)

    logger.info("All figures generated successfully in %s", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
