"""Run the credit-default machine-learning pipeline end-to-end.

This is the entry point for ``make train``. It executes three stages:

    prepare  - load -> clean -> stratified split, persisting each split to
               ``data/processed`` as a CSV (features + target column).
    train + evaluate - fit every model in the registry (logged to MLflow), rank
               them by PR-AUC on the validation set, then score the best one
               once on the held-out test set and persist the final model
               (``models/model.joblib``) plus its metrics (``models/metrics.json``).
    predict  - (optional, ``--predict-test``) score the unlabeled ``cs-test.csv``.

Usage:
    uv run scripts/run_pipeline.py
    uv run scripts/run_pipeline.py --model-names logistic_regression random_forest
    uv run scripts/run_pipeline.py --seed 7 --test-size 0.2 --val-size 0.1 --no-mlflow
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

# Running a script puts only ``scripts/`` on sys.path; make the repo root
# importable so ``src.*`` resolves no matter where ``make train`` is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.clean import clean_data
from src.data.load import load_training_data
from src.data.split import DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, split_dataset
from src.models.evaluate import DEFAULT_FN_COST, DEFAULT_FP_COST
from src.models.predict import predict_batch
from src.models.train import DEFAULT_SEED, MODEL_REGISTRY, run_training
from src.utils.config import get_config_value, load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

TARGET_COL = "SeriousDlqin2yrs"
MLFLOW_EXPERIMENT = "credit-default-prediction"


def save_split(df: pd.DataFrame, name: str, processed_dir: Path | str) -> Path:
    """Write one split (features + target) to ``processed_dir`` as a CSV.

    Args:
        df: DataFrame already containing the target column.
        name: Split name, e.g. ``train``/``val``/``test``.
        processed_dir: Directory to write into.

    Returns:
        Path to the written CSV.
    """
    path = Path(processed_dir) / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %s split (%d rows) to %s.", name, len(df), path)
    return path


def prepare(
    data_dir: Path | str,
    processed_dir: Path | str,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, ...]:
    """Load, clean and split the raw data; persist the splits to disk.

    Returns the in-memory splits as ``(X_train, y_train, X_val, y_val, X_test,
    y_test)`` so the train step can use them without re-reading files.

    Args:
        data_dir: Directory holding ``cs-training.csv``.
        processed_dir: Where the prepared splits are written.
        test_size: Fraction of rows reserved for the test set.
        val_size: Fraction of rows reserved for the validation set.
        seed: Seed for a reproducible split.
    """
    logger.info("Step 1/3 - prepare: load -> clean -> split")
    raw = load_training_data(data_dir)
    clean = clean_data(raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        clean, test_size=test_size, val_size=val_size, random_state=seed
    )

    # Persist each split with its target column reattached, for downstream reuse.
    for name, X, y in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        band = X.copy()
        band[TARGET_COL] = y.values
        save_split(band, name, processed_dir)

    logger.info(
        "Prepared splits: train=%d, val=%d, test=%d.",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_names: list[str],
    seed: int,
    fn_cost: float,
    fp_cost: float,
    mlflow: bool,
    model_path: Path | str,
    metrics_path: Path | str,
) -> tuple[str, dict]:
    """Train every model, pick the best and persist it together with metrics.

    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Data splits.
        model_names: Models to train (subset of ``MODEL_REGISTRY``).
        seed: Seed shared by every model.
        fn_cost / fp_cost: Business costs used for threshold selection.
        mlflow: Whether to log runs to MLflow.
        model_path / metrics_path: Where to persist the best model and metrics.

    Returns:
        ``(best_model_name, test_metrics)``.
    """
    logger.info("Step 2/3 - train + evaluate (models: %s)", model_names)
    if mlflow:
        _configure_mlflow()
    best_name, metrics = run_training(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_names=model_names,
        random_state=seed,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        mlflow_tracking=mlflow,
        model_path=model_path,
        metrics_path=metrics_path,
    )
    logger.info(
        "Best model '%s' - test ROC-AUC %.4f | PR-AUC %.4f | recall %.4f | threshold %.3f",
        best_name, metrics["roc_auc"], metrics["pr_auc"], metrics["recall"],
        metrics["threshold"],
    )
    return best_name, metrics


def predict_test_data(
    data_dir: Path | str,
    output_path: Path | str,
    model_path: Path | str,
    metrics_path: Path | str,
) -> pd.DataFrame:
    """Score the unlabeled ``cs-test.csv`` and persist the predictions.

    Args:
        data_dir: Directory holding ``cs-test.csv``.
        output_path: Where to write the scored CSV.
        model_path / metrics_path: Saved model and its threshold.

    Returns:
        The scored DataFrame.
    """
    logger.info("Step 3/3 - score cs-test.csv")
    input_path = Path(data_dir) / "cs-test.csv"
    return predict_batch(
        input_path, output_path, model_path=model_path, metrics_path=metrics_path
    )


def _configure_mlflow() -> None:
    """Point MLflow at the local tracking store and the project experiment."""
    import mlflow

    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="Run the full credit-default pipeline.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML file.")
    parser.add_argument("--data-dir", default=None, help="Raw data directory (default: from config or data/raw).")
    parser.add_argument("--processed-dir", default=None, help="Where to write prepared splits.")
    parser.add_argument("--model-names", nargs="*", default=None, help="Models to train (default: all in registry).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: from config or 42).")
    parser.add_argument("--test-size", type=float, default=None, help="Test set fraction.")
    parser.add_argument("--val-size", type=float, default=None, help="Validation set fraction.")
    parser.add_argument("--fn-cost", type=float, default=None, help="Cost of a missed defaulter.")
    parser.add_argument("--fp-cost", type=float, default=None, help="Cost of a wrongly rejected customer.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    parser.add_argument("--predict-test", action="store_true", help="Also score cs-test.csv.")
    parser.add_argument("--model-path", default=None, help="Where to save the best model.")
    parser.add_argument("--metrics-path", default=None, help="Where to save the metrics JSON.")
    parser.add_argument("--stage", choices=["all", "prepare", "train", "predict"], default="all", help="Pipeline stage to run.")
    parser.add_argument("--predictions-path", default=None, help="Where to write test predictions.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Run the whole pipeline; returns a process exit code."""
    args = parse_args(argv)
    warnings.filterwarnings("ignore", category=FutureWarning)  # quiet noisy warnings

    # Load configuration from YAML with graceful CLI precedence
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(get_config_value(cfg, "seed", DEFAULT_SEED))
    set_seed(seed)

    data_dir = args.data_dir or get_config_value(cfg, "data.raw_dir", "data/raw")
    processed_dir = args.processed_dir or get_config_value(cfg, "data.processed_dir", "data/processed")
    test_size = args.test_size if args.test_size is not None else float(get_config_value(cfg, "data.split.test_size", DEFAULT_TEST_SIZE))
    val_size = args.val_size if args.val_size is not None else float(get_config_value(cfg, "data.split.val_size", DEFAULT_VAL_SIZE))
    fn_cost = args.fn_cost if args.fn_cost is not None else float(get_config_value(cfg, "evaluation.fn_cost", DEFAULT_FN_COST))
    fp_cost = args.fp_cost if args.fp_cost is not None else float(get_config_value(cfg, "evaluation.fp_cost", DEFAULT_FP_COST))
    model_path = args.model_path or get_config_value(cfg, "artifacts.model_path", "models/model.joblib")
    metrics_path = args.metrics_path or get_config_value(cfg, "artifacts.metrics_path", "models/metrics.json")
    predictions_path = args.predictions_path or get_config_value(cfg, "artifacts.predictions_path", "data/predictions/cs-test-predictions.csv")

    if args.model_names:
        unknown = set(args.model_names) - set(MODEL_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown model(s): {sorted(unknown)}")

    if args.stage == "prepare":
        prepare(data_dir, processed_dir, test_size, val_size, seed)
        logger.info("Stage 'prepare' completed successfully.")
        return 0

    if args.stage == "train":
        train_csv = Path(processed_dir) / "train.csv"
        val_csv = Path(processed_dir) / "val.csv"
        test_csv = Path(processed_dir) / "test.csv"
        if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
            raise FileNotFoundError(f"Prepared splits missing in {processed_dir}. Run prepare first.")

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)

        X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
        X_val, y_val = val_df.drop(columns=[TARGET_COL]), val_df[TARGET_COL]
        X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

        _, metrics = train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test,
            model_names=args.model_names,
            seed=seed,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            mlflow=not args.no_mlflow,
            model_path=model_path,
            metrics_path=metrics_path,
        )
        logger.info("Stage 'train' completed. Best model: %s", metrics.get("best_model"))
        return 0

    if args.stage == "predict":
        predict_test_data(data_dir, predictions_path, model_path, metrics_path)
        logger.info("Stage 'predict' completed successfully.")
        return 0

    # Default: args.stage == "all"
    X_train, y_train, X_val, y_val, X_test, y_test = prepare(
        data_dir, processed_dir, test_size, val_size, seed
    )
    _, metrics = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_names=args.model_names,
        seed=seed,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        mlflow=not args.no_mlflow,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    if args.predict_test:
        predict_test_data(data_dir, predictions_path, model_path, metrics_path)

    logger.info("Pipeline finished. Model: %s", metrics.get("best_model"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
