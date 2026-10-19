"""Data loading utilities for the credit default prediction project."""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Expected columns in the training data (including target)
EXPECTED_COLUMNS = [
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Expected dtypes for validation
EXPECTED_DTYPES = {
    "SeriousDlqin2yrs": "int64",
    "RevolvingUtilizationOfUnsecuredLines": "float64",
    "age": "int64",
    "NumberOfTime30-59DaysPastDueNotWorse": "int64",
    "DebtRatio": "float64",
    "MonthlyIncome": "float64",
    "NumberOfOpenCreditLinesAndLoans": "int64",
    "NumberOfTimes90DaysLate": "int64",
    "NumberRealEstateLoansOrLines": "int64",
    "NumberOfTime60-89DaysPastDueNotWorse": "int64",
    "NumberOfDependents": "float64",  # Has NaN values
}


def load_data(
    file_path: Path | str,
    index_col: int = 0,
    expected_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load CSV data and perform basic validation.

    Args:
        file_path: Path to the CSV file.
        index_col: Column to use as row index (default 0 for Unnamed: 0).
        expected_columns: Optional list of expected column names for validation.

    Returns:
        DataFrame with loaded data.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If column validation fails.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, index_col=index_col)

    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Validate columns
    if expected_columns is None:
        expected_columns = EXPECTED_COLUMNS

    missing_cols = set(expected_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_columns)

    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    if extra_cols:
        logger.warning(f"Unexpected columns found (will be ignored): {extra_cols}")

    return df


def load_training_data(
    data_dir: Path | str = "data/raw",
    filename: str = "cs-training.csv",
) -> pd.DataFrame:
    """
    Load the training dataset (cs-training.csv).

    Args:
        data_dir: Directory containing raw data files.
        filename: Training data filename.

    Returns:
        DataFrame with training data including target column.
    """
    file_path = Path(data_dir) / filename
    return load_data(file_path, expected_columns=EXPECTED_COLUMNS)


def load_test_data(
    data_dir: Path | str = "data/raw",
    filename: str = "cs-test.csv",
) -> pd.DataFrame:
    """
    Load the test dataset (cs-test.csv) without target column.

    The test file may contain a SeriousDlqin2yrs column with all NaN values.
    This function drops the target column if present.

    Args:
        data_dir: Directory containing raw data files.
        filename: Test data filename.

    Returns:
        DataFrame with test data (no target column).
    """
    file_path = Path(data_dir) / filename
    df = load_data(file_path, expected_columns=EXPECTED_COLUMNS)
    # Drop target column if it exists (test data has all NaN)
    if "SeriousDlqin2yrs" in df.columns:
        df = df.drop(columns=["SeriousDlqin2yrs"])
    return df


def validate_schema(df: pd.DataFrame, expected_dtypes: Optional[dict[str, str]] = None) -> None:
    """
    Validate DataFrame schema against expected columns and dtypes.

    Args:
        df: DataFrame to validate.
        expected_dtypes: Optional dict mapping column names to expected dtypes.

    Raises:
        ValueError: If schema validation fails.
    """
    if expected_dtypes is None:
        expected_dtypes = EXPECTED_DTYPES

    logger.info("Validating data schema...")

    # Check columns
    for col in expected_dtypes:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    # Check dtypes (allow some flexibility for nullable columns)
    for col, expected_dtype in expected_dtypes.items():
        actual_dtype = str(df[col].dtype)
        # MonthlyIncome and NumberOfDependents can be float due to NaN
        if col in ("MonthlyIncome", "NumberOfDependents"):
            if not (actual_dtype.startswith("float") or actual_dtype.startswith("int")):
                raise ValueError(
                    f"Column {col}: expected numeric type, got {actual_dtype}"
                )
        elif actual_dtype != expected_dtype:
            logger.warning(
                f"Column {col}: expected dtype {expected_dtype}, got {actual_dtype}"
            )

    # Check target values are binary
    if "SeriousDlqin2yrs" in df.columns:
        target_values = df["SeriousDlqin2yrs"].unique()
        if not set(target_values).issubset({0, 1}):
            raise ValueError(
                f"Target column has unexpected values: {target_values}. Expected only 0 and 1."
            )
        logger.info(f"Target distribution: {df['SeriousDlqin2yrs'].value_counts().to_dict()}")

    # Log basic stats for numeric columns
    logger.info("Basic statistics:")
    logger.info(f"\n{df.describe().to_string()}")

    logger.info("Schema validation passed.")


def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str = "SeriousDlqin2yrs",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y).

    Args:
        df: Input DataFrame containing features and target.
        target_col: Name of the target column.

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y
