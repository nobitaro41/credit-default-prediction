"""Logging utilities for the credit default prediction project."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, level: Optional[int] = None, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Create and configure a logger with console and optional file output.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (e.g., logging.INFO). Defaults to INFO.
        log_file: Optional path to log file. If provided, logs are also written to file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level or logging.INFO)

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Console handler with formatted output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level or logging.INFO)
    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level or logging.INFO)
        file_format = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_filename: str = "app.log",
) -> None:
    """
    Configure root logger for the entire application.

    Args:
        level: Global logging level.
        log_dir: Directory for log files. If None, only console logging is used.
        log_filename: Name of the log file when log_dir is provided.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_filename
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    # Suppress overly verbose third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
