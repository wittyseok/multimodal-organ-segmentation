"""
Logging utilities for the segmentation pipeline.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Global logger registry
_loggers = {}


def setup_logger(
    name: str = "main",
    log_file: Optional[Union[str, Path]] = None,
    level: str = "INFO",
    console: bool = True,
    file_mode: str = "a",
) -> logging.Logger:
    """
    Setup and configure a logger.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to output to console
        file_mode: File mode ('a' for append, 'w' for overwrite)

    Returns:
        Configured logger instance
    """
    # Check if logger already exists
    if name in _loggers:
        return _loggers[name]

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers

    # Format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-5s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode=file_mode, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    # Register logger
    _loggers[name] = logger

    return logger


def get_logger(name: str = "main") -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class LoggerAdapter:
    """
    Adapter class providing additional logging methods.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_config(self, config: dict, prefix: str = "") -> None:
        """Log configuration dictionary."""
        for key, value in config.items():
            if isinstance(value, dict):
                self.log_config(value, prefix=f"{prefix}{key}.")
            else:
                self.logger.info(f"{prefix}{key}: {value}")

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log metrics dictionary."""
        step_str = f"[Step {step}] " if step is not None else ""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"{step_str}{metrics_str}")

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """Log epoch summary."""
        msg = f"Epoch [{epoch}/{total_epochs}] | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
        if metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            msg += f" | {metrics_str}"
        self.logger.info(msg)

    def __getattr__(self, name):
        """Delegate to underlying logger."""
        return getattr(self.logger, name)
