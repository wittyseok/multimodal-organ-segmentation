"""
Training module with Trainer class, loss functions, and evaluation metrics.
"""

from src.trainer.trainer import Trainer
from src.trainer.losses import get_loss
from src.trainer.metrics import get_metrics

__all__ = [
    "Trainer",
    "get_loss",
    "get_metrics",
]
