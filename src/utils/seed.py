"""
Seed utilities for reproducibility.
"""

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode for CUDA
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # For PyTorch >= 1.8
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass

    except ImportError:
        pass


def get_seed() -> Optional[int]:
    """
    Get current random seed (if set).

    Returns:
        Current seed or None if not set
    """
    return os.environ.get("PYTHONHASHSEED")
