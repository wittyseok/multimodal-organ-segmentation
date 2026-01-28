"""
Model module containing backbone architectures, fusion strategies, and task heads.
"""

from src.models.build import build_model, get_model

__all__ = [
    "build_model",
    "get_model",
]
