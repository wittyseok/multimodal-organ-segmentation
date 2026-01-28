"""
Multi-modal fusion modules.
"""

from src.models.fusion.early_fusion import EarlyFusion
from src.models.fusion.late_fusion import LateFusion
from src.models.fusion.attention_fusion import AttentionFusion, CrossAttentionFusion

__all__ = [
    "EarlyFusion",
    "LateFusion",
    "AttentionFusion",
    "CrossAttentionFusion",
]
