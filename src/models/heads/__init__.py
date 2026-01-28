"""
Task-specific heads for segmentation.
"""

from src.models.heads.segmentation import SegmentationHead
from src.models.heads.detection import DetectionHead

__all__ = [
    "SegmentationHead",
    "DetectionHead",
]
