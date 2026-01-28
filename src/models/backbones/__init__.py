"""
Backbone architectures for medical image segmentation.
"""

from src.models.backbones.swin_unetr import SwinUNETR
from src.models.backbones.unet import UNet3D
from src.models.backbones.dual_encoder import DualEncoder

__all__ = [
    "SwinUNETR",
    "UNet3D",
    "DualEncoder",
]
