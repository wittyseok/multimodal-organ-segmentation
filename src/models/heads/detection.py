"""
Detection head for tumor detection.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    Detection head for tumor/lesion detection.

    Outputs bounding box predictions and classification scores.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,  # background + tumor
        num_anchors: int = 3,
        dropout: float = 0.0,
    ):
        """
        Initialize detection head.

        Args:
            in_channels: Input feature channels
            num_classes: Number of classes (including background)
            num_anchors: Number of anchor boxes per location
            dropout: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Shared conv layers
        self.shared = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

        # Classification branch
        self.cls_head = nn.Conv3d(in_channels, num_anchors * num_classes, kernel_size=1)

        # Regression branch (6 values: center_x, center_y, center_z, w, h, d)
        self.reg_head = nn.Conv3d(in_channels, num_anchors * 6, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [B, C, H, W, D]

        Returns:
            Tuple of:
                - cls_scores: Classification scores [B, num_anchors*num_classes, H, W, D]
                - bbox_preds: Bounding box predictions [B, num_anchors*6, H, W, D]
        """
        x = self.shared(x)

        cls_scores = self.cls_head(x)
        bbox_preds = self.reg_head(x)

        return cls_scores, bbox_preds


class CenterNetHead(nn.Module):
    """
    CenterNet-style detection head.

    Predicts center heatmap and offset for anchor-free detection.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,  # tumor only
        dropout: float = 0.0,
    ):
        """
        Initialize CenterNet head.

        Args:
            in_channels: Input feature channels
            num_classes: Number of object classes
            dropout: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes

        # Heatmap prediction (center locations)
        self.heatmap = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

        # Offset prediction (sub-voxel offset)
        self.offset = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 3, kernel_size=1),  # x, y, z offsets
        )

        # Size prediction
        self.size = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 3, kernel_size=1),  # w, h, d
        )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input features [B, C, H, W, D]

        Returns:
            Dictionary with heatmap, offset, and size predictions
        """
        return {
            "heatmap": self.heatmap(x),
            "offset": self.offset(x),
            "size": self.size(x),
        }
