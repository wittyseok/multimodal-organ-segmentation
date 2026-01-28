"""
Segmentation head for multi-organ segmentation.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    """
    Segmentation head for producing class predictions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        dropout: float = 0.0,
        activation: Optional[str] = None,
    ):
        """
        Initialize segmentation head.

        Args:
            in_channels: Input feature channels
            out_channels: Number of output classes
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            activation: Output activation ('softmax', 'sigmoid', None)
        """
        super().__init__()

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)

        if activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W, D]

        Returns:
            Segmentation logits/probabilities [B, num_classes, H, W, D]
        """
        x = self.dropout(x)
        x = self.conv(x)
        x = self.activation(x)
        return x


class DeepSupervisionHead(nn.Module):
    """
    Deep supervision head for multi-scale predictions.

    Produces predictions at multiple scales for deep supervision during training.
    """

    def __init__(
        self,
        in_channels_list: list,
        out_channels: int,
        dropout: float = 0.0,
    ):
        """
        Initialize deep supervision head.

        Args:
            in_channels_list: List of input channels at each scale
            out_channels: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()

        self.heads = nn.ModuleList()
        for in_ch in in_channels_list:
            self.heads.append(
                SegmentationHead(in_ch, out_channels, dropout=dropout)
            )

    def forward(
        self,
        features: list,
        target_size: Optional[tuple] = None,
    ) -> list:
        """
        Args:
            features: List of feature maps at different scales
            target_size: Target output size (H, W, D)

        Returns:
            List of predictions at each scale
        """
        outputs = []
        for feat, head in zip(features, self.heads):
            out = head(feat)

            if target_size is not None and out.shape[2:] != target_size:
                out = nn.functional.interpolate(
                    out, size=target_size, mode="trilinear", align_corners=True
                )

            outputs.append(out)

        return outputs
