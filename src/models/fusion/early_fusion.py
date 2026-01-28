"""
Early fusion strategy for multi-modal medical imaging.

Concatenates input modalities at the input level.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class EarlyFusion(nn.Module):
    """
    Early fusion: concatenate modalities at input level.

    The simplest fusion strategy - just stack input channels.
    """

    def __init__(
        self,
        num_modalities: int = 2,
        in_channels_per_modality: int = 1,
        projection: bool = False,
        out_channels: Optional[int] = None,
    ):
        """
        Initialize early fusion.

        Args:
            num_modalities: Number of input modalities
            in_channels_per_modality: Channels per modality
            projection: Apply projection after concatenation
            out_channels: Output channels if projection is used
        """
        super().__init__()

        self.num_modalities = num_modalities
        self.in_channels = num_modalities * in_channels_per_modality

        if projection:
            out_channels = out_channels or in_channels_per_modality
            self.proj = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.out_channels = out_channels
        else:
            self.proj = nn.Identity()
            self.out_channels = self.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, M*C, H, W, D] or list of [B, C, H, W, D]

        Returns:
            Fused tensor [B, out_channels, H, W, D]
        """
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, dim=1)

        return self.proj(x)
