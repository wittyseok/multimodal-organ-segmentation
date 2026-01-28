"""
Late fusion strategy for multi-modal medical imaging.

Fuses features at the decoder/output level.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class LateFusion(nn.Module):
    """
    Late fusion: combine features after separate processing.

    Each modality is processed independently, then features are fused
    at a later stage (e.g., after encoding).
    """

    def __init__(
        self,
        in_channels: int,
        num_modalities: int = 2,
        fusion_method: str = "concat",  # concat, add, max, mean
        out_channels: Optional[int] = None,
    ):
        """
        Initialize late fusion.

        Args:
            in_channels: Channels per modality feature map
            num_modalities: Number of modalities
            fusion_method: How to combine features
            out_channels: Output channels (for concat method)
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_modalities = num_modalities
        self.fusion_method = fusion_method

        if fusion_method == "concat":
            self.out_channels = out_channels or in_channels
            self.proj = nn.Sequential(
                nn.Conv3d(in_channels * num_modalities, self.out_channels, kernel_size=1),
                nn.InstanceNorm3d(self.out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.out_channels = in_channels
            self.proj = nn.Identity()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from multiple modalities.

        Args:
            features: List of feature tensors [B, C, H, W, D] per modality

        Returns:
            Fused features [B, out_channels, H, W, D]
        """
        if self.fusion_method == "concat":
            fused = torch.cat(features, dim=1)
        elif self.fusion_method == "add":
            fused = sum(features)
        elif self.fusion_method == "max":
            stacked = torch.stack(features, dim=0)
            fused = stacked.max(dim=0)[0]
        elif self.fusion_method == "mean":
            stacked = torch.stack(features, dim=0)
            fused = stacked.mean(dim=0)
        else:
            fused = torch.cat(features, dim=1)

        return self.proj(fused)


class HierarchicalLateFusion(nn.Module):
    """
    Hierarchical late fusion at multiple decoder levels.

    Fuses features at each level of the decoder with level-specific
    fusion modules.
    """

    def __init__(
        self,
        feature_channels: List[int],
        num_modalities: int = 2,
        fusion_method: str = "concat",
    ):
        """
        Initialize hierarchical late fusion.

        Args:
            feature_channels: Channel sizes at each level
            num_modalities: Number of modalities
            fusion_method: Fusion method per level
        """
        super().__init__()

        self.fusion_layers = nn.ModuleList()
        for channels in feature_channels:
            self.fusion_layers.append(
                LateFusion(
                    in_channels=channels,
                    num_modalities=num_modalities,
                    fusion_method=fusion_method,
                )
            )

    def forward(
        self,
        multi_modal_features: List[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
        Fuse features at each level.

        Args:
            multi_modal_features: List (per modality) of lists (per level) of tensors

        Returns:
            List of fused features per level
        """
        num_levels = len(multi_modal_features[0])
        fused_features = []

        for level in range(num_levels):
            level_features = [modal[level] for modal in multi_modal_features]
            fused = self.fusion_layers[level](level_features)
            fused_features.append(fused)

        return fused_features
