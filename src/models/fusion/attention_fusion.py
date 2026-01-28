"""
Attention-based fusion for multi-modal medical imaging.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for multi-modal features.

    Uses channel attention to weight features from different modalities.
    """

    def __init__(
        self,
        in_channels: int,
        num_modalities: int = 2,
        reduction: int = 4,
    ):
        """
        Initialize attention fusion.

        Args:
            in_channels: Channels per modality
            num_modalities: Number of modalities
            reduction: Channel reduction ratio for attention
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_modalities = num_modalities

        # Squeeze-and-excitation style attention
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * num_modalities, in_channels * num_modalities // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * num_modalities // reduction, num_modalities),
            nn.Softmax(dim=1),
        )

        self.out_channels = in_channels

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse features using attention weights.

        Args:
            features: List of feature tensors [B, C, H, W, D] per modality

        Returns:
            Fused features [B, C, H, W, D]
        """
        B = features[0].shape[0]

        # Stack features: [B, M, C, H, W, D]
        stacked = torch.stack(features, dim=1)

        # Global pooling: [B, M*C]
        pooled = torch.cat([self.global_pool(f).flatten(1) for f in features], dim=1)

        # Compute attention weights: [B, M]
        weights = self.fc(pooled)

        # Apply weights
        weights = weights.view(B, self.num_modalities, 1, 1, 1, 1)
        fused = (stacked * weights).sum(dim=1)

        return fused


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between modalities.

    Uses one modality as query and another as key/value for attention.
    Particularly useful for CT (structure) + PET (function) fusion.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        """
        Initialize cross-attention fusion.

        Args:
            in_channels: Number of input channels
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # Output projection
        self.out_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.InstanceNorm3d(in_channels)

        self.out_channels = in_channels

    def forward(
        self,
        query_features: torch.Tensor,
        key_value_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attention from query to key/value features.

        Args:
            query_features: Query modality features [B, C, H, W, D]
            key_value_features: Key/value modality features [B, C, H, W, D]

        Returns:
            Attended features [B, C, H, W, D]
        """
        B, C, H, W, D = query_features.shape

        # Project to Q, K, V
        Q = self.q_proj(query_features)
        K = self.k_proj(key_value_features)
        V = self.v_proj(key_value_features)

        # Reshape for multi-head attention
        # [B, num_heads, head_dim, H*W*D]
        Q = Q.view(B, self.num_heads, self.head_dim, -1)
        K = K.view(B, self.num_heads, self.head_dim, -1)
        V = V.view(B, self.num_heads, self.head_dim, -1)

        # Attention scores
        scale = self.head_dim ** -0.5
        attn = torch.einsum("bhdn,bhdm->bhnm", Q, K) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.einsum("bhnm,bhdm->bhdn", attn, V)

        # Reshape back
        out = out.view(B, C, H, W, D)
        out = self.out_proj(out)

        # Residual connection with normalization
        out = self.norm(query_features + out)

        return out


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between two modalities.

    Each modality attends to the other, creating mutual information exchange.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.cross_attn_1to2 = CrossAttentionFusion(in_channels, num_heads, dropout)
        self.cross_attn_2to1 = CrossAttentionFusion(in_channels, num_heads, dropout)

        # Fusion of bidirectional attention
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = in_channels

    def forward(
        self,
        features_1: torch.Tensor,
        features_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bidirectional cross-attention.

        Args:
            features_1: First modality features [B, C, H, W, D]
            features_2: Second modality features [B, C, H, W, D]

        Returns:
            Fused features [B, C, H, W, D]
        """
        # Cross-attention in both directions
        attn_1to2 = self.cross_attn_1to2(features_1, features_2)
        attn_2to1 = self.cross_attn_2to1(features_2, features_1)

        # Fuse bidirectional attention
        fused = self.fusion(torch.cat([attn_1to2, attn_2to1], dim=1))

        return fused


class SUVGuidedAttention(nn.Module):
    """
    SUV-guided attention for PET/CT fusion.

    Uses high SUV regions from PET to guide attention in CT features.
    Particularly useful for tumor segmentation.
    """

    def __init__(
        self,
        in_channels: int,
        suv_threshold: float = 2.5,
        learnable_threshold: bool = False,
    ):
        """
        Initialize SUV-guided attention.

        Args:
            in_channels: Number of input channels
            suv_threshold: SUV threshold for high uptake regions
            learnable_threshold: Make threshold learnable
        """
        super().__init__()

        self.in_channels = in_channels

        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(suv_threshold))
        else:
            self.register_buffer("threshold", torch.tensor(suv_threshold))

        # Spatial attention from SUV
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Feature modulation
        self.feature_mod = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm3d(in_channels),
        )

        self.out_channels = in_channels

    def forward(
        self,
        ct_features: torch.Tensor,
        pet_suv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply SUV-guided attention to CT features.

        Args:
            ct_features: CT encoder features [B, C, H, W, D]
            pet_suv: PET SUV image [B, 1, H, W, D]

        Returns:
            Attended CT features [B, C, H, W, D]
        """
        # Resize PET to match CT features if needed
        if pet_suv.shape[2:] != ct_features.shape[2:]:
            pet_suv = F.interpolate(
                pet_suv, size=ct_features.shape[2:], mode="trilinear", align_corners=True
            )

        # Create soft attention mask based on SUV
        suv_mask = torch.sigmoid((pet_suv - self.threshold) * 2)
        spatial_attn = self.spatial_attn(suv_mask)

        # Apply attention to CT features
        attended = ct_features * (1 + spatial_attn)
        attended = self.feature_mod(attended)

        return attended
