"""
Dual Encoder architecture for multi-modal medical imaging.

Separate encoders for each modality with fusion at various levels.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.backbones.unet import ConvBlock3D, DownBlock3D, UpBlock3D


class DualEncoder(nn.Module):
    """
    Dual encoder architecture for multi-modal segmentation.

    Uses separate encoders for each modality (e.g., CT and PET),
    then fuses features in the decoder.
    """

    def __init__(
        self,
        in_channels_per_modality: int = 1,
        num_modalities: int = 2,
        out_channels: int = 8,
        features: List[int] = [32, 64, 128, 256, 512],
        norm: str = "instance",
        fusion_type: str = "concat",  # concat, add, attention
        dropout: float = 0.0,
        shared_decoder: bool = True,
        **kwargs,
    ):
        """
        Initialize DualEncoder.

        Args:
            in_channels_per_modality: Input channels per modality
            num_modalities: Number of input modalities
            out_channels: Number of output classes
            features: Feature sizes at each level
            norm: Normalization type
            fusion_type: How to fuse multi-modal features
            dropout: Dropout rate
            shared_decoder: Use single decoder (vs separate decoders)
        """
        super().__init__()

        self.in_channels_per_modality = in_channels_per_modality
        self.num_modalities = num_modalities
        self.out_channels = out_channels
        self.features = features
        self.fusion_type = fusion_type
        self.shared_decoder = shared_decoder

        # Create separate encoders for each modality
        self.encoders = nn.ModuleList()
        for _ in range(num_modalities):
            encoder = self._build_encoder(in_channels_per_modality, features, norm)
            self.encoders.append(encoder)

        # Fusion layers
        if fusion_type == "attention":
            self.fusion_layers = nn.ModuleList()
            for feat in features:
                self.fusion_layers.append(
                    CrossModalAttention(feat, num_modalities)
                )
        elif fusion_type == "concat":
            # Projection after concatenation
            self.fusion_proj = nn.ModuleList()
            for feat in features:
                self.fusion_proj.append(
                    nn.Conv3d(feat * num_modalities, feat, kernel_size=1)
                )

        # Decoder
        decoder_in_features = features if fusion_type != "concat" else features
        self.decoder = self._build_decoder(decoder_in_features, norm)

        # Output
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def _build_encoder(
        self,
        in_channels: int,
        features: List[int],
        norm: str,
    ) -> nn.ModuleDict:
        """Build single encoder."""
        encoder = nn.ModuleDict()

        # Initial conv
        encoder["init_conv"] = ConvBlock3D(in_channels, features[0], norm=norm)

        # Encoder blocks
        encoder["blocks"] = nn.ModuleList()
        for i in range(len(features) - 1):
            encoder["blocks"].append(DownBlock3D(features[i], features[i + 1], norm=norm))

        return encoder

    def _build_decoder(self, features: List[int], norm: str) -> nn.ModuleList:
        """Build decoder."""
        decoder = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            decoder.append(UpBlock3D(features[i], features[i - 1], norm=norm))
        return decoder

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, num_modalities, H, W, D]
            return_features: Whether to return intermediate features

        Returns:
            Segmentation logits, optionally with features
        """
        batch_size = x.shape[0]

        # Process each modality through its encoder
        all_encoder_features = []
        for i, encoder in enumerate(self.encoders):
            # Extract single modality: [B, 1, H, W, D]
            modality_input = x[:, i : i + 1, ...]

            # Initial conv
            feat = encoder["init_conv"](modality_input)
            modality_features = [feat]

            # Encoder blocks
            for block in encoder["blocks"]:
                feat, _ = block(feat)
                modality_features.append(feat)

            all_encoder_features.append(modality_features)

        # Fuse features from all modalities at each level
        fused_features = self._fuse_features(all_encoder_features)

        # Decoder path
        x = fused_features[-1]  # Start from bottleneck
        skip_features = fused_features[:-1]

        for decoder, skip in zip(self.decoder, reversed(skip_features)):
            x = decoder(x, skip)

        # Output
        x = self.dropout(x)
        x = self.out_conv(x)

        if return_features:
            return x, {
                "encoder_features": all_encoder_features,
                "fused_features": fused_features,
            }
        return x

    def _fuse_features(
        self,
        all_features: List[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """Fuse features from multiple modalities at each level."""
        num_levels = len(all_features[0])
        fused = []

        for level in range(num_levels):
            # Gather features from all modalities at this level
            level_features = [feat[level] for feat in all_features]

            if self.fusion_type == "concat":
                # Concatenate and project
                concat = torch.cat(level_features, dim=1)
                fused_feat = self.fusion_proj[level](concat)

            elif self.fusion_type == "add":
                # Element-wise addition
                fused_feat = sum(level_features)

            elif self.fusion_type == "attention":
                # Cross-modal attention
                stacked = torch.stack(level_features, dim=1)  # [B, M, C, H, W, D]
                fused_feat = self.fusion_layers[level](stacked)

            else:
                # Default: mean
                fused_feat = torch.stack(level_features).mean(dim=0)

            fused.append(fused_feat)

        return fused

    @property
    def encoder_channels(self) -> List[int]:
        """Get encoder output channels at each stage."""
        return self.features


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for feature fusion.

    Learns to weight features from different modalities.
    """

    def __init__(
        self,
        channels: int,
        num_modalities: int,
        reduction: int = 4,
    ):
        super().__init__()

        self.channels = channels
        self.num_modalities = num_modalities

        # Channel attention for each modality
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels * num_modalities, channels * num_modalities // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels * num_modalities // reduction, num_modalities),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Stacked features [B, M, C, H, W, D] where M is num_modalities

        Returns:
            Fused features [B, C, H, W, D]
        """
        B, M, C, H, W, D = x.shape

        # Compute attention weights
        # Reshape to [B, M*C, H, W, D] for pooling
        x_flat = x.view(B, M * C, H, W, D)
        weights = self.attention(x_flat)  # [B, M]

        # Apply weights and sum
        weights = weights.view(B, M, 1, 1, 1, 1)
        fused = (x * weights).sum(dim=1)  # [B, C, H, W, D]

        return fused


def build_dual_encoder(config: Dict[str, Any]) -> DualEncoder:
    """
    Build DualEncoder from configuration.

    Args:
        config: Model configuration

    Returns:
        DualEncoder model
    """
    backbone_config = config.get("model", {}).get("backbone", {})
    fusion_config = config.get("model", {}).get("fusion", {})

    num_modalities = len(config["data"]["modalities"])

    return DualEncoder(
        in_channels_per_modality=1,
        num_modalities=num_modalities,
        out_channels=config["model"]["out_channels"],
        features=backbone_config.get("features", [32, 64, 128, 256, 512]),
        norm=backbone_config.get("norm", "instance"),
        fusion_type=fusion_config.get("type", "concat"),
        dropout=config["model"].get("head", {}).get("dropout", 0.0),
    )
