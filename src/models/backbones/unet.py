"""
3D UNet backbone for medical image segmentation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """3D Convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        norm: str = "instance",
        activation: str = "relu",
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)

        # Normalization
        if norm == "batch":
            self.norm1 = nn.BatchNorm3d(out_channels)
            self.norm2 = nn.BatchNorm3d(out_channels)
        elif norm == "instance":
            self.norm1 = nn.InstanceNorm3d(out_channels)
            self.norm2 = nn.InstanceNorm3d(out_channels)
        elif norm == "group":
            self.norm1 = nn.GroupNorm(8, out_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Activation
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x


class DownBlock3D(nn.Module):
    """Downsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "instance",
    ):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock3D(in_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_pool = self.pool(x)
        x_conv = self.conv(x_pool)
        return x_conv, x_pool


class UpBlock3D(nn.Module):
    """Upsampling block with skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "instance",
        mode: str = "transpose",
    ):
        super().__init__()

        if mode == "transpose":
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
                nn.Conv3d(in_channels, in_channels // 2, kernel_size=1),
            )

        self.conv = ConvBlock3D(in_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    """
    3D UNet for medical image segmentation.

    Standard encoder-decoder architecture with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 8,
        features: List[int] = [32, 64, 128, 256, 512],
        norm: str = "instance",
        dropout: float = 0.0,
        **kwargs,
    ):
        """
        Initialize UNet3D.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output classes
            features: Feature sizes at each level
            norm: Normalization type
            dropout: Dropout rate
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        # Initial convolution
        self.init_conv = ConvBlock3D(in_channels, features[0], norm=norm)

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(len(features) - 1):
            self.encoders.append(DownBlock3D(features[i], features[i + 1], norm=norm))

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.decoders.append(UpBlock3D(features[i], features[i - 1], norm=norm))

        # Output
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W, D]
            return_features: Whether to return encoder features

        Returns:
            Segmentation logits, optionally with features
        """
        # Initial conv
        x = self.init_conv(x)
        encoder_features = [x]

        # Encoder path
        for encoder in self.encoders:
            x, _ = encoder(x)
            encoder_features.append(x)

        # Decoder path
        encoder_features = encoder_features[:-1]  # Remove bottleneck
        for decoder, skip in zip(self.decoders, reversed(encoder_features)):
            x = decoder(x, skip)

        # Output
        x = self.dropout(x)
        x = self.out_conv(x)

        if return_features:
            return x, encoder_features
        return x

    @property
    def encoder_channels(self) -> List[int]:
        """Get encoder output channels at each stage."""
        return self.features


def build_unet3d(config: Dict[str, Any]) -> UNet3D:
    """
    Build UNet3D from configuration.

    Args:
        config: Model configuration

    Returns:
        UNet3D model
    """
    backbone_config = config.get("model", {}).get("backbone", {})

    return UNet3D(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        features=backbone_config.get("features", [32, 64, 128, 256, 512]),
        norm=backbone_config.get("norm", "instance"),
        dropout=config["model"].get("head", {}).get("dropout", 0.0),
    )
