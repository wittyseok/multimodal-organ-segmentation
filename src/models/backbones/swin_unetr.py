"""
Swin UNETR backbone for 3D medical image segmentation.

Based on: "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images"
https://arxiv.org/abs/2201.01266
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:
    from monai.networks.nets import SwinUNETR as MONAISwinUNETR
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


class SwinUNETR(nn.Module):
    """
    Swin UNETR model for 3D medical image segmentation.

    Wrapper around MONAI's SwinUNETR with additional functionality
    for multi-modal inputs and feature extraction.
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 8,
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str = "merging",
        use_v2: bool = False,
        pretrained: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize SwinUNETR.

        Args:
            img_size: Input image size (H, W, D)
            in_channels: Number of input channels
            out_channels: Number of output segmentation classes
            feature_size: Base feature size
            depths: Number of Swin Transformer blocks at each stage
            num_heads: Number of attention heads at each stage
            norm_name: Normalization type
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Stochastic depth rate
            normalize: Whether to normalize output
            use_checkpoint: Use gradient checkpointing to save memory
            spatial_dims: Spatial dimensions (2 or 3)
            downsample: Downsampling method
            use_v2: Use Swin Transformer V2
            pretrained: Path to pretrained weights
        """
        super().__init__()

        if not HAS_MONAI:
            raise ImportError("MONAI is required for SwinUNETR. Install with: pip install monai")

        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size

        # Build MONAI SwinUNETR
        self.model = MONAISwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
            use_v2=use_v2,
        )

        # Load pretrained weights if provided
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W, D]
            return_features: Whether to return intermediate features

        Returns:
            Segmentation logits, optionally with feature maps
        """
        if return_features:
            return self._forward_features(x)
        return self.model(x)

    def _forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with feature extraction."""
        # Access internal encoder features
        # Note: This depends on MONAI's implementation details
        hidden_states = self.model.swinViT(x, self.model.normalize)
        features = [hidden_states[i] for i in range(len(hidden_states))]

        # Get output
        out = self.model(x)

        return out, features

    def load_pretrained(self, path: str) -> None:
        """Load pretrained weights."""
        state_dict = torch.load(path, map_location="cpu")

        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Load with strict=False to allow partial loading
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

    def get_encoder(self) -> nn.Module:
        """Get encoder part of the network."""
        return self.model.swinViT

    def get_decoder(self) -> nn.Module:
        """Get decoder part of the network."""
        return nn.ModuleList([
            self.model.decoder5,
            self.model.decoder4,
            self.model.decoder3,
            self.model.decoder2,
            self.model.decoder1,
        ])

    @property
    def encoder_channels(self) -> List[int]:
        """Get encoder output channels at each stage."""
        return [
            self.feature_size,
            self.feature_size * 2,
            self.feature_size * 4,
            self.feature_size * 8,
            self.feature_size * 16,
        ]


def build_swin_unetr(config: Dict[str, Any]) -> SwinUNETR:
    """
    Build SwinUNETR from configuration.

    Args:
        config: Model configuration

    Returns:
        SwinUNETR model
    """
    backbone_config = config.get("model", {}).get("backbone", {})

    return SwinUNETR(
        img_size=tuple(backbone_config.get("img_size", [96, 96, 96])),
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        feature_size=backbone_config.get("feature_size", 48),
        depths=tuple(backbone_config.get("depths", [2, 2, 2, 2])),
        num_heads=tuple(backbone_config.get("num_heads", [3, 6, 12, 24])),
        drop_rate=config["model"].get("head", {}).get("dropout", 0.0),
        use_checkpoint=config.get("training", {}).get("use_checkpoint", False),
    )
