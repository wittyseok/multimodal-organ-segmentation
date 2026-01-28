"""
Model factory for building segmentation models.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.backbones.swin_unetr import SwinUNETR, build_swin_unetr
from src.models.backbones.unet import UNet3D, build_unet3d
from src.models.backbones.dual_encoder import DualEncoder, build_dual_encoder


# Model registry
MODEL_REGISTRY = {
    "swin_unetr": build_swin_unetr,
    "unet": build_unet3d,
    "unet3d": build_unet3d,
    "dual_encoder": build_dual_encoder,
}


class MultiModalSegmentationModel(nn.Module):
    """
    Wrapper for multi-modal segmentation models.

    Handles input preprocessing and output postprocessing.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: Dict[str, Any],
    ):
        """
        Initialize model.

        Args:
            backbone: Backbone network
            config: Configuration dictionary
        """
        super().__init__()

        self.backbone = backbone
        self.config = config
        self.num_modalities = len(config["data"]["modalities"])

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        """
        Forward pass.

        Args:
            x: Input tensor [B, num_modalities, H, W, D]
            return_features: Whether to return intermediate features

        Returns:
            Segmentation output
        """
        return self.backbone(x, return_features=return_features)

    def load_pretrained(self, path: str) -> None:
        """Load pretrained weights."""
        if hasattr(self.backbone, "load_pretrained"):
            self.backbone.load_pretrained(path)
        else:
            state_dict = torch.load(path, map_location="cpu")
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.load_state_dict(state_dict, strict=False)


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Model instance
    """
    model_name = config["model"]["name"].lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    # Update in_channels based on modalities
    num_modalities = len(config["data"]["modalities"])

    # For models that expect concatenated input
    if model_name in ["swin_unetr", "unet", "unet3d"]:
        config["model"]["in_channels"] = num_modalities

    # Build backbone
    backbone = MODEL_REGISTRY[model_name](config)

    # Wrap in MultiModalSegmentationModel
    model = MultiModalSegmentationModel(backbone, config)

    # Move to device
    device = config["hardware"]["device"]
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to("mps")

    return model


def get_model(config: Dict[str, Any]) -> nn.Module:
    """Alias for build_model."""
    return build_model(config)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce state_dict matching

    Returns:
        Checkpoint dictionary (with training state, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict)

    return checkpoint


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    checkpoint_path: str,
    **kwargs,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        checkpoint_path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    checkpoint.update(kwargs)

    torch.save(checkpoint, checkpoint_path)
