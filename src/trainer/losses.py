"""
Loss functions for segmentation.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = "mean",
        softmax: bool = True,
        include_background: bool = True,
    ):
        """
        Initialize Dice loss.

        Args:
            smooth: Smoothing factor
            reduction: Reduction method ('mean', 'sum', 'none')
            softmax: Apply softmax to predictions
            include_background: Include background class in loss
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.softmax = softmax
        self.include_background = include_background

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, C, H, W, D]
            target: Ground truth [B, H, W, D] (class indices)

        Returns:
            Dice loss
        """
        num_classes = pred.shape[1]

        # Apply softmax
        if self.softmax:
            pred = F.softmax(pred, dim=1)

        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes)  # [B, H, W, D, C]
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()  # [B, C, H, W, D]

        # Skip background if specified
        if not self.include_background:
            pred = pred[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        # Flatten
        pred_flat = pred.flatten(2)  # [B, C, N]
        target_flat = target_one_hot.flatten(2)

        # Compute Dice
        intersection = (pred_flat * target_flat).sum(-1)
        union = pred_flat.sum(-1) + target_flat.sum(-1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        # Reduction
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize Focal loss.

        Args:
            alpha: Class weights
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, C, H, W, D]
            target: Ground truth [B, H, W, D]

        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(pred, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky loss - generalization of Dice loss with control over FP/FN.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        reduction: str = "mean",
    ):
        """
        Initialize Tversky loss.

        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, C, H, W, D]
            target: Ground truth [B, H, W, D]

        Returns:
            Tversky loss
        """
        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)

        target_one_hot = F.one_hot(target, num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        pred_flat = pred.flatten(2)
        target_flat = target_one_hot.flatten(2)

        tp = (pred_flat * target_flat).sum(-1)
        fp = (pred_flat * (1 - target_flat)).sum(-1)
        fn = ((1 - pred_flat) * target_flat).sum(-1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_loss = 1.0 - tversky

        if self.reduction == "mean":
            return tversky_loss.mean()
        elif self.reduction == "sum":
            return tversky_loss.sum()
        else:
            return tversky_loss


class DiceCELoss(nn.Module):
    """
    Combined Dice and Cross-Entropy loss.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        include_background: bool = True,
    ):
        """
        Initialize combined loss.

        Args:
            dice_weight: Weight for Dice loss
            ce_weight: Weight for CE loss
            class_weights: Per-class weights for CE
            include_background: Include background in Dice
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.dice_loss = DiceLoss(include_background=include_background)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, C, H, W, D]
            target: Ground truth [B, H, W, D]

        Returns:
            Combined loss
        """
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)

        return self.dice_weight * dice + self.ce_weight * ce


def get_loss(config: Dict[str, Any]) -> nn.Module:
    """
    Get loss function from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Loss function
    """
    loss_config = config["training"]["loss"]
    loss_name = loss_config["name"].lower()

    # Class weights
    class_weights = loss_config.get("class_weights")
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

    if loss_name == "dice":
        return DiceLoss()
    elif loss_name == "ce" or loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "dice_ce":
        return DiceCELoss(
            dice_weight=loss_config.get("dice_weight", 0.5),
            ce_weight=loss_config.get("ce_weight", 0.5),
            class_weights=class_weights,
        )
    elif loss_name == "focal":
        return FocalLoss(alpha=class_weights)
    elif loss_name == "tversky":
        return TverskyLoss(
            alpha=loss_config.get("tversky_alpha", 0.5),
            beta=loss_config.get("tversky_beta", 0.5),
        )
    else:
        return DiceCELoss()
