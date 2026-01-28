"""
Evaluation metrics for segmentation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class DiceMetric:
    """
    Dice coefficient metric for segmentation evaluation.
    """

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        reduction: str = "mean",
    ):
        """
        Initialize Dice metric.

        Args:
            num_classes: Number of segmentation classes
            include_background: Include background in computation
            reduction: Reduction method ('mean', 'none')
        """
        self.num_classes = num_classes
        self.include_background = include_background
        self.reduction = reduction

        self.reset()

    def reset(self) -> None:
        """Reset metric state."""
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
        self.count = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """
        Update metric with new predictions.

        Args:
            pred: Predictions [B, H, W, D] (class indices)
            target: Ground truth [B, H, W, D] (class indices)
        """
        pred = pred.cpu()
        target = target.cpu()

        for c in range(self.num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            self.intersection[c] += intersection
            self.union[c] += union

        self.count += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute Dice scores.

        Returns:
            Dictionary with Dice scores
        """
        smooth = 1e-5
        dice_per_class = (2.0 * self.intersection + smooth) / (self.union + smooth)

        # Exclude background if specified
        start_idx = 0 if self.include_background else 1
        dice_foreground = dice_per_class[start_idx:]

        results = {
            "dice": dice_foreground.mean().item(),
            "dice_per_class": dice_per_class.tolist(),
        }

        return results


class HausdorffDistance:
    """
    Hausdorff distance metric for boundary evaluation.
    """

    def __init__(self, percentile: float = 95):
        """
        Initialize Hausdorff distance.

        Args:
            percentile: Percentile for robust HD (e.g., 95%)
        """
        self.percentile = percentile
        self.distances = []

    def reset(self) -> None:
        """Reset metric state."""
        self.distances = []

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        spacing: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Update with new predictions.

        Args:
            pred: Predictions [B, H, W, D]
            target: Ground truth [B, H, W, D]
            spacing: Voxel spacing (optional)
        """
        from scipy.ndimage import distance_transform_edt

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        spacing = spacing or (1.0, 1.0, 1.0)

        for b in range(pred.shape[0]):
            pred_b = pred[b] > 0
            target_b = target[b] > 0

            if pred_b.sum() == 0 or target_b.sum() == 0:
                continue

            # Distance transforms
            dist_pred = distance_transform_edt(~pred_b, sampling=spacing)
            dist_target = distance_transform_edt(~target_b, sampling=spacing)

            # Surface distances
            border_pred = pred_b ^ np.roll(pred_b, 1, axis=0)
            border_target = target_b ^ np.roll(target_b, 1, axis=0)

            distances_pred_to_target = dist_target[border_pred]
            distances_target_to_pred = dist_pred[border_target]

            all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])

            if len(all_distances) > 0:
                hd = np.percentile(all_distances, self.percentile)
                self.distances.append(hd)

    def compute(self) -> Dict[str, float]:
        """Compute Hausdorff distance."""
        if len(self.distances) == 0:
            return {"hausdorff_distance": float("inf")}

        return {
            "hausdorff_distance": np.mean(self.distances),
            "hausdorff_distance_std": np.std(self.distances),
        }


class ConfusionMatrix:
    """
    Confusion matrix for multi-class segmentation.
    """

    def __init__(self, num_classes: int):
        """
        Initialize confusion matrix.

        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset confusion matrix."""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update confusion matrix.

        Args:
            pred: Predictions [B, H, W, D]
            target: Ground truth [B, H, W, D]
        """
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        for p, t in zip(pred, target):
            self.matrix[t, p] += 1

    def compute(self) -> Dict[str, Any]:
        """
        Compute metrics from confusion matrix.

        Returns:
            Dictionary with precision, recall, accuracy, etc.
        """
        # Per-class metrics
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Overall accuracy
        accuracy = tp.sum() / (self.matrix.sum() + 1e-8)

        return {
            "accuracy": accuracy,
            "precision": precision.mean(),
            "recall": recall.mean(),
            "f1": f1.mean(),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "confusion_matrix": self.matrix.tolist(),
        }


def get_metrics(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get metrics from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of metric instances
    """
    num_classes = config["model"]["out_channels"]

    return {
        "dice": DiceMetric(num_classes=num_classes),
        "confusion": ConfusionMatrix(num_classes=num_classes),
    }
