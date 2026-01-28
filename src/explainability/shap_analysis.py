"""
SHAP (SHapley Additive exPlanations) analysis for model interpretation.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class SHAPAnalyzer:
    """
    SHAP analysis for understanding model predictions.

    Uses gradient-based approximation for efficiency with 3D medical images.
    """

    def __init__(
        self,
        model: nn.Module,
        background_samples: Optional[torch.Tensor] = None,
        n_samples: int = 100,
    ):
        """
        Initialize SHAP analyzer.

        Args:
            model: PyTorch model
            background_samples: Background samples for SHAP
            n_samples: Number of samples for SHAP estimation
        """
        self.model = model
        self.background = background_samples
        self.n_samples = n_samples

    def compute_shap_values(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        method: str = "gradient",
    ) -> np.ndarray:
        """
        Compute SHAP values for input.

        Args:
            input_tensor: Input tensor [B, C, H, W, D]
            target_class: Target class (default: predicted class)
            method: SHAP method ('gradient', 'integrated_gradients', 'deep')

        Returns:
            SHAP values with same shape as input
        """
        if method == "gradient":
            return self._gradient_shap(input_tensor, target_class)
        elif method == "integrated_gradients":
            return self._integrated_gradients(input_tensor, target_class)
        else:
            return self._gradient_shap(input_tensor, target_class)

    def _gradient_shap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute GradientSHAP approximation.

        Args:
            input_tensor: Input tensor
            target_class: Target class

        Returns:
            SHAP values
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Get target
        if target_class is None:
            target_class = output.argmax(dim=1)[0].item()

        # Get class score
        if output.dim() > 2:  # Segmentation
            score = output[0, target_class].mean()
        else:
            score = output[0, target_class]

        # Compute gradients
        self.model.zero_grad()
        score.backward()

        # SHAP approximation: gradient * (input - baseline)
        if self.background is not None:
            baseline = self.background.mean(dim=0, keepdim=True)
        else:
            baseline = torch.zeros_like(input_tensor)

        shap_values = input_tensor.grad * (input_tensor - baseline)

        return shap_values.detach().cpu().numpy()[0]

    def _integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        n_steps: int = 50,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients.

        Args:
            input_tensor: Input tensor
            target_class: Target class
            n_steps: Number of interpolation steps

        Returns:
            Attribution values
        """
        self.model.eval()

        # Baseline
        if self.background is not None:
            baseline = self.background.mean(dim=0, keepdim=True)
        else:
            baseline = torch.zeros_like(input_tensor)

        # Interpolation
        scaled_inputs = [
            baseline + (float(i) / n_steps) * (input_tensor - baseline)
            for i in range(n_steps + 1)
        ]

        # Accumulate gradients
        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.requires_grad_(True)

            output = self.model(scaled_input)

            if target_class is None:
                target_class = output.argmax(dim=1)[0].item()

            if output.dim() > 2:
                score = output[0, target_class].mean()
            else:
                score = output[0, target_class]

            self.model.zero_grad()
            score.backward()

            gradients.append(scaled_input.grad.detach())

        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Integrated gradients: (input - baseline) * avg_gradients
        integrated_grads = (input_tensor - baseline) * avg_gradients

        return integrated_grads.cpu().numpy()[0]

    def visualize_shap(
        self,
        shap_values: np.ndarray,
        input_image: np.ndarray,
        slice_idx: Optional[int] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Visualize SHAP values.

        Args:
            shap_values: SHAP values [C, H, W, D]
            input_image: Original input [C, H, W, D]
            slice_idx: Slice to visualize
            save_path: Path to save figure
        """
        # Average over channels
        shap_mean = np.mean(np.abs(shap_values), axis=0)
        image_mean = np.mean(input_image, axis=0)

        if slice_idx is None:
            slice_idx = shap_mean.shape[-1] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image_mean[:, :, slice_idx].T, cmap="gray")
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # SHAP values
        vmax = np.percentile(np.abs(shap_mean), 99)
        im = axes[1].imshow(
            shap_mean[:, :, slice_idx].T,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        axes[1].set_title("SHAP Values")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Overlay
        axes[2].imshow(image_mean[:, :, slice_idx].T, cmap="gray", alpha=0.7)
        axes[2].imshow(
            shap_mean[:, :, slice_idx].T,
            cmap="RdBu_r",
            alpha=0.5,
            vmin=-vmax,
            vmax=vmax,
        )
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close()

    def summary_plot(
        self,
        shap_values: List[np.ndarray],
        feature_names: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Create SHAP summary plot.

        Args:
            shap_values: List of SHAP values for multiple samples
            feature_names: Names for features/channels
            save_path: Path to save figure
        """
        # Stack and compute mean absolute SHAP
        stacked = np.stack(shap_values)

        # Reduce spatial dimensions
        mean_shap = np.mean(np.abs(stacked), axis=tuple(range(2, stacked.ndim)))

        # Feature importance
        importance = np.mean(mean_shap, axis=0)

        if feature_names is None:
            feature_names = [f"Channel {i}" for i in range(len(importance))]

        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(range(len(importance)), importance[sorted_idx])
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close()
