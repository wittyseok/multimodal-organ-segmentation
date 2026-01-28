"""
Visualization utilities for medical images and results.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Visualization utilities for medical image segmentation."""

    # Default colormap for segmentation labels
    LABEL_COLORS = {
        0: [0, 0, 0],        # Background - Black
        1: [255, 0, 0],      # Bladder - Red
        2: [0, 255, 0],      # Right Kidney - Green
        3: [0, 0, 255],      # Left Kidney - Blue
        4: [255, 255, 0],    # Heart - Yellow
        5: [255, 0, 255],    # Liver - Magenta
        6: [0, 255, 255],    # Spleen - Cyan
        7: [255, 128, 0],    # Brain - Orange
    }

    LABEL_NAMES = {
        0: "Background",
        1: "Bladder",
        2: "Right Kidney",
        3: "Left Kidney",
        4: "Heart",
        5: "Liver",
        6: "Spleen",
        7: "Brain",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize visualizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.figsize = self.config.get("figsize", (12, 8))
        self.dpi = self.config.get("dpi", 100)

    def plot_slice(
        self,
        image: np.ndarray,
        slice_idx: Optional[int] = None,
        axis: int = 2,
        title: str = "",
        cmap: str = "gray",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot a single slice from 3D volume.

        Args:
            image: 3D image array
            slice_idx: Slice index (default: middle slice)
            axis: Axis to slice along (0, 1, or 2)
            title: Plot title
            cmap: Colormap
            save_path: Path to save figure
            show: Whether to display figure
        """
        if slice_idx is None:
            slice_idx = image.shape[axis] // 2

        # Get slice
        if axis == 0:
            slice_data = image[slice_idx, :, :]
        elif axis == 1:
            slice_data = image[:, slice_idx, :]
        else:
            slice_data = image[:, :, slice_idx]

        # Plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        im = ax.imshow(slice_data.T, cmap=cmap, origin="lower")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_multimodal(
        self,
        images: Dict[str, np.ndarray],
        slice_idx: Optional[int] = None,
        axis: int = 2,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot multiple modalities side by side.

        Args:
            images: Dictionary of {modality_name: image_array}
            slice_idx: Slice index
            axis: Axis to slice along
            save_path: Path to save figure
            show: Whether to display figure
        """
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4), dpi=self.dpi)

        if n_images == 1:
            axes = [axes]

        for ax, (name, image) in zip(axes, images.items()):
            if slice_idx is None:
                idx = image.shape[axis] // 2
            else:
                idx = slice_idx

            if axis == 0:
                slice_data = image[idx, :, :]
            elif axis == 1:
                slice_data = image[:, idx, :]
            else:
                slice_data = image[:, :, idx]

            ax.imshow(slice_data.T, cmap="gray", origin="lower")
            ax.set_title(name)
            ax.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_segmentation(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        slice_idx: Optional[int] = None,
        axis: int = 2,
        alpha: float = 0.5,
        title: str = "",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot image with segmentation overlay.

        Args:
            image: 3D image array
            segmentation: 3D segmentation mask
            slice_idx: Slice index
            axis: Axis to slice along
            alpha: Overlay transparency
            title: Plot title
            save_path: Path to save figure
            show: Whether to display figure
        """
        if slice_idx is None:
            slice_idx = image.shape[axis] // 2

        # Get slices
        if axis == 0:
            img_slice = image[slice_idx, :, :]
            seg_slice = segmentation[slice_idx, :, :]
        elif axis == 1:
            img_slice = image[:, slice_idx, :]
            seg_slice = segmentation[:, slice_idx, :]
        else:
            img_slice = image[:, :, slice_idx]
            seg_slice = segmentation[:, :, slice_idx]

        # Create RGB segmentation overlay
        seg_rgb = self._labels_to_rgb(seg_slice)

        # Normalize image
        img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        img_rgb = np.stack([img_norm] * 3, axis=-1)

        # Blend
        mask = seg_slice > 0
        blended = img_rgb.copy()
        blended[mask] = (1 - alpha) * img_rgb[mask] + alpha * seg_rgb[mask] / 255.0

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=self.dpi)

        axes[0].imshow(img_slice.T, cmap="gray", origin="lower")
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(seg_rgb.transpose(1, 0, 2), origin="lower")
        axes[1].set_title("Segmentation")
        axes[1].axis("off")

        axes[2].imshow(blended.transpose(1, 0, 2), origin="lower")
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot training curves.

        Args:
            history: Dictionary of {metric_name: values_list}
            save_path: Path to save figure
            show: Whether to display figure
        """
        n_metrics = len(history)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), dpi=self.dpi)

        if n_metrics == 1:
            axes = [axes]

        for ax, (name, values) in zip(axes, history.items()):
            ax.plot(values)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize
            save_path: Path to save figure
            show: Whether to display figure
        """
        if normalize:
            cm = confusion_matrix.astype("float") / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-8)
        else:
            cm = confusion_matrix

        if class_names is None:
            class_names = [str(i) for i in range(len(cm))]

        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)
        im = ax.imshow(cm, cmap="Blues")

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def _labels_to_rgb(self, labels: np.ndarray) -> np.ndarray:
        """Convert label mask to RGB image."""
        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)

        for label_id, color in self.LABEL_COLORS.items():
            mask = labels == label_id
            rgb[mask] = color

        return rgb

    @staticmethod
    def create_montage(
        images: List[np.ndarray],
        grid_shape: Optional[Tuple[int, int]] = None,
        padding: int = 2,
    ) -> np.ndarray:
        """
        Create montage from list of 2D images.

        Args:
            images: List of 2D image arrays
            grid_shape: (rows, cols) for grid layout
            padding: Padding between images

        Returns:
            Montage image
        """
        n = len(images)

        if grid_shape is None:
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
        else:
            rows, cols = grid_shape

        # Get max dimensions
        h = max(img.shape[0] for img in images)
        w = max(img.shape[1] for img in images)

        # Create montage
        montage = np.zeros((rows * (h + padding) - padding, cols * (w + padding) - padding))

        for i, img in enumerate(images):
            r, c = i // cols, i % cols
            y = r * (h + padding)
            x = c * (w + padding)

            # Center image in cell
            dy = (h - img.shape[0]) // 2
            dx = (w - img.shape[1]) // 2

            montage[y + dy : y + dy + img.shape[0], x + dx : x + dx + img.shape[1]] = img

        return montage
