"""
Attention visualization for transformer-based models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class AttentionVisualizer:
    """
    Visualize attention weights from transformer models.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize attention visualizer.

        Args:
            model: Model with attention layers
        """
        self.model = model
        self.attention_weights = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register hooks to capture attention weights."""
        def get_attention(name):
            def hook(module, input, output):
                # Handle different attention output formats
                if isinstance(output, tuple) and len(output) > 1:
                    # (output, attention_weights)
                    self.attention_weights[name] = output[1].detach()
                elif hasattr(module, "attention_weights"):
                    self.attention_weights[name] = module.attention_weights.detach()
            return hook

        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                module.register_forward_hook(get_attention(name))

    def get_attention_maps(
        self,
        input_tensor: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Get attention maps for input.

        Args:
            input_tensor: Input tensor [B, C, H, W, D]

        Returns:
            Dictionary of attention maps per layer
        """
        self.model.eval()
        self.attention_weights.clear()

        with torch.no_grad():
            _ = self.model(input_tensor)

        # Convert to numpy
        attention_maps = {}
        for name, weights in self.attention_weights.items():
            attention_maps[name] = weights.cpu().numpy()

        return attention_maps

    def visualize_attention(
        self,
        attention_map: np.ndarray,
        input_shape: Tuple[int, ...],
        head_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize attention map.

        Args:
            attention_map: Attention weights [B, heads, seq_len, seq_len] or [B, heads, H, W, D]
            input_shape: Original input shape (H, W, D)
            head_idx: Which attention head to visualize
            save_path: Path to save visualization

        Returns:
            Reshaped attention for visualization
        """
        # Get single head attention
        if len(attention_map.shape) == 4:
            attn = attention_map[0, head_idx]  # [seq_len, seq_len]

            # Average over query dimension for spatial attention
            attn_spatial = attn.mean(axis=0)

            # Reshape to spatial dimensions
            # Assuming attention is over flattened spatial dims
            seq_len = attn_spatial.shape[0]

            # Try to infer spatial shape
            if len(input_shape) == 3:
                h, w, d = input_shape
                # Check if seq_len matches
                possible_sizes = [
                    (h // 4, w // 4, d // 4),
                    (h // 8, w // 8, d // 8),
                    (h // 16, w // 16, d // 16),
                ]
                for size in possible_sizes:
                    if np.prod(size) == seq_len:
                        attn_spatial = attn_spatial.reshape(size)
                        break
        else:
            attn_spatial = attention_map[0, head_idx]

        # Resize to input shape
        from scipy.ndimage import zoom

        if attn_spatial.shape != input_shape:
            zoom_factors = [o / i for o, i in zip(input_shape, attn_spatial.shape)]
            attn_spatial = zoom(attn_spatial, zoom_factors, order=1)

        # Normalize
        attn_spatial = (attn_spatial - attn_spatial.min()) / (attn_spatial.max() - attn_spatial.min() + 1e-8)

        if save_path:
            self._save_attention_figure(attn_spatial, save_path)

        return attn_spatial

    def _save_attention_figure(
        self,
        attention: np.ndarray,
        save_path: str,
    ) -> None:
        """Save attention visualization."""
        if attention.ndim == 3:
            # 3D - show middle slices
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Axial (z middle)
            axes[0].imshow(attention[:, :, attention.shape[2] // 2].T, cmap="hot")
            axes[0].set_title("Axial")
            axes[0].axis("off")

            # Coronal (y middle)
            axes[1].imshow(attention[:, attention.shape[1] // 2, :].T, cmap="hot")
            axes[1].set_title("Coronal")
            axes[1].axis("off")

            # Sagittal (x middle)
            axes[2].imshow(attention[attention.shape[0] // 2, :, :].T, cmap="hot")
            axes[2].set_title("Sagittal")
            axes[2].axis("off")

        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(attention, cmap="hot")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def visualize_all_heads(
        self,
        attention_map: np.ndarray,
        input_shape: Tuple[int, ...],
        save_path: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Visualize all attention heads.

        Args:
            attention_map: Attention weights
            input_shape: Original input shape
            save_path: Path to save visualization

        Returns:
            List of attention maps per head
        """
        num_heads = attention_map.shape[1]
        all_heads = []

        for head_idx in range(num_heads):
            head_attn = self.visualize_attention(attention_map, input_shape, head_idx)
            all_heads.append(head_attn)

        if save_path:
            # Create grid visualization
            cols = min(4, num_heads)
            rows = (num_heads + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = np.array(axes).flatten()

            for idx, (ax, head_attn) in enumerate(zip(axes, all_heads)):
                if head_attn.ndim == 3:
                    # Show middle slice
                    ax.imshow(head_attn[:, :, head_attn.shape[2] // 2].T, cmap="hot")
                else:
                    ax.imshow(head_attn, cmap="hot")
                ax.set_title(f"Head {idx}")
                ax.axis("off")

            # Hide unused axes
            for idx in range(len(all_heads), len(axes)):
                axes[idx].axis("off")

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

        return all_heads
