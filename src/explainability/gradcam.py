"""
GradCAM and GradCAM++ for model interpretation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM).

    Visualizes which regions of the input image contribute most
    to a particular class prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str],
        use_cuda: bool = True,
    ):
        """
        Initialize GradCAM.

        Args:
            model: PyTorch model
            target_layers: Names of layers to compute CAM for
            use_cuda: Whether to use CUDA
        """
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.activations = {}
        self.gradients = {}

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layers."""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(get_activation(name))
                module.register_full_backward_hook(get_gradient(name))

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_layer: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute GradCAM.

        Args:
            input_tensor: Input image tensor [B, C, H, W, D]
            target_class: Target class for CAM (default: predicted class)
            target_layer: Specific layer to use (default: first target layer)

        Returns:
            CAM heatmap [H, W, D]
        """
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
            self.model = self.model.cuda()

        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1)[0].item()

        # Get target layer
        if target_layer is None:
            target_layer = self.target_layers[0]

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        # For segmentation, we need to select a specific output location
        # Using global max for simplicity
        if output.dim() > 2:  # Segmentation output [B, C, H, W, D]
            class_output = output[0, target_class].max()
        else:
            class_output = output[0, target_class]

        class_output.backward(retain_graph=True)

        # Get activations and gradients
        activations = self.activations[target_layer]
        gradients = self.gradients[target_layer]

        # Global average pooling of gradients
        if gradients.dim() == 5:  # 3D
            weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        else:  # 2D
            weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze()

        # Resize to input size
        if cam.dim() == 3:
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=input_tensor.shape[2:],
                mode="trilinear",
                align_corners=False,
            ).squeeze()
        else:
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=input_tensor.shape[2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


class GradCAMPlusPlus(GradCAM):
    """
    GradCAM++ with improved weighting scheme.

    Better handles multiple instances of the same class.
    """

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_layer: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute GradCAM++.

        Args:
            input_tensor: Input image tensor [B, C, H, W, D]
            target_class: Target class for CAM
            target_layer: Specific layer to use

        Returns:
            CAM heatmap [H, W, D]
        """
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
            self.model = self.model.cuda()

        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)[0].item()

        if target_layer is None:
            target_layer = self.target_layers[0]

        self.model.zero_grad()

        # Backward pass
        if output.dim() > 2:
            class_output = output[0, target_class].max()
        else:
            class_output = output[0, target_class]

        class_output.backward(retain_graph=True)

        activations = self.activations[target_layer]
        gradients = self.gradients[target_layer]

        # GradCAM++ weighting
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        # Spatial sum of activations
        if gradients.dim() == 5:
            sum_activations = activations.sum(dim=(2, 3, 4), keepdim=True)
        else:
            sum_activations = activations.sum(dim=(2, 3), keepdim=True)

        # Alpha weights
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom

        # Positive gradients only
        weights = (alpha * F.relu(gradients))

        if weights.dim() == 5:
            weights = weights.sum(dim=(2, 3, 4), keepdim=True)
        else:
            weights = weights.sum(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze()

        # Resize
        if cam.dim() == 3:
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=input_tensor.shape[2:],
                mode="trilinear",
                align_corners=False,
            ).squeeze()
        else:
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=input_tensor.shape[2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def visualize_gradcam(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay GradCAM on image.

    Args:
        image: Original image [H, W] or [H, W, D]
        cam: CAM heatmap [H, W] or [H, W, D]
        alpha: Overlay transparency
        colormap: Matplotlib colormap name

    Returns:
        Overlaid visualization
    """
    import matplotlib.pyplot as plt

    # Get colormap
    cmap = plt.cm.get_cmap(colormap)

    # Normalize image
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Apply colormap to CAM
    cam_colored = cmap(cam)[..., :3]  # Remove alpha channel

    # Blend
    if image_norm.ndim == 2:
        image_rgb = np.stack([image_norm] * 3, axis=-1)
    else:
        image_rgb = np.stack([image_norm] * 3, axis=-1)

    overlay = (1 - alpha) * image_rgb + alpha * cam_colored

    return np.clip(overlay, 0, 1)
