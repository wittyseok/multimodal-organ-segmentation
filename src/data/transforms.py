"""
Transform functions for medical image preprocessing and augmentation.

Designed for multi-modal 3D medical images.
"""

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from scipy import ndimage
    from scipy.ndimage import zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomFlip:
    """Random flip along specified axes."""

    def __init__(self, axes: Sequence[int] = (0, 1, 2), prob: float = 0.5):
        """
        Args:
            axes: Axes to potentially flip (relative to spatial dims)
            prob: Probability of flipping each axis
        """
        self.axes = axes
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]  # [C, H, W, D]

        for axis in self.axes:
            if random.random() < self.prob:
                # Axis + 1 because first dim is channels
                image = np.flip(image, axis=axis + 1)
                if "label" in sample:
                    sample["label"] = np.flip(sample["label"], axis=axis)

        sample["image"] = image.copy()
        if "label" in sample:
            sample["label"] = sample["label"].copy()

        return sample


class RandomRotate90:
    """Random 90-degree rotation in xy plane."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.prob:
            k = random.randint(1, 3)  # Number of 90-degree rotations
            image = sample["image"]

            # Rotate in xy plane (axes 1, 2 for image with channels)
            image = np.rot90(image, k=k, axes=(1, 2))
            sample["image"] = image.copy()

            if "label" in sample:
                label = np.rot90(sample["label"], k=k, axes=(0, 1))
                sample["label"] = label.copy()

        return sample


class RandomIntensityShift:
    """Random intensity shift and scale."""

    def __init__(
        self,
        shift_range: Tuple[float, float] = (-0.1, 0.1),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        prob: float = 0.5,
        per_channel: bool = True,
    ):
        """
        Args:
            shift_range: Range for additive shift
            scale_range: Range for multiplicative scale
            prob: Probability of applying transform
            per_channel: Apply different shifts/scales per channel
        """
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.prob = prob
        self.per_channel = per_channel

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.prob:
            image = sample["image"]
            n_channels = image.shape[0]

            if self.per_channel:
                for c in range(n_channels):
                    shift = random.uniform(*self.shift_range)
                    scale = random.uniform(*self.scale_range)
                    image[c] = image[c] * scale + shift
            else:
                shift = random.uniform(*self.shift_range)
                scale = random.uniform(*self.scale_range)
                image = image * scale + shift

            sample["image"] = image

        return sample


class RandomGaussianNoise:
    """Add random Gaussian noise."""

    def __init__(self, mean: float = 0.0, std: float = 0.1, prob: float = 0.5):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.prob:
            image = sample["image"]
            noise = np.random.normal(self.mean, self.std, image.shape).astype(image.dtype)
            sample["image"] = image + noise

        return sample


class RandomCrop:
    """Random crop to specified size."""

    def __init__(self, size: Tuple[int, int, int]):
        """
        Args:
            size: Target crop size (H, W, D)
        """
        self.size = size

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]  # [C, H, W, D]
        _, h, w, d = image.shape

        # Calculate valid crop range
        max_h = max(0, h - self.size[0])
        max_w = max(0, w - self.size[1])
        max_d = max(0, d - self.size[2])

        start_h = random.randint(0, max_h) if max_h > 0 else 0
        start_w = random.randint(0, max_w) if max_w > 0 else 0
        start_d = random.randint(0, max_d) if max_d > 0 else 0

        # Crop image
        sample["image"] = image[
            :,
            start_h : start_h + self.size[0],
            start_w : start_w + self.size[1],
            start_d : start_d + self.size[2],
        ].copy()

        # Crop label if present
        if "label" in sample:
            sample["label"] = sample["label"][
                start_h : start_h + self.size[0],
                start_w : start_w + self.size[1],
                start_d : start_d + self.size[2],
            ].copy()

        return sample


class CenterCrop:
    """Center crop to specified size."""

    def __init__(self, size: Tuple[int, int, int]):
        self.size = size

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]
        _, h, w, d = image.shape

        start_h = max(0, (h - self.size[0]) // 2)
        start_w = max(0, (w - self.size[1]) // 2)
        start_d = max(0, (d - self.size[2]) // 2)

        sample["image"] = image[
            :,
            start_h : start_h + self.size[0],
            start_w : start_w + self.size[1],
            start_d : start_d + self.size[2],
        ].copy()

        if "label" in sample:
            sample["label"] = sample["label"][
                start_h : start_h + self.size[0],
                start_w : start_w + self.size[1],
                start_d : start_d + self.size[2],
            ].copy()

        return sample


class Resize:
    """Resize to target size."""

    def __init__(self, size: Tuple[int, int, int], order: int = 1):
        """
        Args:
            size: Target size (H, W, D)
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required for Resize transform")
        self.size = size
        self.order = order

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]  # [C, H, W, D]
        _, h, w, d = image.shape

        # Calculate zoom factors
        zoom_h = self.size[0] / h
        zoom_w = self.size[1] / w
        zoom_d = self.size[2] / d

        # Resize each channel
        resized_channels = []
        for c in range(image.shape[0]):
            resized = zoom(image[c], (zoom_h, zoom_w, zoom_d), order=self.order)
            resized_channels.append(resized)

        sample["image"] = np.stack(resized_channels, axis=0)

        if "label" in sample:
            # Use nearest neighbor for labels
            sample["label"] = zoom(sample["label"], (zoom_h, zoom_w, zoom_d), order=0)

        return sample


class Normalize:
    """Normalize image intensities."""

    def __init__(
        self,
        mean: Optional[Union[float, Sequence[float]]] = None,
        std: Optional[Union[float, Sequence[float]]] = None,
        per_channel: bool = True,
    ):
        """
        Args:
            mean: Mean value(s) for normalization
            std: Std value(s) for normalization
            per_channel: Normalize each channel independently
        """
        self.mean = mean
        self.std = std
        self.per_channel = per_channel

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]

        if self.per_channel:
            for c in range(image.shape[0]):
                if self.mean is None:
                    mean = image[c].mean()
                elif isinstance(self.mean, (list, tuple)):
                    mean = self.mean[c]
                else:
                    mean = self.mean

                if self.std is None:
                    std = image[c].std() + 1e-8
                elif isinstance(self.std, (list, tuple)):
                    std = self.std[c]
                else:
                    std = self.std

                image[c] = (image[c] - mean) / std
        else:
            mean = self.mean if self.mean is not None else image.mean()
            std = self.std if self.std is not None else (image.std() + 1e-8)
            image = (image - mean) / std

        sample["image"] = image
        return sample


class ClipIntensity:
    """Clip intensity values to specified range."""

    def __init__(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        percentile: Optional[Tuple[float, float]] = None,
    ):
        """
        Args:
            min_val: Minimum value
            max_val: Maximum value
            percentile: Percentile range (e.g., (1, 99))
        """
        self.min_val = min_val
        self.max_val = max_val
        self.percentile = percentile

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]

        for c in range(image.shape[0]):
            if self.percentile is not None:
                min_val = np.percentile(image[c], self.percentile[0])
                max_val = np.percentile(image[c], self.percentile[1])
            else:
                min_val = self.min_val if self.min_val is not None else image[c].min()
                max_val = self.max_val if self.max_val is not None else image[c].max()

            image[c] = np.clip(image[c], min_val, max_val)

        sample["image"] = image
        return sample


class ScaleIntensity:
    """Scale intensity to [0, 1] range."""

    def __init__(self, per_channel: bool = True):
        self.per_channel = per_channel

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]

        if self.per_channel:
            for c in range(image.shape[0]):
                min_val = image[c].min()
                max_val = image[c].max()
                if max_val - min_val > 1e-8:
                    image[c] = (image[c] - min_val) / (max_val - min_val)
        else:
            min_val = image.min()
            max_val = image.max()
            if max_val - min_val > 1e-8:
                image = (image - min_val) / (max_val - min_val)

        sample["image"] = image
        return sample


class ModalitySpecificNormalize:
    """Apply modality-specific normalization."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration with modality-specific parameters
        """
        self.config = config
        self.modalities = config["data"]["modalities"]
        self.preprocess_config = config["data"]["preprocessing"]

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]

        for c, modality in enumerate(self.modalities):
            mod_config = self.preprocess_config.get(modality.lower(), {})

            if modality == "CT":
                # CT windowing
                center = mod_config.get("window_center", 0)
                width = mod_config.get("window_width", 400)
                min_val = center - width / 2
                max_val = center + width / 2
                image[c] = np.clip(image[c], min_val, max_val)
                image[c] = (image[c] - min_val) / (max_val - min_val)

            elif modality == "PET":
                # PET normalization (assume already SUV)
                if mod_config.get("normalize", True):
                    max_val = image[c].max()
                    if max_val > 0:
                        image[c] = image[c] / max_val

            elif modality in ("MRI", "US"):
                # Standard z-score normalization
                if mod_config.get("normalize", True):
                    mean = image[c].mean()
                    std = image[c].std() + 1e-8
                    image[c] = (image[c] - mean) / std

        sample["image"] = image
        return sample


def get_transforms(config: Dict[str, Any], mode: str = "train") -> Compose:
    """
    Get transforms based on configuration.

    Args:
        config: Configuration dictionary
        mode: Dataset mode ('train', 'val', 'test')

    Returns:
        Composed transforms
    """
    transforms_list = []

    # Modality-specific normalization
    transforms_list.append(ModalitySpecificNormalize(config))

    if mode == "train":
        aug_config = config["data"].get("augmentation", {})

        if aug_config.get("enabled", False):
            # Random flip
            if aug_config.get("random_flip", True):
                transforms_list.append(RandomFlip(prob=0.5))

            # Random rotation
            if aug_config.get("random_rotate", 0) > 0:
                transforms_list.append(RandomRotate90(prob=0.5))

            # Random intensity
            if aug_config.get("random_intensity", 0) > 0:
                shift = aug_config["random_intensity"]
                transforms_list.append(
                    RandomIntensityShift(shift_range=(-shift, shift), prob=0.3)
                )

            # Random noise
            transforms_list.append(RandomGaussianNoise(std=0.05, prob=0.2))

    # Resize if needed
    backbone_config = config["model"].get("backbone", {})
    img_size = backbone_config.get("img_size", [96, 96, 96])
    if len(img_size) == 3:
        transforms_list.append(Resize(tuple(img_size)))

    return Compose(transforms_list)
