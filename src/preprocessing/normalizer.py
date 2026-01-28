"""
Intensity normalization utilities for medical images.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np


class IntensityNormalizer:
    """
    Intensity normalization for multi-modal medical images.

    Supports modality-specific normalization strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize normalizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def normalize_ct(
        self,
        image: np.ndarray,
        window_center: float = 0,
        window_width: float = 400,
        output_range: Tuple[float, float] = (0, 1),
    ) -> np.ndarray:
        """
        Normalize CT image using windowing.

        Args:
            image: CT image array (in HU)
            window_center: Window center in HU
            window_width: Window width in HU
            output_range: Output intensity range

        Returns:
            Normalized image
        """
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2

        # Apply window
        normalized = np.clip(image, min_val, max_val)

        # Scale to output range
        normalized = (normalized - min_val) / (max_val - min_val)
        normalized = normalized * (output_range[1] - output_range[0]) + output_range[0]

        return normalized.astype(np.float32)

    def normalize_pet(
        self,
        image: np.ndarray,
        method: str = "max",
        reference_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Normalize PET/SUV image.

        Args:
            image: PET/SUV image array
            method: Normalization method ('max', 'percentile', 'reference')
            reference_value: Reference value for normalization

        Returns:
            Normalized image
        """
        if method == "max":
            max_val = image.max()
            if max_val > 0:
                normalized = image / max_val
            else:
                normalized = image.copy()

        elif method == "percentile":
            percentile_val = np.percentile(image[image > 0], 99) if np.any(image > 0) else 1.0
            normalized = image / percentile_val
            normalized = np.clip(normalized, 0, 1)

        elif method == "reference":
            if reference_value is None:
                raise ValueError("reference_value required for reference normalization")
            normalized = image / reference_value
            normalized = np.clip(normalized, 0, None)

        else:
            normalized = image.copy()

        return normalized.astype(np.float32)

    def normalize_mri(
        self,
        image: np.ndarray,
        method: str = "zscore",
        percentile_range: Tuple[float, float] = (1, 99),
    ) -> np.ndarray:
        """
        Normalize MRI image.

        Args:
            image: MRI image array
            method: Normalization method ('zscore', 'minmax', 'percentile')
            percentile_range: Percentile range for clipping

        Returns:
            Normalized image
        """
        # Create foreground mask
        threshold = np.percentile(image, percentile_range[0])
        mask = image > threshold

        if method == "zscore":
            if mask.sum() > 0:
                mean = image[mask].mean()
                std = image[mask].std() + 1e-8
                normalized = (image - mean) / std
            else:
                normalized = image.copy()

        elif method == "minmax":
            min_val = np.percentile(image, percentile_range[0])
            max_val = np.percentile(image, percentile_range[1])
            normalized = np.clip(image, min_val, max_val)
            normalized = (normalized - min_val) / (max_val - min_val + 1e-8)

        elif method == "percentile":
            min_val = np.percentile(image, percentile_range[0])
            max_val = np.percentile(image, percentile_range[1])
            normalized = np.clip(image, min_val, max_val)
            normalized = (normalized - min_val) / (max_val - min_val + 1e-8)

        else:
            normalized = image.copy()

        return normalized.astype(np.float32)

    def normalize_ultrasound(
        self,
        image: np.ndarray,
        method: str = "minmax",
    ) -> np.ndarray:
        """
        Normalize ultrasound image.

        Args:
            image: Ultrasound image array
            method: Normalization method

        Returns:
            Normalized image
        """
        if method == "minmax":
            min_val = image.min()
            max_val = image.max()
            if max_val - min_val > 1e-8:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = image.copy()

        elif method == "zscore":
            mean = image.mean()
            std = image.std() + 1e-8
            normalized = (image - mean) / std

        else:
            normalized = image.copy()

        return normalized.astype(np.float32)

    def normalize(
        self,
        image: np.ndarray,
        modality: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply modality-specific normalization.

        Args:
            image: Image array
            modality: Imaging modality (CT, PET, MRI, US)
            **kwargs: Modality-specific parameters

        Returns:
            Normalized image
        """
        modality = modality.upper()

        if modality == "CT":
            return self.normalize_ct(image, **kwargs)
        elif modality == "PET":
            return self.normalize_pet(image, **kwargs)
        elif modality == "MRI":
            return self.normalize_mri(image, **kwargs)
        elif modality == "US":
            return self.normalize_ultrasound(image, **kwargs)
        else:
            # Default: min-max normalization
            min_val = image.min()
            max_val = image.max()
            if max_val - min_val > 1e-8:
                return ((image - min_val) / (max_val - min_val)).astype(np.float32)
            return image.astype(np.float32)

    def normalize_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        modality: str,
        **kwargs,
    ) -> str:
        """
        Normalize NIfTI file and save.

        Args:
            input_path: Input NIfTI path
            output_path: Output NIfTI path
            modality: Imaging modality
            **kwargs: Modality-specific parameters

        Returns:
            Output file path
        """
        nii = nib.load(str(input_path))
        data = nii.get_fdata()

        normalized = self.normalize(data, modality, **kwargs)

        output_nii = nib.Nifti1Image(normalized, nii.affine, nii.header)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(output_nii, str(output_path))

        return str(output_path)
