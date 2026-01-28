"""
Image registration for multi-modal medical imaging.

Supports registration between different imaging modalities (CT, PET, MRI, US).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


class ImageRegistration:
    """
    Multi-modal image registration.

    Supports various registration methods:
    - Translation
    - Rigid (translation + rotation)
    - Affine
    - Deformable (B-spline)
    """

    REGISTRATION_METHODS = ["translation", "rigid", "affine", "deformable"]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize registration.

        Args:
            config: Configuration dictionary
        """
        if not HAS_SITK:
            raise ImportError("SimpleITK is required for image registration")

        self.config = config
        self.reg_config = config.get("data", {}).get("registration", {})
        self.method = self.reg_config.get("method", "translation")
        self.metric = self.reg_config.get("metric", "mattes_mutual_information")

    def register(
        self,
        data_dir: Union[str, Path],
        reference_modality: str = "CT",
        target_modalities: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Register all modalities to reference modality.

        Args:
            data_dir: Directory containing modality subdirectories
            reference_modality: Reference modality for registration
            target_modalities: Modalities to register (default: all except reference)

        Returns:
            Dictionary of {modality: registered_path}
        """
        data_dir = Path(data_dir)
        results = {}

        # Get modalities
        modalities = self.config["data"]["modalities"]
        if target_modalities is None:
            target_modalities = [m for m in modalities if m != reference_modality]

        # Load reference image
        ref_path = data_dir / reference_modality.lower() / f"{reference_modality.lower()}.nii.gz"
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")

        fixed_image = sitk.ReadImage(str(ref_path), sitk.sitkFloat32)

        # Register each target modality
        for modality in target_modalities:
            mod_path = data_dir / modality.lower() / f"{modality.lower()}.nii.gz"
            if not mod_path.exists():
                continue

            moving_image = sitk.ReadImage(str(mod_path), sitk.sitkFloat32)

            # Perform registration
            registered, transform = self._register_images(fixed_image, moving_image)

            # Save registered image
            output_path = data_dir / modality.lower() / f"{modality.lower()}_registered.nii.gz"
            sitk.WriteImage(registered, str(output_path))

            # Save transform
            transform_path = data_dir / modality.lower() / f"{modality.lower()}_transform.tfm"
            sitk.WriteTransform(transform, str(transform_path))

            results[modality] = str(output_path)

        return results

    def register_pair(
        self,
        fixed_path: Union[str, Path],
        moving_path: Union[str, Path],
        output_path: Union[str, Path],
        method: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Register a pair of images.

        Args:
            fixed_path: Path to fixed (reference) image
            moving_path: Path to moving image
            output_path: Output path for registered image
            method: Registration method (overrides config)

        Returns:
            Tuple of (registered_image_path, transform_path)
        """
        method = method or self.method

        fixed_image = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
        moving_image = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

        registered, transform = self._register_images(fixed_image, moving_image, method)

        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(registered, str(output_path))

        transform_path = output_path.parent / f"{output_path.stem}_transform.tfm"
        sitk.WriteTransform(transform, str(transform_path))

        return str(output_path), str(transform_path)

    def apply_transform(
        self,
        image_path: Union[str, Path],
        transform_path: Union[str, Path],
        reference_path: Union[str, Path],
        output_path: Union[str, Path],
        interpolation: str = "linear",
    ) -> str:
        """
        Apply transform to an image.

        Args:
            image_path: Path to image to transform
            transform_path: Path to transform file
            reference_path: Path to reference image (for output geometry)
            output_path: Output path
            interpolation: Interpolation method ('linear', 'nearest', 'bspline')

        Returns:
            Path to transformed image
        """
        image = sitk.ReadImage(str(image_path))
        reference = sitk.ReadImage(str(reference_path))
        transform = sitk.ReadTransform(str(transform_path))

        # Set interpolator
        if interpolation == "nearest":
            interpolator = sitk.sitkNearestNeighbor
        elif interpolation == "bspline":
            interpolator = sitk.sitkBSpline
        else:
            interpolator = sitk.sitkLinear

        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)

        transformed = resampler.Execute(image)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(transformed, str(output_path))

        return str(output_path)

    def _register_images(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        method: Optional[str] = None,
    ) -> Tuple[sitk.Image, sitk.Transform]:
        """
        Perform image registration.

        Args:
            fixed_image: Fixed (reference) image
            moving_image: Moving image
            method: Registration method

        Returns:
            Tuple of (registered_image, transform)
        """
        method = method or self.method

        # Initialize transform
        if method == "translation":
            initial_transform = sitk.TranslationTransform(3)
        elif method == "rigid":
            initial_transform = sitk.Euler3DTransform()
        elif method == "affine":
            initial_transform = sitk.AffineTransform(3)
        elif method == "deformable":
            # Use B-spline
            mesh_size = [4] * 3
            initial_transform = sitk.BSplineTransformInitializer(
                fixed_image, mesh_size, order=3
            )
        else:
            raise ValueError(f"Unknown registration method: {method}")

        # Center the transform
        if method != "deformable":
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                initial_transform,
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )

        # Setup registration
        registration = sitk.ImageRegistrationMethod()

        # Set metric
        if self.metric == "mattes_mutual_information":
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        elif self.metric == "correlation":
            registration.SetMetricAsCorrelation()
        elif self.metric == "mean_squares":
            registration.SetMetricAsMeanSquares()
        else:
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.1)

        # Set optimizer
        if method == "deformable":
            registration.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=1e-5,
                numberOfIterations=100,
            )
        else:
            registration.SetOptimizerAsRegularStepGradientDescent(
                learningRate=2.0,
                minStep=1e-4,
                numberOfIterations=200,
                relaxationFactor=0.5,
            )

        registration.SetOptimizerScalesFromPhysicalShift()

        # Multi-resolution
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Set interpolator
        registration.SetInterpolator(sitk.sitkLinear)

        # Set initial transform
        registration.SetInitialTransform(initial_transform, inPlace=False)

        # Execute
        final_transform = registration.Execute(fixed_image, moving_image)

        # Resample moving image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(final_transform)

        registered = resampler.Execute(moving_image)

        return registered, final_transform


class IntensityNormalizer:
    """
    Intensity normalization for multi-modal images.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def normalize(
        self,
        image_path: Union[str, Path],
        modality: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Apply modality-specific normalization.

        Args:
            image_path: Path to image
            modality: Imaging modality
            output_path: Optional output path

        Returns:
            Normalized image array
        """
        nii = nib.load(str(image_path))
        data = nii.get_fdata().astype(np.float32)

        preprocess_config = self.config.get("data", {}).get("preprocessing", {})
        mod_config = preprocess_config.get(modality.lower(), {})

        if modality == "CT":
            # CT windowing
            center = mod_config.get("window_center", 0)
            width = mod_config.get("window_width", 400)
            min_val = center - width / 2
            max_val = center + width / 2
            data = np.clip(data, min_val, max_val)
            data = (data - min_val) / (max_val - min_val)

        elif modality == "PET":
            # PET normalization
            if mod_config.get("normalize", True):
                max_val = np.percentile(data[data > 0], 99) if np.any(data > 0) else 1.0
                data = data / max_val
                data = np.clip(data, 0, 1)

        elif modality in ("MRI", "US"):
            # Z-score normalization
            if mod_config.get("normalize", True):
                mask = data > np.percentile(data, 1)
                mean = data[mask].mean()
                std = data[mask].std() + 1e-8
                data = (data - mean) / std

        if output_path is not None:
            output_nii = nib.Nifti1Image(data, nii.affine, nii.header)
            nib.save(output_nii, str(output_path))

        return data
