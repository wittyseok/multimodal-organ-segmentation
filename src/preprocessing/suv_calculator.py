"""
SUV (Standardized Uptake Value) calculator for PET images.

Supports multiple SUV normalization methods:
- SUV_bw: Body weight
- SUV_bsa: Body surface area
- SUV_lbm: Lean body mass (James and Janmahasatian formulas)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np


class SUVCalculator:
    """
    Calculate SUV (Standardized Uptake Value) from PET images.
    """

    # SUV calculation methods
    SUV_METHODS = ["bw", "bsa", "lbm_james", "lbm_jan"]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SUV calculator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.methods = config.get("analysis", {}).get("suv", {}).get("methods", ["bw"])

    def calculate(
        self,
        pet_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        methods: Optional[list] = None,
    ) -> Dict[str, str]:
        """
        Calculate SUV images.

        Args:
            pet_path: Path to PET NIfTI file or DICOM directory
            output_path: Output directory for SUV images
            metadata: PET metadata (if not provided, will try to load)
            methods: SUV methods to calculate (default: from config)

        Returns:
            Dictionary of {method: output_path}
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        methods = methods or self.methods

        # Load PET image
        pet_nii = nib.load(str(pet_path))
        pet_data = pet_nii.get_fdata().astype(np.float32)

        # Load metadata if not provided
        if metadata is None:
            metadata_path = Path(pet_path).parent / "pet_metadata.npy"
            if metadata_path.exists():
                metadata = np.load(str(metadata_path), allow_pickle=True).item()
            else:
                raise ValueError("PET metadata required for SUV calculation")

        # Extract required values
        pet_info = metadata.get("pet_info", {})

        patient_weight = pet_info.get("patient_weight")  # kg
        patient_size = pet_info.get("patient_size")  # m (height)
        total_dose = pet_info.get("radionuclide_total_dose")  # Bq
        half_life = pet_info.get("radionuclide_half_life")  # seconds
        start_time = pet_info.get("radiopharmaceutical_start_time")
        acquisition_time = pet_info.get("acquisition_time") or pet_info.get("series_time")

        if patient_weight is None or total_dose is None:
            raise ValueError("Patient weight and total dose required for SUV calculation")

        # Calculate decay factor
        decay_factor = self._calculate_decay_factor(
            start_time, acquisition_time, half_life
        )

        # Corrected dose
        corrected_dose = total_dose * decay_factor

        results = {}

        for method in methods:
            if method not in self.SUV_METHODS:
                raise ValueError(f"Unknown SUV method: {method}")

            # Calculate normalization factor based on method
            if method == "bw":
                # Body weight (most common)
                norm_factor = patient_weight * 1000 / corrected_dose  # g / Bq

            elif method == "bsa":
                # Body surface area
                if patient_size is None:
                    raise ValueError("Patient height required for SUV_bsa")
                bsa = self._calculate_bsa(patient_weight, patient_size)
                norm_factor = bsa * 10000 / corrected_dose  # cm² / Bq

            elif method == "lbm_james":
                # Lean body mass (James formula)
                sex = metadata.get("patient_sex", "M")
                if patient_size is None:
                    raise ValueError("Patient height required for SUV_lbm")
                lbm = self._calculate_lbm_james(patient_weight, patient_size * 100, sex)
                norm_factor = lbm * 1000 / corrected_dose

            elif method == "lbm_jan":
                # Lean body mass (Janmahasatian formula)
                sex = metadata.get("patient_sex", "M")
                if patient_size is None:
                    raise ValueError("Patient height required for SUV_lbm")
                lbm = self._calculate_lbm_janmahasatian(patient_weight, patient_size * 100, sex)
                norm_factor = lbm * 1000 / corrected_dose

            # Calculate SUV
            suv = pet_data * norm_factor

            # Save SUV image
            output_file = output_path / f"pet_suv_{method}.nii.gz"
            suv_nii = nib.Nifti1Image(suv, pet_nii.affine, pet_nii.header)
            nib.save(suv_nii, str(output_file))

            results[method] = str(output_file)

        return results

    def _calculate_decay_factor(
        self,
        start_time: Optional[str],
        acquisition_time: Optional[str],
        half_life: Optional[float],
    ) -> float:
        """Calculate radioactive decay factor."""
        if start_time is None or acquisition_time is None or half_life is None:
            return 1.0

        # Parse time strings (HHMMSS.fraction format)
        try:
            start_seconds = self._time_to_seconds(start_time)
            acq_seconds = self._time_to_seconds(acquisition_time)

            # Time difference in seconds
            delta_t = acq_seconds - start_seconds

            # Handle day rollover
            if delta_t < 0:
                delta_t += 24 * 3600

            # Decay factor
            decay_factor = np.exp(-np.log(2) * delta_t / half_life)

            return decay_factor

        except (ValueError, TypeError):
            return 1.0

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert DICOM time string to seconds since midnight."""
        time_str = str(time_str).strip()

        # Handle fractional seconds
        if "." in time_str:
            main_part, fraction = time_str.split(".")
            fraction = float(f"0.{fraction}")
        else:
            main_part = time_str
            fraction = 0.0

        # Pad to 6 digits
        main_part = main_part.ljust(6, "0")

        hours = int(main_part[0:2])
        minutes = int(main_part[2:4])
        seconds = int(main_part[4:6])

        return hours * 3600 + minutes * 60 + seconds + fraction

    def _calculate_bsa(self, weight: float, height: float) -> float:
        """
        Calculate Body Surface Area using Du Bois formula.

        Args:
            weight: Weight in kg
            height: Height in meters

        Returns:
            BSA in m²
        """
        height_cm = height * 100
        return 0.007184 * (weight ** 0.425) * (height_cm ** 0.725)

    def _calculate_lbm_james(self, weight: float, height: float, sex: str) -> float:
        """
        Calculate Lean Body Mass using James formula.

        Args:
            weight: Weight in kg
            height: Height in cm
            sex: 'M' for male, 'F' for female

        Returns:
            LBM in kg
        """
        if sex.upper() == "M":
            lbm = 1.10 * weight - 128 * (weight / height) ** 2
        else:
            lbm = 1.07 * weight - 148 * (weight / height) ** 2

        return max(lbm, weight * 0.5)  # Sanity check

    def _calculate_lbm_janmahasatian(self, weight: float, height: float, sex: str) -> float:
        """
        Calculate Lean Body Mass using Janmahasatian formula.

        Args:
            weight: Weight in kg
            height: Height in cm
            sex: 'M' for male, 'F' for female

        Returns:
            LBM in kg
        """
        # Calculate BMI
        height_m = height / 100
        bmi = weight / (height_m ** 2)

        if sex.upper() == "M":
            lbm = (9270 * weight) / (6680 + 216 * bmi)
        else:
            lbm = (9270 * weight) / (8780 + 244 * bmi)

        return lbm

    def get_suv_stats(
        self,
        suv_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """
        Calculate SUV statistics.

        Args:
            suv_path: Path to SUV NIfTI file
            mask_path: Optional mask file path

        Returns:
            Dictionary of SUV statistics
        """
        suv_nii = nib.load(str(suv_path))
        suv_data = suv_nii.get_fdata()

        if mask_path is not None:
            mask_nii = nib.load(str(mask_path))
            mask = mask_nii.get_fdata() > 0
            suv_masked = suv_data[mask]
        else:
            suv_masked = suv_data[suv_data > 0]

        if len(suv_masked) == 0:
            return {"max": 0, "mean": 0, "std": 0, "median": 0}

        return {
            "max": float(np.max(suv_masked)),
            "mean": float(np.mean(suv_masked)),
            "std": float(np.std(suv_masked)),
            "median": float(np.median(suv_masked)),
            "min": float(np.min(suv_masked)),
            "volume": len(suv_masked),
        }
