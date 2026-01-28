"""
TMTV (Total Metabolic Tumor Volume) analysis.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd


class TMTVAnalyzer:
    """
    Analyze Total Metabolic Tumor Volume from PET/SUV images.

    TMTV is a prognostic biomarker in lymphoma and other cancers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TMTV analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tmtv_config = config.get("analysis", {}).get("tmtv", {})

        # Default thresholds
        self.absolute_threshold = self.tmtv_config.get("absolute_threshold", 2.5)
        self.percentage_threshold = self.tmtv_config.get("percentage_threshold", 0.4)

    def analyze(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Calculate TMTV using multiple methods.

        Args:
            input_path: Directory with SUV and segmentation files
            output_path: Output directory

        Returns:
            TMTV analysis results
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find files
        suv_file = self._find_suv_file(input_path)
        seg_file = self._find_seg_file(input_path)

        if suv_file is None:
            raise FileNotFoundError("SUV file not found")

        # Load SUV
        suv_nii = nib.load(str(suv_file))
        suv_data = suv_nii.get_fdata()
        voxel_volume = np.prod(suv_nii.header.get_zooms()) / 1000.0  # ml

        # Load segmentation if available
        seg_data = None
        if seg_file is not None:
            seg_nii = nib.load(str(seg_file))
            seg_data = seg_nii.get_fdata().astype(np.int32)

        results = {}

        # Method 1: Absolute threshold (SUV > 2.5)
        tmtv_abs = self._calculate_tmtv_absolute(suv_data, seg_data, voxel_volume)
        results["absolute"] = tmtv_abs

        # Save absolute TMTV mask
        abs_mask = self._create_tmtv_mask(suv_data, seg_data, method="absolute")
        self._save_mask(abs_mask, suv_nii, output_path / "tmtv_absolute.nii.gz")

        # Method 2: Percentage of max (e.g., 40% of max)
        tmtv_pct = self._calculate_tmtv_percentage(suv_data, seg_data, voxel_volume)
        results["percentage"] = tmtv_pct

        # Save percentage TMTV mask
        pct_mask = self._create_tmtv_mask(suv_data, seg_data, method="percentage")
        self._save_mask(pct_mask, suv_nii, output_path / "tmtv_percentage.nii.gz")

        # Method 3: Liver-based threshold (mean liver + 2*std)
        if seg_data is not None:
            tmtv_liver = self._calculate_tmtv_liver_based(suv_data, seg_data, voxel_volume)
            results["liver_based"] = tmtv_liver

            liver_mask = self._create_tmtv_mask(suv_data, seg_data, method="liver")
            self._save_mask(liver_mask, suv_nii, output_path / "tmtv_liver_based.nii.gz")

        # Calculate TLG (Total Lesion Glycolysis)
        results["tlg"] = self._calculate_tlg(suv_data, seg_data, voxel_volume)

        # Save summary
        summary_df = pd.DataFrame([{
            "metric": k,
            **v
        } for k, v in results.items()])
        summary_df.to_csv(output_path / "tmtv_analysis.csv", index=False)
        summary_df.to_excel(output_path / "tmtv_analysis.xlsx", index=False)

        return results

    def _calculate_tmtv_absolute(
        self,
        suv_data: np.ndarray,
        seg_data: Optional[np.ndarray],
        voxel_volume: float,
    ) -> Dict[str, float]:
        """Calculate TMTV using absolute SUV threshold."""
        # Exclude normal organs if segmentation available
        if seg_data is not None:
            tumor_region = (seg_data == 0) | (seg_data > 7)  # Background or unknown
        else:
            tumor_region = np.ones_like(suv_data, dtype=bool)

        mask = (suv_data >= self.absolute_threshold) & tumor_region
        tumor_suv = suv_data[mask]

        if mask.sum() == 0:
            return {
                "volume_ml": 0,
                "suv_max": 0,
                "suv_mean": 0,
                "threshold": self.absolute_threshold,
            }

        return {
            "volume_ml": float(mask.sum() * voxel_volume),
            "suv_max": float(np.max(tumor_suv)),
            "suv_mean": float(np.mean(tumor_suv)),
            "suv_peak": float(self._calculate_suv_peak(suv_data, mask)),
            "num_voxels": int(mask.sum()),
            "threshold": self.absolute_threshold,
        }

    def _calculate_tmtv_percentage(
        self,
        suv_data: np.ndarray,
        seg_data: Optional[np.ndarray],
        voxel_volume: float,
    ) -> Dict[str, float]:
        """Calculate TMTV using percentage of max SUV."""
        if seg_data is not None:
            tumor_region = (seg_data == 0) | (seg_data > 7)
        else:
            tumor_region = np.ones_like(suv_data, dtype=bool)

        max_suv = np.max(suv_data[tumor_region]) if tumor_region.any() else np.max(suv_data)
        threshold = max_suv * self.percentage_threshold

        mask = (suv_data >= threshold) & tumor_region
        tumor_suv = suv_data[mask]

        if mask.sum() == 0:
            return {
                "volume_ml": 0,
                "suv_max": 0,
                "suv_mean": 0,
                "threshold": threshold,
                "percentage": self.percentage_threshold,
            }

        return {
            "volume_ml": float(mask.sum() * voxel_volume),
            "suv_max": float(np.max(tumor_suv)),
            "suv_mean": float(np.mean(tumor_suv)),
            "num_voxels": int(mask.sum()),
            "threshold": float(threshold),
            "percentage": self.percentage_threshold,
        }

    def _calculate_tmtv_liver_based(
        self,
        suv_data: np.ndarray,
        seg_data: np.ndarray,
        voxel_volume: float,
    ) -> Dict[str, float]:
        """Calculate TMTV using liver-based threshold."""
        # Get liver SUV (label 5)
        liver_mask = seg_data == 5
        if liver_mask.sum() == 0:
            return {"volume_ml": 0, "error": "Liver not found in segmentation"}

        liver_suv = suv_data[liver_mask]
        mean_liver = np.mean(liver_suv)
        std_liver = np.std(liver_suv)

        # Threshold: mean + 2*std
        threshold = mean_liver + 2 * std_liver

        # Exclude normal organs
        tumor_region = (seg_data == 0) | (seg_data > 7)
        mask = (suv_data >= threshold) & tumor_region
        tumor_suv = suv_data[mask]

        if mask.sum() == 0:
            return {
                "volume_ml": 0,
                "suv_max": 0,
                "suv_mean": 0,
                "threshold": float(threshold),
                "liver_mean": float(mean_liver),
                "liver_std": float(std_liver),
            }

        return {
            "volume_ml": float(mask.sum() * voxel_volume),
            "suv_max": float(np.max(tumor_suv)),
            "suv_mean": float(np.mean(tumor_suv)),
            "num_voxels": int(mask.sum()),
            "threshold": float(threshold),
            "liver_mean": float(mean_liver),
            "liver_std": float(std_liver),
        }

    def _calculate_tlg(
        self,
        suv_data: np.ndarray,
        seg_data: Optional[np.ndarray],
        voxel_volume: float,
    ) -> Dict[str, float]:
        """Calculate Total Lesion Glycolysis (TMTV × mean SUV)."""
        if seg_data is not None:
            tumor_region = (seg_data == 0) | (seg_data > 7)
        else:
            tumor_region = np.ones_like(suv_data, dtype=bool)

        mask = (suv_data >= self.absolute_threshold) & tumor_region
        tumor_suv = suv_data[mask]

        if mask.sum() == 0:
            return {"tlg": 0, "volume_ml": 0, "mean_suv": 0}

        volume_ml = mask.sum() * voxel_volume
        mean_suv = np.mean(tumor_suv)
        tlg = volume_ml * mean_suv

        return {
            "tlg": float(tlg),
            "volume_ml": float(volume_ml),
            "mean_suv": float(mean_suv),
        }

    def _calculate_suv_peak(
        self,
        suv_data: np.ndarray,
        mask: np.ndarray,
        sphere_radius_mm: float = 6.0,
    ) -> float:
        """Calculate SUV peak (mean in 1ml sphere around max)."""
        # Find location of max SUV
        masked_suv = np.where(mask, suv_data, -np.inf)
        max_idx = np.unravel_index(np.argmax(masked_suv), suv_data.shape)

        # For simplicity, use local neighborhood mean
        # In practice, should use 1ml sphere
        neighborhood_size = 3
        slices = tuple(
            slice(max(0, idx - neighborhood_size), min(s, idx + neighborhood_size + 1))
            for idx, s in zip(max_idx, suv_data.shape)
        )
        neighborhood = suv_data[slices]

        return float(np.mean(neighborhood))

    def _create_tmtv_mask(
        self,
        suv_data: np.ndarray,
        seg_data: Optional[np.ndarray],
        method: str = "absolute",
    ) -> np.ndarray:
        """Create binary TMTV mask."""
        if seg_data is not None:
            tumor_region = (seg_data == 0) | (seg_data > 7)
        else:
            tumor_region = np.ones_like(suv_data, dtype=bool)

        if method == "absolute":
            threshold = self.absolute_threshold
        elif method == "percentage":
            max_suv = np.max(suv_data[tumor_region]) if tumor_region.any() else np.max(suv_data)
            threshold = max_suv * self.percentage_threshold
        elif method == "liver" and seg_data is not None:
            liver_mask = seg_data == 5
            if liver_mask.sum() > 0:
                liver_suv = suv_data[liver_mask]
                threshold = np.mean(liver_suv) + 2 * np.std(liver_suv)
            else:
                threshold = self.absolute_threshold
        else:
            threshold = self.absolute_threshold

        return ((suv_data >= threshold) & tumor_region).astype(np.uint8)

    def _save_mask(
        self,
        mask: np.ndarray,
        reference_nii: nib.Nifti1Image,
        output_path: Path,
    ) -> None:
        """Save mask as NIfTI."""
        mask_nii = nib.Nifti1Image(mask, reference_nii.affine, reference_nii.header)
        nib.save(mask_nii, str(output_path))

    def _find_suv_file(self, directory: Path) -> Optional[Path]:
        """Find SUV file in directory."""
        patterns = ["*suv*.nii*", "*SUV*.nii*", "*pet*.nii*", "*PET*.nii*"]
        for pattern in patterns:
            matches = list(directory.rglob(pattern))
            if matches:
                return matches[0]
        return None

    def _find_seg_file(self, directory: Path) -> Optional[Path]:
        """Find segmentation file in directory."""
        patterns = ["*seg*.nii*", "*label*.nii*", "*pred*.nii*", "*mask*.nii*"]
        for pattern in patterns:
            matches = list(directory.rglob(pattern))
            if matches:
                return matches[0]
        return None
