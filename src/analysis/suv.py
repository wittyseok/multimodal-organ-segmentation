"""
SUV analysis module for PET imaging.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd


class SUVAnalyzer:
    """
    Analyze SUV values within segmented organs.
    """

    # Organ labels (matching segmentation output)
    ORGAN_LABELS = {
        1: "bladder",
        2: "kidney_right",
        3: "kidney_left",
        4: "heart",
        5: "liver",
        6: "spleen",
        7: "brain",
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SUV analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.analysis_config = config.get("analysis", {}).get("suv", {})

    def analyze(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Analyze SUV values in segmented regions.

        Args:
            input_path: Path to directory with SUV and segmentation files
            output_path: Output directory for results

        Returns:
            Analysis results
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find SUV and segmentation files
        suv_file = self._find_file(input_path, patterns=["*suv*.nii*", "*SUV*.nii*"])
        seg_file = self._find_file(input_path, patterns=["*seg*.nii*", "*label*.nii*", "*pred*.nii*"])

        if suv_file is None or seg_file is None:
            raise FileNotFoundError("SUV or segmentation file not found")

        # Load data
        suv_nii = nib.load(str(suv_file))
        suv_data = suv_nii.get_fdata()

        seg_nii = nib.load(str(seg_file))
        seg_data = seg_nii.get_fdata().astype(np.int32)

        # Get voxel volume in ml
        voxel_volume = np.prod(suv_nii.header.get_zooms()) / 1000.0

        # Analyze each organ
        results = []
        for label_id, organ_name in self.ORGAN_LABELS.items():
            mask = seg_data == label_id

            if mask.sum() == 0:
                continue

            organ_suv = suv_data[mask]

            # Calculate statistics
            stats = {
                "organ": organ_name,
                "label_id": label_id,
                "suv_max": float(np.max(organ_suv)),
                "suv_mean": float(np.mean(organ_suv)),
                "suv_std": float(np.std(organ_suv)),
                "suv_median": float(np.median(organ_suv)),
                "suv_min": float(np.min(organ_suv)),
                "volume_ml": float(mask.sum() * voxel_volume),
                "volume_voxels": int(mask.sum()),
            }

            # High uptake volumes (different thresholds)
            max_suv = stats["suv_max"]
            stats["suv_40_volume"] = float((organ_suv >= max_suv * 0.4).sum() * voxel_volume)
            stats["suv_50_volume"] = float((organ_suv >= max_suv * 0.5).sum() * voxel_volume)
            stats["suv_60_volume"] = float((organ_suv >= max_suv * 0.6).sum() * voxel_volume)

            results.append(stats)

        # Create DataFrame and save
        df = pd.DataFrame(results)

        # Save results
        df.to_csv(output_path / "suv_analysis.csv", index=False)
        df.to_excel(output_path / "suv_analysis.xlsx", index=False)

        return {
            "organs": results,
            "summary": {
                "num_organs_analyzed": len(results),
                "total_volume_ml": sum(r["volume_ml"] for r in results),
            }
        }

    def analyze_tumor(
        self,
        suv_path: Union[str, Path],
        seg_path: Union[str, Path],
        threshold: float = 2.5,
    ) -> Dict[str, Any]:
        """
        Analyze tumor regions based on SUV threshold.

        Args:
            suv_path: Path to SUV NIfTI file
            seg_path: Path to segmentation file
            threshold: SUV threshold for tumor detection

        Returns:
            Tumor analysis results
        """
        suv_nii = nib.load(str(suv_path))
        suv_data = suv_nii.get_fdata()

        seg_nii = nib.load(str(seg_path))
        seg_data = seg_nii.get_fdata().astype(np.int32)

        voxel_volume = np.prod(suv_nii.header.get_zooms()) / 1000.0

        # Create mask excluding normal organs
        organ_mask = seg_data > 0
        tumor_candidates = (suv_data >= threshold) & ~organ_mask

        if tumor_candidates.sum() == 0:
            return {
                "num_lesions": 0,
                "total_volume_ml": 0,
                "max_suv": 0,
            }

        tumor_suv = suv_data[tumor_candidates]

        return {
            "num_voxels": int(tumor_candidates.sum()),
            "volume_ml": float(tumor_candidates.sum() * voxel_volume),
            "suv_max": float(np.max(tumor_suv)),
            "suv_mean": float(np.mean(tumor_suv)),
            "suv_median": float(np.median(tumor_suv)),
            "threshold_used": threshold,
        }

    def _find_file(
        self,
        directory: Path,
        patterns: List[str],
    ) -> Optional[Path]:
        """Find file matching patterns."""
        for pattern in patterns:
            matches = list(directory.glob(pattern))
            if matches:
                return matches[0]

            # Check subdirectories
            matches = list(directory.rglob(pattern))
            if matches:
                return matches[0]

        return None
