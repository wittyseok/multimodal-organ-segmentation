"""
Histogram analysis for SUV distributions.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


class HistogramAnalyzer:
    """
    Generate histograms for SUV distributions in organs.
    """

    ORGAN_LABELS = {
        1: "bladder",
        2: "kidney_right",
        3: "kidney_left",
        4: "heart",
        5: "liver",
        6: "spleen",
        7: "brain",
    }

    ORGAN_COLORS = {
        1: "#FF6B6B",  # Red
        2: "#4ECDC4",  # Teal
        3: "#45B7D1",  # Blue
        4: "#F7DC6F",  # Yellow
        5: "#BB8FCE",  # Purple
        6: "#58D68D",  # Green
        7: "#F8B500",  # Orange
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize histogram analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.hist_config = config.get("analysis", {}).get("histogram", {})
        self.bins = self.hist_config.get("bins", 100)

    def analyze(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Generate histograms for all organs.

        Args:
            input_path: Directory with SUV and segmentation files
            output_path: Output directory for plots

        Returns:
            Histogram statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find files
        suv_file = self._find_file(input_path, ["*suv*.nii*", "*SUV*.nii*"])
        seg_file = self._find_file(input_path, ["*seg*.nii*", "*label*.nii*", "*pred*.nii*"])

        if suv_file is None or seg_file is None:
            raise FileNotFoundError("SUV or segmentation file not found")

        # Load data
        suv_nii = nib.load(str(suv_file))
        suv_data = suv_nii.get_fdata()

        seg_nii = nib.load(str(seg_file))
        seg_data = seg_nii.get_fdata().astype(np.int32)

        results = {}

        # Generate individual organ histograms
        results["per_organ"] = self._generate_organ_histograms(
            suv_data, seg_data, output_path
        )

        # Generate combined histogram
        self._generate_combined_histogram(suv_data, seg_data, output_path)

        # Generate threshold-volume curves
        results["threshold_curves"] = self._generate_threshold_curves(
            suv_data, seg_data, output_path
        )

        # Generate cumulative distribution
        self._generate_cumulative_distribution(suv_data, seg_data, output_path)

        return results

    def _generate_organ_histograms(
        self,
        suv_data: np.ndarray,
        seg_data: np.ndarray,
        output_path: Path,
    ) -> Dict[str, Dict[str, float]]:
        """Generate histogram for each organ."""
        results = {}

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for idx, (label_id, organ_name) in enumerate(self.ORGAN_LABELS.items()):
            mask = seg_data == label_id
            organ_suv = suv_data[mask]

            if len(organ_suv) == 0:
                continue

            ax = axes[idx]
            color = self.ORGAN_COLORS.get(label_id, "#333333")

            # Plot histogram
            counts, bin_edges, _ = ax.hist(
                organ_suv,
                bins=self.bins,
                color=color,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            # Add statistics
            stats = {
                "mean": float(np.mean(organ_suv)),
                "std": float(np.std(organ_suv)),
                "median": float(np.median(organ_suv)),
                "max": float(np.max(organ_suv)),
                "min": float(np.min(organ_suv)),
            }
            results[organ_name] = stats

            # Add text with statistics
            ax.axvline(stats["mean"], color="red", linestyle="--", linewidth=1, label=f'Mean: {stats["mean"]:.2f}')
            ax.axvline(stats["median"], color="blue", linestyle="--", linewidth=1, label=f'Median: {stats["median"]:.2f}')

            ax.set_title(f'{organ_name.replace("_", " ").title()}')
            ax.set_xlabel("SUV")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

        # Hide empty subplot
        if len(self.ORGAN_LABELS) < len(axes):
            axes[-1].axis("off")

        plt.tight_layout()
        plt.savefig(output_path / "organ_histograms.png", dpi=150, bbox_inches="tight")
        plt.close()

        return results

    def _generate_combined_histogram(
        self,
        suv_data: np.ndarray,
        seg_data: np.ndarray,
        output_path: Path,
    ) -> None:
        """Generate combined histogram for all organs."""
        fig, ax = plt.subplots(figsize=(12, 6))

        for label_id, organ_name in self.ORGAN_LABELS.items():
            mask = seg_data == label_id
            organ_suv = suv_data[mask]

            if len(organ_suv) == 0:
                continue

            color = self.ORGAN_COLORS.get(label_id, "#333333")

            # Normalize histogram for comparison
            ax.hist(
                organ_suv,
                bins=self.bins,
                color=color,
                alpha=0.5,
                label=organ_name.replace("_", " ").title(),
                density=True,
            )

        ax.set_xlabel("SUV", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("SUV Distribution by Organ", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "combined_histogram.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _generate_threshold_curves(
        self,
        suv_data: np.ndarray,
        seg_data: np.ndarray,
        output_path: Path,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Generate threshold vs volume curves."""
        results = {}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Threshold percentage vs volume percentage
        ax1 = axes[0]
        for label_id, organ_name in self.ORGAN_LABELS.items():
            mask = seg_data == label_id
            organ_suv = suv_data[mask]

            if len(organ_suv) == 0:
                continue

            max_suv = np.max(organ_suv)
            total_volume = len(organ_suv)

            thresholds = np.linspace(0, 1, 50)
            volumes = []

            for thresh_pct in thresholds:
                thresh_val = max_suv * thresh_pct
                vol = (organ_suv >= thresh_val).sum() / total_volume * 100
                volumes.append(vol)

            color = self.ORGAN_COLORS.get(label_id, "#333333")
            ax1.plot(
                thresholds * 100,
                volumes,
                color=color,
                label=organ_name.replace("_", " ").title(),
                linewidth=2,
            )

            results[organ_name] = list(zip(thresholds.tolist(), volumes))

        ax1.set_xlabel("Threshold (% of max SUV)", fontsize=12)
        ax1.set_ylabel("Volume above threshold (%)", fontsize=12)
        ax1.set_title("Volume vs Threshold (Relative)", fontsize=14)
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Absolute threshold vs volume percentage
        ax2 = axes[1]
        for label_id, organ_name in self.ORGAN_LABELS.items():
            mask = seg_data == label_id
            organ_suv = suv_data[mask]

            if len(organ_suv) == 0:
                continue

            total_volume = len(organ_suv)
            thresholds = np.linspace(0, 20, 50)  # SUV 0-20
            volumes = []

            for thresh_val in thresholds:
                vol = (organ_suv >= thresh_val).sum() / total_volume * 100
                volumes.append(vol)

            color = self.ORGAN_COLORS.get(label_id, "#333333")
            ax2.plot(
                thresholds,
                volumes,
                color=color,
                label=organ_name.replace("_", " ").title(),
                linewidth=2,
            )

        ax2.set_xlabel("SUV Threshold", fontsize=12)
        ax2.set_ylabel("Volume above threshold (%)", fontsize=12)
        ax2.set_title("Volume vs Threshold (Absolute)", fontsize=14)
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "threshold_curves.png", dpi=150, bbox_inches="tight")
        plt.close()

        return results

    def _generate_cumulative_distribution(
        self,
        suv_data: np.ndarray,
        seg_data: np.ndarray,
        output_path: Path,
    ) -> None:
        """Generate cumulative distribution function plots."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for label_id, organ_name in self.ORGAN_LABELS.items():
            mask = seg_data == label_id
            organ_suv = suv_data[mask]

            if len(organ_suv) == 0:
                continue

            # Sort and compute CDF
            sorted_suv = np.sort(organ_suv)
            cdf = np.arange(1, len(sorted_suv) + 1) / len(sorted_suv)

            color = self.ORGAN_COLORS.get(label_id, "#333333")
            ax.plot(
                sorted_suv,
                cdf,
                color=color,
                label=organ_name.replace("_", " ").title(),
                linewidth=2,
            )

        ax.set_xlabel("SUV", fontsize=12)
        ax.set_ylabel("Cumulative Probability", fontsize=12)
        ax.set_title("Cumulative Distribution Function by Organ", fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_path / "cumulative_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _find_file(
        self,
        directory: Path,
        patterns: List[str],
    ) -> Optional[Path]:
        """Find file matching patterns."""
        for pattern in patterns:
            matches = list(directory.rglob(pattern))
            if matches:
                return matches[0]
        return None
