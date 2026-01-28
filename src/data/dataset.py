"""
Multi-modal medical image dataset.

Supports multiple imaging modalities: CT, PET, MRI, Ultrasound.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.io import load_nifti


class MultiModalDataset(Dataset):
    """
    Dataset for multi-modal medical image segmentation.

    Supports loading multiple imaging modalities (CT, PET, MRI, US) and
    corresponding segmentation labels.
    """

    SUPPORTED_MODALITIES = ["CT", "PET", "MRI", "US"]

    def __init__(
        self,
        config: Dict[str, Any],
        data_list: pd.DataFrame,
        mode: str = "train",
        transforms: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            config: Configuration dictionary
            data_list: DataFrame with columns for each modality path and label path
                       Expected columns: 'patient_id', 'CT', 'PET', 'MRI', 'US', 'label'
            mode: Dataset mode ('train', 'val', 'test')
            transforms: Transform function to apply
        """
        self.config = config
        self.data_list = data_list
        self.mode = mode
        self.transforms = transforms

        # Get modalities to use from config
        self.modalities = config["data"]["modalities"]

        # Validate modalities
        for mod in self.modalities:
            if mod not in self.SUPPORTED_MODALITIES:
                raise ValueError(f"Unsupported modality: {mod}. Supported: {self.SUPPORTED_MODALITIES}")

        # Get data root
        self.data_root = Path(config["data"]["data_root"])

        # Verify required columns exist
        required_cols = ["patient_id"] + self.modalities
        if mode != "inference":
            required_cols.append("label")

        for col in required_cols:
            if col not in data_list.columns:
                raise ValueError(f"Required column '{col}' not found in data_list")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - 'patient_id': Patient identifier
                - 'image': Stacked multi-modal image tensor [C, H, W, D]
                - 'label': Segmentation label tensor [H, W, D] (if not inference mode)
                - '{modality}': Individual modality tensors (optional)
        """
        row = self.data_list.iloc[idx]
        sample = {"patient_id": row["patient_id"]}

        # Load each modality
        images = []
        for modality in self.modalities:
            path = self.data_root / row[modality]
            image = load_nifti(path, dtype=np.float32)
            images.append(image)
            sample[modality] = image

        # Stack modalities along channel dimension
        sample["image"] = np.stack(images, axis=0)  # [C, H, W, D]

        # Load label if not inference mode
        if self.mode != "inference" and "label" in row:
            label_path = self.data_root / row["label"]
            label = load_nifti(label_path, dtype=np.int64)
            sample["label"] = label

        # Apply transforms
        if self.transforms is not None:
            sample = self.transforms(sample)

        # Convert to tensors
        sample["image"] = torch.from_numpy(sample["image"].copy()).float()
        if "label" in sample:
            sample["label"] = torch.from_numpy(sample["label"].copy()).long()

        return sample


class InferenceDataset(Dataset):
    """
    Dataset for inference on new data without labels.
    """

    def __init__(
        self,
        input_paths: Dict[str, List[Union[str, Path]]],
        config: Dict[str, Any],
        transforms: Optional[Callable] = None,
    ):
        """
        Initialize inference dataset.

        Args:
            input_paths: Dictionary mapping modality to list of file paths
            config: Configuration dictionary
            transforms: Transform function to apply
        """
        self.config = config
        self.transforms = transforms
        self.modalities = config["data"]["modalities"]

        # Validate input paths
        n_samples = None
        for modality, paths in input_paths.items():
            if n_samples is None:
                n_samples = len(paths)
            elif len(paths) != n_samples:
                raise ValueError(f"Inconsistent number of files for {modality}")

        self.input_paths = input_paths
        self.n_samples = n_samples or 0

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {"idx": idx}

        # Load each modality
        images = []
        for modality in self.modalities:
            path = self.input_paths[modality][idx]
            image = load_nifti(path, dtype=np.float32)
            images.append(image)
            sample[f"{modality}_path"] = str(path)

        sample["image"] = np.stack(images, axis=0)

        # Apply transforms
        if self.transforms is not None:
            sample = self.transforms(sample)

        sample["image"] = torch.from_numpy(sample["image"].copy()).float()

        return sample


def get_dataset(
    config: Dict[str, Any],
    split: str = "train",
    transforms: Optional[Callable] = None,
) -> Dataset:
    """
    Factory function to create dataset.

    Args:
        config: Configuration dictionary
        split: Data split ('train', 'val', 'test')
        transforms: Transform function to apply

    Returns:
        Dataset instance
    """
    data_root = Path(config["data"]["data_root"])

    # Load data list
    if split == "train":
        csv_path = data_root / config["data"]["train_csv"]
    elif split == "val":
        csv_path = data_root / config["data"]["val_csv"]
    elif split == "test":
        csv_path = data_root / config["data"]["test_csv"]
    else:
        raise ValueError(f"Invalid split: {split}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Data list not found: {csv_path}")

    data_list = pd.read_csv(csv_path)

    return MultiModalDataset(
        config=config,
        data_list=data_list,
        mode=split,
        transforms=transforms,
    )
