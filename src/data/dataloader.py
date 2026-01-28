"""
DataLoader utilities for multi-modal medical imaging.
"""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import get_dataset
from src.data.transforms import get_transforms


def get_dataloader(
    config: Dict[str, Any],
    split: str = "train",
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """
    Create DataLoader with appropriate settings.

    Args:
        config: Configuration dictionary
        split: Data split ('train', 'val', 'test')
        shuffle: Override shuffle setting
        drop_last: Override drop_last setting

    Returns:
        DataLoader instance
    """
    # Get transforms
    transforms = get_transforms(config, mode=split)

    # Get dataset
    dataset = get_dataset(config, split=split, transforms=transforms)

    # DataLoader settings
    batch_size = config["training"]["batch_size"]
    num_workers = config["hardware"]["num_workers"]
    pin_memory = config["hardware"]["pin_memory"]

    # Mode-specific defaults
    if shuffle is None:
        shuffle = (split == "train")
    if drop_last is None:
        drop_last = (split == "train")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )

    return dataloader


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-sized images.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    # Find common keys
    keys = batch[0].keys()

    result = {}
    for key in keys:
        values = [sample[key] for sample in batch]

        if isinstance(values[0], torch.Tensor):
            # Stack tensors
            try:
                result[key] = torch.stack(values)
            except RuntimeError:
                # Handle variable sizes by padding
                result[key] = pad_tensors(values)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            # Keep as list for non-tensor data
            result[key] = values

    return result


def pad_tensors(tensors: list, pad_value: float = 0) -> torch.Tensor:
    """
    Pad list of tensors to same size and stack.

    Args:
        tensors: List of tensors with potentially different sizes
        pad_value: Value to use for padding

    Returns:
        Stacked padded tensors
    """
    # Find max dimensions
    max_dims = []
    ndim = tensors[0].ndim

    for d in range(ndim):
        max_size = max(t.shape[d] for t in tensors)
        max_dims.append(max_size)

    # Pad each tensor
    padded = []
    for t in tensors:
        pad_sizes = []
        for d in range(ndim - 1, -1, -1):
            diff = max_dims[d] - t.shape[d]
            pad_sizes.extend([0, diff])

        padded_t = torch.nn.functional.pad(t, pad_sizes, value=pad_value)
        padded.append(padded_t)

    return torch.stack(padded)
