"""
I/O utilities for loading and saving data.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import nibabel as nib
import numpy as np
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove internal keys
    config_to_save = {k: v for k, v in config.items() if not k.startswith("_")}

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)


def load_nifti(
    file_path: Union[str, Path],
    return_header: bool = False,
    dtype: Optional[np.dtype] = None,
) -> Union[np.ndarray, tuple]:
    """
    Load NIfTI file.

    Args:
        file_path: Path to NIfTI file (.nii or .nii.gz)
        return_header: Whether to return header and affine
        dtype: Target data type (optional)

    Returns:
        Image data array, optionally with header and affine
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")

    nii = nib.load(str(file_path))
    data = nii.get_fdata()

    if dtype is not None:
        data = data.astype(dtype)

    if return_header:
        return data, nii.header, nii.affine
    return data


def save_nifti(
    data: np.ndarray,
    save_path: Union[str, Path],
    affine: Optional[np.ndarray] = None,
    header: Optional[nib.Nifti1Header] = None,
) -> None:
    """
    Save array as NIfTI file.

    Args:
        data: Image data array
        save_path: Path to save NIfTI file
        affine: Affine transformation matrix (default: identity)
        header: NIfTI header (optional)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if affine is None:
        affine = np.eye(4)

    if header is not None:
        nii = nib.Nifti1Image(data, affine, header)
    else:
        nii = nib.Nifti1Image(data, affine)

    nib.save(nii, str(save_path))


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary from JSON
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def save_json(data: Dict[str, Any], save_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save dictionary as JSON file.

    Args:
        data: Dictionary to save
        save_path: Path to save JSON file
        indent: Indentation level
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_list(
    directory: Union[str, Path],
    extensions: Optional[list] = None,
    recursive: bool = False,
) -> list:
    """
    Get list of files in directory.

    Args:
        directory: Directory path
        extensions: List of file extensions to filter (e.g., ['.nii', '.nii.gz'])
        recursive: Whether to search recursively

    Returns:
        List of file paths
    """
    directory = Path(directory)

    if recursive:
        files = list(directory.rglob("*"))
    else:
        files = list(directory.iterdir())

    # Filter by extension
    if extensions:
        extensions = [ext.lower() for ext in extensions]
        files = [
            f for f in files
            if f.is_file() and any(str(f).lower().endswith(ext) for ext in extensions)
        ]

    return sorted(files)
