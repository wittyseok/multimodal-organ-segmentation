"""
DICOM to NIfTI conversion for multiple imaging modalities.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

try:
    import pydicom as dcm
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


class DicomConverter:
    """
    Convert DICOM files to NIfTI format.

    Supports CT, PET, MRI, and Ultrasound modalities.
    """

    SUPPORTED_MODALITIES = ["CT", "PET", "MRI", "US"]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize converter.

        Args:
            config: Configuration dictionary
        """
        if not HAS_PYDICOM:
            raise ImportError("pydicom is required for DICOM conversion")

        self.config = config

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        modality: str = "CT",
    ) -> str:
        """
        Convert DICOM directory to NIfTI file.

        Args:
            input_path: Path to directory containing DICOM files
            output_path: Output directory for NIfTI file
            modality: Imaging modality (CT, PET, MRI, US)

        Returns:
            Path to output NIfTI file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if modality not in self.SUPPORTED_MODALITIES:
            raise ValueError(f"Unsupported modality: {modality}")

        # Load DICOM files
        dicom_files = self._load_dicom_series(input_path)

        if len(dicom_files) == 0:
            raise ValueError(f"No DICOM files found in {input_path}")

        # Convert based on modality
        if modality == "CT":
            volume, affine, metadata = self._convert_ct(dicom_files)
        elif modality == "PET":
            volume, affine, metadata = self._convert_pet(dicom_files)
        elif modality == "MRI":
            volume, affine, metadata = self._convert_mri(dicom_files)
        elif modality == "US":
            volume, affine, metadata = self._convert_ultrasound(dicom_files)

        # Save NIfTI
        output_file = output_path / f"{modality.lower()}.nii.gz"
        nii = nib.Nifti1Image(volume, affine)
        nib.save(nii, str(output_file))

        # Save metadata
        metadata_file = output_path / f"{modality.lower()}_metadata.npy"
        np.save(str(metadata_file), metadata)

        return str(output_file)

    def _load_dicom_series(self, dicom_dir: Path) -> List[dcm.Dataset]:
        """Load and sort DICOM files from directory."""
        dicom_files = []

        for f in dicom_dir.iterdir():
            if f.suffix.lower() in [".dcm", ""] or f.name.isdigit():
                try:
                    ds = dcm.dcmread(str(f))
                    dicom_files.append(ds)
                except Exception:
                    continue

        # Sort by instance number or slice location
        try:
            dicom_files.sort(key=lambda x: float(x.InstanceNumber))
        except AttributeError:
            try:
                dicom_files.sort(key=lambda x: float(x.SliceLocation))
            except AttributeError:
                try:
                    dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
                except (AttributeError, IndexError):
                    pass

        return dicom_files

    def _convert_ct(
        self, dicom_files: List[dcm.Dataset]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Convert CT DICOM to volume."""
        # Stack slices
        slices = []
        for ds in dicom_files:
            pixel_array = ds.pixel_array.astype(np.float32)

            # Apply rescale
            slope = getattr(ds, "RescaleSlope", 1)
            intercept = getattr(ds, "RescaleIntercept", 0)
            pixel_array = pixel_array * slope + intercept

            slices.append(pixel_array)

        volume = np.stack(slices, axis=-1)

        # Get spacing and create affine
        affine, spacing = self._get_affine(dicom_files[0])

        # Extract metadata
        metadata = self._extract_metadata(dicom_files[0])
        metadata["spacing"] = spacing

        return volume, affine, metadata

    def _convert_pet(
        self, dicom_files: List[dcm.Dataset]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Convert PET DICOM to volume."""
        slices = []
        for ds in dicom_files:
            pixel_array = ds.pixel_array.astype(np.float32)

            # Apply rescale
            slope = getattr(ds, "RescaleSlope", 1)
            intercept = getattr(ds, "RescaleIntercept", 0)
            pixel_array = pixel_array * slope + intercept

            slices.append(pixel_array)

        volume = np.stack(slices, axis=-1)

        affine, spacing = self._get_affine(dicom_files[0])

        metadata = self._extract_metadata(dicom_files[0])
        metadata["spacing"] = spacing

        # Extract PET-specific metadata for SUV calculation
        ds = dicom_files[0]
        metadata["pet_info"] = {
            "patient_weight": getattr(ds, "PatientWeight", None),
            "patient_size": getattr(ds, "PatientSize", None),
            "series_time": getattr(ds, "SeriesTime", None),
            "acquisition_time": getattr(ds, "AcquisitionTime", None),
        }

        # Radiopharmaceutical info
        if hasattr(ds, "RadiopharmaceuticalInformationSequence"):
            radio_info = ds.RadiopharmaceuticalInformationSequence[0]
            metadata["pet_info"].update({
                "radionuclide_total_dose": getattr(radio_info, "RadionuclideTotalDose", None),
                "radionuclide_half_life": getattr(radio_info, "RadionuclideHalfLife", None),
                "radiopharmaceutical_start_time": getattr(
                    radio_info, "RadiopharmaceuticalStartTime", None
                ),
            })

        return volume, affine, metadata

    def _convert_mri(
        self, dicom_files: List[dcm.Dataset]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Convert MRI DICOM to volume."""
        slices = []
        for ds in dicom_files:
            pixel_array = ds.pixel_array.astype(np.float32)
            slices.append(pixel_array)

        volume = np.stack(slices, axis=-1)

        affine, spacing = self._get_affine(dicom_files[0])

        metadata = self._extract_metadata(dicom_files[0])
        metadata["spacing"] = spacing

        # MRI-specific metadata
        ds = dicom_files[0]
        metadata["mri_info"] = {
            "sequence_name": getattr(ds, "SequenceName", None),
            "repetition_time": getattr(ds, "RepetitionTime", None),
            "echo_time": getattr(ds, "EchoTime", None),
            "magnetic_field_strength": getattr(ds, "MagneticFieldStrength", None),
        }

        return volume, affine, metadata

    def _convert_ultrasound(
        self, dicom_files: List[dcm.Dataset]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Convert Ultrasound DICOM to volume."""
        slices = []
        for ds in dicom_files:
            pixel_array = ds.pixel_array.astype(np.float32)

            # Handle RGB ultrasound images
            if len(pixel_array.shape) == 3 and pixel_array.shape[-1] == 3:
                pixel_array = np.mean(pixel_array, axis=-1)

            slices.append(pixel_array)

        if len(slices) == 1:
            # 2D ultrasound - add depth dimension
            volume = slices[0][..., np.newaxis]
        else:
            volume = np.stack(slices, axis=-1)

        affine, spacing = self._get_affine(dicom_files[0])

        metadata = self._extract_metadata(dicom_files[0])
        metadata["spacing"] = spacing

        return volume, affine, metadata

    def _get_affine(self, ds: dcm.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate affine transformation matrix from DICOM header."""
        # Get pixel spacing
        pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
        slice_thickness = getattr(ds, "SliceThickness", 1.0)
        spacing = np.array([float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)])

        # Get image position
        image_position = getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])
        image_position = np.array([float(p) for p in image_position])

        # Get image orientation
        image_orientation = getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
        row_cosines = np.array([float(o) for o in image_orientation[:3]])
        col_cosines = np.array([float(o) for o in image_orientation[3:]])

        # Calculate slice direction
        slice_cosines = np.cross(row_cosines, col_cosines)

        # Build affine matrix
        affine = np.eye(4)
        affine[:3, 0] = row_cosines * spacing[0]
        affine[:3, 1] = col_cosines * spacing[1]
        affine[:3, 2] = slice_cosines * spacing[2]
        affine[:3, 3] = image_position

        return affine, spacing

    def _extract_metadata(self, ds: dcm.Dataset) -> Dict[str, Any]:
        """Extract common metadata from DICOM."""
        metadata = {
            "patient_id": getattr(ds, "PatientID", "Unknown"),
            "patient_name": str(getattr(ds, "PatientName", "Unknown")),
            "patient_sex": getattr(ds, "PatientSex", "Unknown"),
            "patient_age": getattr(ds, "PatientAge", "Unknown"),
            "study_date": getattr(ds, "StudyDate", "Unknown"),
            "modality": getattr(ds, "Modality", "Unknown"),
            "manufacturer": getattr(ds, "Manufacturer", "Unknown"),
            "institution": getattr(ds, "InstitutionName", "Unknown"),
            "rows": getattr(ds, "Rows", 0),
            "columns": getattr(ds, "Columns", 0),
        }

        return metadata
