"""
Data preprocessing: convert all raw datasets into nnUNet-compatible format.

Handles:
  - READ-PE (DICOM → NIfTI, with pixel-level masks)
  - FUMPE (PAT directories + .mat GT → NIfTI)
  - CAD-PE (NRRD → NIfTI)
  - RSPECT (DICOM slices, image-level labels)
  - Augmented_RSPECT (DICOM slices + box annotations)

Output: nnUNet raw data structure under nnUNet_raw/DatasetXXX_Name/
"""

import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import nrrd
from scipy.io import loadmat
from scipy.ndimage import zoom


# ============================================================================
# Helper functions
# ============================================================================


def dcm_series_to_nifti(dcm_dir: str | Path, output_path: str | Path) -> None:
    """Convert a DICOM series directory to a single NIfTI file."""
    dcm_dir = Path(dcm_dir)
    slices = []
    for f in sorted(dcm_dir.iterdir()):
        if f.suffix == ".dcm":
            ds = pydicom.dcmread(str(f))
            slices.append(ds)

    if not slices:
        raise ValueError(f"No DICOM files found in {dcm_dir}")

    # Sort by ImagePositionPatient or InstanceNumber
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except (AttributeError, TypeError):
        slices.sort(key=lambda s: int(s.InstanceNumber))

    # Build 3D volume
    pixel_arrays = [s.pixel_array.astype(np.float32) for s in slices]
    volume = np.stack(pixel_arrays, axis=0)  # (D, H, W)

    # Apply rescale slope/intercept for HU values
    ds0 = slices[0]
    slope = float(getattr(ds0, "RescaleSlope", 1))
    intercept = float(getattr(ds0, "RescaleIntercept", 0))
    volume = volume * slope + intercept

    # Build affine from DICOM metadata
    try:
        ps = [float(x) for x in ds0.PixelSpacing]
        if len(slices) > 1:
            dz = abs(
                float(slices[1].ImagePositionPatient[2])
                - float(slices[0].ImagePositionPatient[2])
            )
        else:
            dz = float(getattr(ds0, "SliceThickness", 1.0))
        affine = np.diag([ps[0], ps[1], dz, 1.0])
    except (AttributeError, TypeError):
        affine = np.eye(4)

    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, str(output_path))


def nrrd_to_nifti(nrrd_path: str | Path, output_path: str | Path) -> None:
    """Convert NRRD file to NIfTI."""
    data, header = nrrd.read(str(nrrd_path))
    # Build affine from NRRD space directions
    try:
        directions = np.array(header["space directions"])
        origin = np.array(header["space origin"])
        affine = np.eye(4)
        affine[:3, :3] = directions
        affine[:3, 3] = origin
    except (KeyError, ValueError):
        affine = np.eye(4)

    nii = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(nii, str(output_path))


def mat_mask_to_nifti(
    mat_path: str | Path, output_path: str | Path, key: str = "mask"
) -> None:
    """Convert MATLAB .mat ground truth to NIfTI."""
    mat = loadmat(str(mat_path))
    # Try common key names
    data = None
    for k in [key, "groundTruth", "GroundTruth", "gt", "label", "seg"]:
        if k in mat:
            data = mat[k]
            break
    if data is None:
        # Try the first non-metadata key
        for k, v in mat.items():
            if not k.startswith("_") and isinstance(v, np.ndarray):
                data = v
                break
    if data is None:
        raise ValueError(f"No array data found in {mat_path}")

    data = data.astype(np.float32)
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, str(output_path))


# ============================================================================
# Dataset converters
# ============================================================================


def convert_read_pe(
    data_root: str | Path,
    output_dir: str | Path,
    dataset_id: int = 100,
) -> None:
    """
    Convert the READ-PE dataset to nnUNet format.

    Input structure:
        data/READ/
        ├── CT scans (DICOM files)/
        │   ├── GE (DICOM files)/01GE/, 02GE/, ...
        │   └── TOSHIBA (DICOM files)/01TS/, 02TS/, ...
        └── Ground truth (pixel level segmentation - NiFTI files)/
            ├── GE (NiFTI files)/01GE.nii.gz, ...
            └── TOSHIBA (NiFTI files)/01TS.nii.gz, ...

    Output: Dataset100_READPE/imagesTr/, labelsTr/, dataset.json
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir) / f"Dataset{dataset_id:03d}_READPE"
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    ct_root = data_root / "CT scans (DICOM files)"
    gt_root = data_root / "Ground truth (pixel level segmentation - NiFTI files)"

    case_list = []

    for vendor_ct, vendor_gt, prefix in [
        ("GE (DICOM files)", "GE (NiFTI files)", "GE"),
        ("TOSHIBA (DICOM files)", "TOSHIBA (NiFTI files)", "TS"),
    ]:
        ct_vendor_dir = ct_root / vendor_ct
        gt_vendor_dir = gt_root / vendor_gt

        for case_dir in sorted(ct_vendor_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_name = case_dir.name  # e.g., "01GE"
            case_id = f"READPE_{case_name}"

            # Convert DICOM to NIfTI
            img_out = images_dir / f"{case_id}_0000.nii.gz"
            if not img_out.exists():
                print(f"  Converting CT: {case_name}")
                dcm_series_to_nifti(case_dir, img_out)

            # Copy label
            gt_file = gt_vendor_dir / f"{case_name}.nii.gz"
            lbl_out = labels_dir / f"{case_id}.nii.gz"
            if gt_file.exists() and not lbl_out.exists():
                print(f"  Copying label: {case_name}")
                shutil.copy2(gt_file, lbl_out)

            case_list.append(case_id)

    # Write dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "PE": 1},
        "numTraining": len(case_list),
        "file_ending": ".nii.gz",
        "name": "READPE",
        "description": "READ Pulmonary Embolism dataset - pixel-level annotations",
        "reference": "https://doi.org/10.1148/radiol.2019191832",
    }
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"READ-PE: {len(case_list)} cases converted to {output_dir}")
    return case_list


def convert_fumpe(
    data_root: str | Path,
    output_dir: str | Path,
    dataset_id: int = 101,
) -> None:
    """
    Convert FUMPE dataset to nnUNet format.

    Input:
        data/FUMPE/
        ├── CT_scans/PAT001/, PAT002/, ...  (DICOM series per patient)
        └── GroundTruth/PAT001.mat, ...

    Output: Dataset101_FUMPE/
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir) / f"Dataset{dataset_id:03d}_FUMPE"
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    ct_root = data_root / "CT_scans"
    gt_root = data_root / "GroundTruth"

    case_list = []
    for pat_dir in sorted(ct_root.iterdir()):
        if not pat_dir.is_dir():
            continue
        case_name = pat_dir.name  # PAT001
        case_id = f"FUMPE_{case_name}"

        img_out = images_dir / f"{case_id}_0000.nii.gz"
        if not img_out.exists():
            # Check if it contains DICOM files directly or in subdirs
            dcm_files = list(pat_dir.rglob("*.dcm")) + list(pat_dir.rglob("*.DCM"))
            if dcm_files:
                # Find the directory containing DICOMs
                dcm_dir = dcm_files[0].parent
                print(f"  Converting CT: {case_name}")
                dcm_series_to_nifti(dcm_dir, img_out)
            else:
                # Maybe it's already a NIfTI or other format
                nii_files = list(pat_dir.rglob("*.nii*"))
                if nii_files:
                    shutil.copy2(nii_files[0], img_out)
                else:
                    print(f"  Warning: No image files found for {case_name}")
                    continue

        # Convert label
        gt_file = gt_root / f"{case_name}.mat"
        lbl_out = labels_dir / f"{case_id}.nii.gz"
        if gt_file.exists() and not lbl_out.exists():
            print(f"  Converting label: {case_name}")
            mat_mask_to_nifti(gt_file, lbl_out)

        case_list.append(case_id)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "PE": 1},
        "numTraining": len(case_list),
        "file_ending": ".nii.gz",
        "name": "FUMPE",
        "description": "FUMPE Pulmonary Embolism dataset",
    }
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"FUMPE: {len(case_list)} cases converted to {output_dir}")
    return case_list


def convert_cadpe(
    data_root: str | Path,
    output_dir: str | Path,
    dataset_id: int = 102,
) -> None:
    """
    Convert CAD-PE dataset to nnUNet format.

    Input:
        data/CAD-PE/
        ├── images/001.nrrd, 002.nrrd, e0032.nrrd, ...
        └── rs/0001RefStd.nrrd, e0032RefStd.nrrd, ...

    Output: Dataset102_CADPE/
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir) / f"Dataset{dataset_id:03d}_CADPE"
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    img_root = data_root / "images"
    rs_root = data_root / "rs"

    case_list = []
    for img_file in sorted(img_root.glob("*.nrrd")):
        stem = img_file.stem  # e.g., "001" or "e0032"
        case_id = f"CADPE_{stem}"

        img_out = images_dir / f"{case_id}_0000.nii.gz"
        if not img_out.exists():
            print(f"  Converting image: {stem}")
            nrrd_to_nifti(img_file, img_out)

        # Find matching reference standard
        # Naming conventions: 001 → 0001RefStd.nrrd or e0032 → e0032RefStd.nrrd
        rs_candidates = [
            rs_root / f"{stem}RefStd.nrrd",
            rs_root / f"0{stem}RefStd.nrrd",
            rs_root / f"00{stem}RefStd.nrrd",
            rs_root / f"000{stem}RefStd.nrrd",
        ]
        rs_file = None
        for c in rs_candidates:
            if c.exists():
                rs_file = c
                break

        if rs_file is None:
            # Try glob matching
            matches = list(rs_root.glob(f"*{stem}*RefStd*"))
            if matches:
                rs_file = matches[0]

        lbl_out = labels_dir / f"{case_id}.nii.gz"
        if rs_file is not None and not lbl_out.exists():
            print(f"  Converting label: {stem}")
            nrrd_to_nifti(rs_file, lbl_out)
        elif rs_file is None:
            print(f"  Warning: No reference standard found for {stem}")

        case_list.append(case_id)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "PE": 1},
        "numTraining": len(case_list),
        "file_ending": ".nii.gz",
        "name": "CADPE",
        "description": "CAD-PE Computer Aided Detection of PE",
    }
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"CAD-PE: {len(case_list)} cases converted to {output_dir}")
    return case_list


def build_rspect_manifest(
    data_root: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Build manifest for RSPECT image-level data.

    Reads train.csv and maps to DICOM file paths.

    Output CSV columns:
        study_uid, series_uid, sop_uid, pe_present, file_path
    """
    data_root = Path(data_root)
    train_csv = pd.read_csv(data_root / "train.csv")

    # Aggregate to study level
    study_labels = (
        train_csv.groupby("StudyInstanceUID")
        .agg(
            pe_present=("pe_present_on_image", "max"),
            negative_exam=("negative_exam_for_pe", "max"),
            num_slices=("SOPInstanceUID", "count"),
        )
        .reset_index()
    )

    # Map study UIDs to directories
    train_dir = data_root / "train"
    records = []
    for _, row in study_labels.iterrows():
        study_uid = row["StudyInstanceUID"]
        study_dir = train_dir / study_uid
        if not study_dir.exists():
            continue
        # Find series subdirectory
        for series_dir in study_dir.iterdir():
            if series_dir.is_dir():
                records.append(
                    {
                        "study_uid": study_uid,
                        "series_uid": series_dir.name,
                        "pe_present": int(row["pe_present"]),
                        "negative_exam": int(row["negative_exam"]),
                        "num_slices": int(row["num_slices"]),
                        "dicom_dir": str(series_dir),
                    }
                )

    df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"RSPECT manifest: {len(df)} studies written to {output_path}")
    return df


def build_augmented_rspect_manifest(
    data_root: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Build manifest for Augmented_RSPECT box-level data.

    Reads augmented_rspect.csv and maps to DICOM paths + box annotations.

    Output CSV columns:
        study_uid, series_uid, sop_uid, artery, x, y, w, h, file_path
    """
    data_root = Path(data_root)
    box_csv = pd.read_csv(data_root / "augmented_rspect.csv")

    # Aggregate boxes per study
    study_boxes = (
        box_csv.groupby("StudyInstanceUID")
        .agg(
            num_boxes=("Artery", "count"),
            arteries=("Artery", lambda x: list(set(x))),
        )
        .reset_index()
    )

    # Map to directories
    train_dir = data_root / "train"
    records = []
    for _, row in box_csv.iterrows():
        study_uid = row["StudyInstanceUID"]
        series_uid = row["SeriesInstanceUID"]
        sop_uid = row["SOPInstanceUID"]

        # Find DICOM file
        dcm_path = train_dir / study_uid / series_uid / f"{sop_uid}.dcm"

        records.append(
            {
                "study_uid": study_uid,
                "series_uid": series_uid,
                "sop_uid": sop_uid,
                "artery": row["Artery"],
                "x": row["x"],
                "y": row["y"],
                "width": row["width"],
                "height": row["height"],
                "dicom_path": str(dcm_path) if dcm_path.exists() else "",
            }
        )

    df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        f"Augmented RSPECT manifest: {len(df)} boxes across "
        f"{df['study_uid'].nunique()} studies written to {output_path}"
    )
    return df
