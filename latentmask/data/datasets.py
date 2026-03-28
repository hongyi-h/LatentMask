"""
Multi-granularity data loading for LatentMask.

Provides three dataset types that feed into the unified trainer:
  1. PixelDataset — pixel-labeled volumes (READ-PE, FUMPE, CAD-PE; also LiTS)
  2. BoxDataset — box-annotated studies (Augmented_RSPECT)
  3. ImageDataset — image-level labeled studies (RSPECT)

Each returns a dict compatible with nnUNet's batch format, plus extra keys
for supervision type and annotations.

Design: domain-agnostic. Vesselness is NOT computed here (optional, only
if the trainer requests it via use_vesselness_hint).
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class PixelDataset(Dataset):
    """
    Dataset for pixel-labeled 3D volumes.

    Returns pre-computed patches during training (via nnUNet preprocessing),
    or full volumes for evaluation.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        case_ids: list[str] | None = None,
        patch_size: tuple[int, ...] | None = None,
        transform=None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.patch_size = patch_size
        self.transform = transform

        if case_ids is not None:
            self.case_ids = case_ids
        else:
            self.case_ids = sorted(
                [
                    f.name.replace("_0000.nii.gz", "")
                    for f in self.images_dir.glob("*_0000.nii.gz")
                ]
            )

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> dict:
        case_id = self.case_ids[idx]

        # Load image
        img_path = self.images_dir / f"{case_id}_0000.nii.gz"
        img_nii = nib.load(str(img_path))
        image = img_nii.get_fdata().astype(np.float32)

        # Load label
        lbl_path = self.labels_dir / f"{case_id}.nii.gz"
        label = nib.load(str(lbl_path)).get_fdata().astype(np.float32)
        label = (label > 0).astype(np.float32)

        # Random patch extraction
        if self.patch_size is not None:
            image, label = self._extract_patch(image, label)

        # Normalise CT to [0, 1] range (clip to lung window)
        image = np.clip(image, -1024, 600)
        image = (image - (-1024)) / (600 - (-1024))

        # To tensors: (1, D, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0)

        batch = {
            "data": image,
            "target": label,
            "batch_type": "pixel",
            "keys": case_id,
        }

        if self.transform:
            batch = self.transform(batch)

        return batch

    def _extract_patch(
        self,
        image: np.ndarray,
        label: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a random patch, preferring foreground."""
        D, H, W = image.shape
        pD, pH, pW = self.patch_size

        # 33% chance to center on foreground
        fg_coords = np.argwhere(label > 0)
        if len(fg_coords) > 0 and np.random.rand() < 0.33:
            center = fg_coords[np.random.randint(len(fg_coords))]
            d0 = max(0, min(center[0] - pD // 2, D - pD))
            h0 = max(0, min(center[1] - pH // 2, H - pH))
            w0 = max(0, min(center[2] - pW // 2, W - pW))
        else:
            d0 = np.random.randint(0, max(1, D - pD + 1))
            h0 = np.random.randint(0, max(1, H - pH + 1))
            w0 = np.random.randint(0, max(1, W - pW + 1))

        image = image[d0 : d0 + pD, h0 : h0 + pH, w0 : w0 + pW]
        label = label[d0 : d0 + pD, h0 : h0 + pH, w0 : w0 + pW]

        # Pad if needed
        pad = []
        for actual, target in zip(image.shape, self.patch_size):
            if actual < target:
                pad.append((0, target - actual))
            else:
                pad.append((0, 0))
        if any(p != (0, 0) for p in pad):
            image = np.pad(image, pad, mode="constant", constant_values=0)
            label = np.pad(label, pad, mode="constant", constant_values=0)

        return image, label


class BoxDataset(Dataset):
    """
    Dataset for box-annotated CTPA studies (Augmented_RSPECT).

    Loads 3D volumes from DICOM series and constructs box masks
    from bounding box annotations.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        patch_size: tuple[int, ...] | None = None,
        transform=None,
    ):
        self.manifest = pd.read_csv(manifest_csv)
        self.patch_size = patch_size
        self.transform = transform

        # Group by study
        self.studies = self.manifest.groupby("study_uid")
        self.study_ids = list(self.studies.groups.keys())

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> dict:
        study_uid = self.study_ids[idx]
        study_df = self.studies.get_group(study_uid)

        # Load DICOM series → 3D volume
        series_uid = study_df["series_uid"].iloc[0]
        dicom_dir = study_df["dicom_dir"].iloc[0] if "dicom_dir" in study_df.columns else None

        if dicom_dir and Path(dicom_dir).exists():
            volume, spacing = self._load_dicom_series(Path(dicom_dir))
        else:
            # Construct path from UIDs
            # Fallback: create placeholder
            volume = np.zeros((64, 512, 512), dtype=np.float32)
            spacing = (1.0, 1.0, 1.0)

        # Build box mask from annotations
        box_mask = np.zeros_like(volume)
        box_target = np.zeros_like(volume)  # 1 inside boxes where PE present

        for _, row in study_df.iterrows():
            if pd.isna(row.get("x")) or pd.isna(row.get("y")):
                continue
            # Map 2D box to the correct slice (SOPInstanceUID identifies slice)
            sop_uid = row["sop_uid"]
            x, y = int(row["x"]), int(row["y"])
            w, h = int(row["width"]) + 1, int(row["height"]) + 1

            # Find slice index (would need SOP→slice mapping from DICOM load)
            # For now, use all-slice bounding box
            y1, y2 = max(0, y), min(volume.shape[1], y + h)
            x1, x2 = max(0, x), min(volume.shape[2], x + w)

            box_mask[:, y1:y2, x1:x2] = 1.0
            box_target[:, y1:y2, x1:x2] = 1.0  # PE inside box

        # Normalise CT
        volume = np.clip(volume, -1024, 600)
        volume = (volume - (-1024)) / (600 - (-1024))

        # Random patch
        if self.patch_size is not None:
            volume, box_mask, box_target = self._extract_patch_3(
                volume, box_mask, box_target
            )

        batch = {
            "data": torch.from_numpy(volume).unsqueeze(0),
            "target": torch.from_numpy(box_target).unsqueeze(0),
            "box_mask": torch.from_numpy(box_mask).unsqueeze(0),
            "batch_type": "box",
            "keys": study_uid,
        }

        if self.transform:
            batch = self.transform(batch)

        return batch

    @staticmethod
    def _load_dicom_series(dcm_dir: Path) -> tuple[np.ndarray, tuple]:
        slices = []
        for f in sorted(dcm_dir.iterdir()):
            if f.suffix == ".dcm":
                ds = pydicom.dcmread(str(f))
                slices.append(ds)

        try:
            slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
        except (AttributeError, TypeError):
            slices.sort(key=lambda s: int(s.InstanceNumber))

        volume = np.stack(
            [s.pixel_array.astype(np.float32) for s in slices], axis=0
        )
        ds0 = slices[0]
        slope = float(getattr(ds0, "RescaleSlope", 1))
        intercept = float(getattr(ds0, "RescaleIntercept", 0))
        volume = volume * slope + intercept

        try:
            ps = [float(x) for x in ds0.PixelSpacing]
            dz = abs(
                float(slices[1].ImagePositionPatient[2])
                - float(slices[0].ImagePositionPatient[2])
            ) if len(slices) > 1 else float(getattr(ds0, "SliceThickness", 1.0))
            spacing = (dz, ps[0], ps[1])
        except (AttributeError, TypeError):
            spacing = (1.0, 1.0, 1.0)

        return volume, spacing

    def _extract_patch_3(self, *arrays):
        D, H, W = arrays[0].shape
        pD, pH, pW = self.patch_size
        d0 = np.random.randint(0, max(1, D - pD + 1))
        h0 = np.random.randint(0, max(1, H - pH + 1))
        w0 = np.random.randint(0, max(1, W - pW + 1))
        result = []
        for arr in arrays:
            patch = arr[d0 : d0 + pD, h0 : h0 + pH, w0 : w0 + pW]
            # Pad if needed
            pad = [(0, max(0, s - patch.shape[i])) for i, s in enumerate(self.patch_size)]
            if any(p[1] > 0 for p in pad):
                patch = np.pad(patch, pad, mode="constant")
            result.append(patch)
        return tuple(result)


class ImageDataset(Dataset):
    """
    Dataset for image-level labeled CT studies (RSPECT).

    Each study has a binary label: PE present or absent.
    No voxel-level annotation.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        patch_size: tuple[int, ...] | None = None,
        max_slices: int = 128,
        transform=None,
    ):
        self.manifest = pd.read_csv(manifest_csv)
        self.patch_size = patch_size
        self.max_slices = max_slices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        study_uid = row["study_uid"]
        pe_present = int(row["pe_present"])
        dicom_dir = row.get("dicom_dir", "")

        # Load DICOM series
        if dicom_dir and Path(dicom_dir).exists():
            volume, _ = BoxDataset._load_dicom_series(Path(dicom_dir))
        else:
            volume = np.zeros((64, 512, 512), dtype=np.float32)

        # Subsample if too many slices
        if volume.shape[0] > self.max_slices:
            step = volume.shape[0] // self.max_slices
            volume = volume[::step][: self.max_slices]

        # Normalise
        volume = np.clip(volume, -1024, 600)
        volume = (volume - (-1024)) / (600 - (-1024))

        # Random patch
        if self.patch_size is not None:
            D, H, W = volume.shape
            pD, pH, pW = self.patch_size
            d0 = np.random.randint(0, max(1, D - pD + 1))
            h0 = np.random.randint(0, max(1, H - pH + 1))
            w0 = np.random.randint(0, max(1, W - pW + 1))
            volume = volume[d0 : d0 + pD, h0 : h0 + pH, w0 : w0 + pW]
            # Pad if needed
            pad = [(0, max(0, s - volume.shape[i])) for i, s in enumerate(self.patch_size)]
            if any(p[1] > 0 for p in pad):
                volume = np.pad(volume, pad, mode="constant")

        batch = {
            "data": torch.from_numpy(volume).unsqueeze(0),
            "target": torch.zeros(1, *volume.shape),  # dummy
            "image_label": torch.tensor(pe_present, dtype=torch.float32),
            "batch_type": "image",
            "keys": study_uid,
        }

        if self.transform:
            batch = self.transform(batch)

        return batch


def build_multi_granularity_loaders(
    pixel_images_dir: str | Path,
    pixel_labels_dir: str | Path,
    pixel_case_ids: list[str],
    rspect_manifest: str | Path,
    aug_rspect_manifest: str | Path,
    patch_size: tuple[int, int, int] = (64, 128, 128),
    batch_size_pixel: int = 2,
    batch_size_box: int = 1,
    batch_size_image: int = 2,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Build data loaders for all three granularity levels."""

    pixel_ds = PixelDataset(
        images_dir=pixel_images_dir,
        labels_dir=pixel_labels_dir,
        case_ids=pixel_case_ids,
        patch_size=patch_size,
    )

    box_ds = BoxDataset(
        manifest_csv=aug_rspect_manifest,
        patch_size=patch_size,
    )

    image_ds = ImageDataset(
        manifest_csv=rspect_manifest,
        patch_size=patch_size,
    )

    loaders = {
        "pixel": DataLoader(
            pixel_ds,
            batch_size=batch_size_pixel,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "box": DataLoader(
            box_ds,
            batch_size=batch_size_box,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "image": DataLoader(
            image_ds,
            batch_size=batch_size_image,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
    }
    return loaders
