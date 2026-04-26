"""Dataset wrapper that swaps GT seg with box_seg for box-supervised scans.

Used by LatentMaskTrainer's box dataloader so that spatial augmentations
(rotation, mirror, crop) apply to box annotations the same way they apply
to image data. The box_seg volume contains rectangular box regions (not GT
foreground shapes), so no label leakage occurs.
"""
import os
import numpy as np


class BoxSegDatasetWrapper:
    """Wraps an nnUNet dataset, replacing seg with box_seg on load_case."""

    def __init__(self, base_dataset, box_seg_dir):
        """
        Args:
            base_dataset: nnUNet dataset instance (nnUNetDatasetNumpy or Blosc2)
            box_seg_dir: path to box_segmentations/ for one protocol
        """
        self._base = base_dataset
        self._box_seg_dir = box_seg_dir
        self.identifiers = base_dataset.identifiers
        self.source_folder = base_dataset.source_folder

    def load_case(self, identifier):
        data, _seg, seg_prev, properties = self._base.load_case(identifier)

        box_seg_path = os.path.join(self._box_seg_dir, f'{identifier}.npy')
        if os.path.exists(box_seg_path):
            box_seg = np.load(box_seg_path, mmap_mode='r')
        else:
            box_seg = np.zeros_like(_seg)

        return data, box_seg, seg_prev, properties

    def __getitem__(self, identifier):
        return self.load_case(identifier)
