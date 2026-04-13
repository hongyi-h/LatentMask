"""Connected component extraction from 3D segmentation masks."""
import numpy as np
from scipy import ndimage


def extract_connected_components(mask, min_size=1, fg_label=None):
    """Extract connected components from a binary or multi-class mask.

    Args:
        mask: 3D numpy array. Non-zero values treated as foreground.
        min_size: ignore CCs smaller than this.
        fg_label: if set, only this label value is treated as foreground.
                  E.g. fg_label=2 for LiTS tumor-only extraction.

    Returns:
        list of dicts: each with 'size', 'log_size', 'bbox', 'label_id'.
        bbox format: ((z1,z2), (y1,y2), (x1,x2)) as half-open intervals.
    """
    if fg_label is not None:
        binary = (mask == fg_label).astype(np.int32)
    else:
        binary = (mask > 0).astype(np.int32)
    labeled, num = ndimage.label(binary)
    if num == 0:
        return []

    slices_list = ndimage.find_objects(labeled)
    ccs = []
    for i, slices in enumerate(slices_list):
        if slices is None:
            continue
        label_id = i + 1
        size = int((labeled[slices] == label_id).sum())
        if size < min_size:
            continue
        bbox = tuple((s.start, s.stop) for s in slices)
        ccs.append({
            'label_id': label_id,
            'size': size,
            'log_size': float(np.log(max(size, 1))),
            'bbox': bbox,
        })
    return ccs


def extract_ccs_from_patch(seg_patch, min_size=1, fg_label=None):
    """Extract CCs from a cropped segmentation patch.

    Same as extract_connected_components but operates on a patch
    where coordinates are relative to the patch origin.
    """
    return extract_connected_components(seg_patch, min_size=min_size,
                                        fg_label=fg_label)


def compute_safe_zone_mask(seg_patch, d_margin, fg_label=None):
    """Compute safe zone: voxels at least d_margin away from any foreground.

    The safe zone excludes ALL foreground voxels (not just fg_label),
    because the network's foreground probability covers all non-background
    classes. This prevents penalizing correct predictions in non-target
    foreground regions (e.g. liver when fg_label=tumor).

    Args:
        seg_patch: 3D segmentation patch (foreground > 0).
        d_margin: minimum distance in voxels from any foreground.
        fg_label: used for CC extraction elsewhere; safe zone always
                  excludes all foreground (seg > 0).

    Returns:
        Binary mask where 1 = safe zone.
    """
    # Always exclude ALL foreground from safe zone, regardless of fg_label,
    # to avoid penalizing predictions in non-target foreground regions.
    fg = (seg_patch > 0).astype(np.float32)
    dist = ndimage.distance_transform_edt(1 - fg)
    return (dist >= d_margin).astype(np.float32)
