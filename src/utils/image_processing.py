"""Medical image processing and spatial manipulation utilities.

This module contains functions for intensity normalization, geometric 
transformations, cropping, and padding of MRI slices and segmentation masks.
"""

import numpy as np


def normalize_for_display(img: np.ndarray, ignore_background: bool = True) -> np.ndarray:
    """Normalize image intensities.
    
    Args:
        img: The image array to be normalized.
        ignore_background: If True, calculates percentiles exclusively on non-zero pixels.
            
    Returns:
        A normalized float32 numpy array with values bounded between 0 and 1.
    """
    img = img.astype(np.float32)
    
    if ignore_background and np.any(img > 0):
        p1, p99 = np.percentile(img[img > 0], [1, 99])
    else:
        p1, p99 = np.percentile(img, [1, 99])
        
    img = np.clip(img, p1, p99)

    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    return img


def get_tumor_center_slices(seg: np.ndarray) -> tuple[int, int, int]:
    """Calculate the 3D center of mass of the tumor to extract optimal slice indices."""
    coords = np.argwhere(seg > 0)

    if coords.size == 0:
        return seg.shape[0] // 2, seg.shape[1] // 2, seg.shape[2] // 2

    center = np.round(coords.mean(axis=0)).astype(int)
    return int(center[0]), int(center[1]), int(center[2])


def crop_to_content(img: np.ndarray, mask: np.ndarray, margin: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Crop a 2D slice to the bounding box of its non-zero content plus a margin."""
    content = (img > 0.05) | (mask > 0)
    coords = np.argwhere(content)

    if coords.size == 0:
        return img, mask

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)
    y_max = min(y_max + margin, img.shape[0] - 1)
    x_max = min(x_max + margin, img.shape[1] - 1)

    return img[y_min:y_max + 1, x_min:x_max + 1], mask[y_min:y_max + 1, x_min:x_max + 1]


def pad_to_square(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad a 2D image equally to make its dimensions square."""
    h, w = img.shape
    size = max(h, w)

    pad_h = size - h
    pad_w = size - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")

    return img, mask


def prepare_plane(img_slice: np.ndarray, mask_slice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Process a 2D slice and its mask by normalizing, rotating, cropping, and padding."""
    img_slice = normalize_for_display(img_slice, ignore_background=True)

    img_slice = np.rot90(img_slice)
    mask_slice = np.rot90(mask_slice)

    img_slice, mask_slice = crop_to_content(img_slice, mask_slice)
    img_slice, mask_slice = pad_to_square(img_slice, mask_slice)

    return img_slice, mask_slice