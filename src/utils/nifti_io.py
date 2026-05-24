"""NIfTI file input/output utilities."""

from pathlib import Path
from typing import Union, Tuple, Any

import nibabel as nib
import numpy as np


def load_volume(case_dir: Path, suffix: str, return_header: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
    """Find and load a NIfTI volume based on its file suffix.

    Args:
        case_dir: Path object pointing to the patient's directory.
        suffix: The sequence identifier (e.g., 't1c', 'seg').
        return_header: If True, returns a tuple of (data, header object).

    Returns:
        A 3D numpy array containing the image data. If return_header is True, 
        returns (data, header).

    Raises:
        FileNotFoundError: If the corresponding file is not found.
    """
    matches = list(case_dir.glob(f"*{suffix}*.nii.gz"))
    if not matches:
        raise FileNotFoundError(f"File not found for suffix '{suffix}' in {case_dir}")
    
    img = nib.load(matches[0])
    data = img.get_fdata()
    
    if return_header:
        return data, img.header
        
    return data