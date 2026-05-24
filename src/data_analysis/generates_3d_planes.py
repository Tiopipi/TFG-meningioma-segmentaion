"""Generate and save a multi-planar visualization of a tumor segmentation.

This script loads the T1c modality and segmentation mask for a specific patient,
calculates the 3D center of the tumor, and extracts the corresponding axial,
coronal, and sagittal planes.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import configs.config as cfg
from src.utils.nifti_io import load_volume
from src.utils.image_processing import prepare_plane, get_tumor_center_slices
from src.utils.plot_utils import setup_plotting_environment


def show_plane(ax: plt.Axes, img_slice: np.ndarray, mask_slice: np.ndarray, title: str) -> None:
    """Format and display a 2D MRI slice with its corresponding segmentation mask overlay."""
    img_slice, mask_slice = prepare_plane(img_slice, mask_slice)

    ax.imshow(img_slice, cmap="gray")
    overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    ax.imshow(overlay, cmap="jet", alpha=0.45, interpolation="nearest")

    ax.set_title(title, pad=10)
    ax.axis("off")


def main() -> None:
    # 1. Configuration and Setup
    setup_plotting_environment(
        output_dir=cfg.data_analysis_dir, 
        font_size=13, 
        title_size=16
    )

    patient_id = "BraTS-MEN-00010-000"
    case_dir = cfg.train_data_dir / patient_id

    # 2. Data Loading
    try:
        t1c = load_volume(case_dir, "t1c")
        seg = load_volume(case_dir, "seg")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 3. Processing
    slice_x, slice_y, slice_z = get_tumor_center_slices(seg)

    # 4. Figure Generation
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    show_plane(
        axes[0],
        t1c[:, :, slice_z],
        seg[:, :, slice_z],
        "Plano axial"
    )

    show_plane(
        axes[1],
        t1c[:, slice_y, :],
        seg[:, slice_y, :],
        "Plano coronal"
    )

    show_plane(
        axes[2],
        t1c[slice_x, :, :],
        seg[slice_x, :, :],
        "Plano sagital"
    )

    plt.subplots_adjust(wspace=0.08)

    output_path = cfg.data_analysis_dir / "caso_planos_3d.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close()

if __name__ == "__main__":
    main()