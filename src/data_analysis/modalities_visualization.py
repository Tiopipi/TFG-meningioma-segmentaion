"""Generate and save a visual breakdown of MRI modalities and their segmentation mask

This script loads a specific patient, extracts all MRI modalities, the segmentation
mask, normalizes the intensities for visualization and generates a plot.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import configs.config as cfg
from src.utils.nifti_io import load_volume
from src.utils.image_processing import normalize_for_display
from src.utils.plot_utils import setup_plotting_environment

def main() -> None:
    # 1. Configuration and Setup
    setup_plotting_environment(
        output_dir=cfg.data_analysis_dir, 
        font_size=14, 
        title_size=18
    )

    patient_id = "BraTS-MEN-00010-000"
    case_dir = cfg.train_data_dir / patient_id

    modalities = {
        "T1 precontraste": "t1n",
        "T1 postcontraste": "t1c",
        "T2": "t2w",
        "T2 FLAIR": "t2f",
        "Máscara": "seg"
    }

    # 2. Data Loading
    try:
        volumes = {name: load_volume(case_dir, suffix) for name, suffix in modalities.items()}
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 3. Processing
    seg_vol = volumes["Máscara"]
    tumor_slices = np.where(np.sum(seg_vol > 0, axis=(0, 1)) > 0)[0]

    if len(tumor_slices) > 0:
        slice_z = tumor_slices[len(tumor_slices) // 2]
    else:
        slice_z = seg_vol.shape[2] // 2

    # 4. Figure Generation
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    display_names = ["T1 precontraste", "T1 postcontraste", "T2", "T2 FLAIR", "Máscara"]

    for i, name in enumerate(display_names):
        slice_img = volumes[name][:, :, slice_z]
        if name != "Máscara":
            slice_img = normalize_for_display(slice_img, ignore_background=False)
            axes[i].imshow(np.rot90(slice_img), cmap="gray")
        else:
            axes[i].imshow(np.rot90(slice_img), cmap="nipy_spectral", interpolation="nearest")
        
        axes[i].set_title(name)
        axes[i].axis("off")

    base_mri = normalize_for_display(volumes["T1 postcontraste"][:, :, slice_z], ignore_background=False)
    mask_slice = volumes["Máscara"][:, :, slice_z]

    axes[5].imshow(np.rot90(base_mri), cmap="gray")
    masked_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    axes[5].imshow(np.rot90(masked_overlay), cmap="jet", alpha=0.45, interpolation="nearest")
    axes[5].set_title("Máscara sobre T1 postcontraste")
    axes[5].axis("off")

    plt.tight_layout()
    
    output_path = cfg.data_analysis_dir / "caso_modalidades_mascara.svg"
    plt.savefig(output_path, format='svg', bbox_inches="tight")
    plt.close()
    
if __name__ == "__main__":
    main()