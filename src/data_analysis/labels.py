"""Generate and save visual breakdowns of meningioma segmentation labels

This script loads a sample patient MRI and its corresponding
segmentation mask, and generates two sets of figures:
- The base BraTS subregions
- The combined clinical regions
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

    patient_id = "BraTS-MEN-00891-000"
    patient_dir = cfg.train_data_dir / patient_id

    # 2. Data Loading
    try:
        mask = load_volume(patient_dir, "seg")
        t1 = load_volume(patient_dir, "t1c")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 3. Processing
    slice_idx = int(np.argmax((mask > 0).sum(axis=(0, 1))))
    
    s = mask[:, :, slice_idx]
    bg = t1[:, :, slice_idx]
    
    bg_norm = normalize_for_display(bg, ignore_background=False)

    # 4. Figure Generation: BraTS 2023 Original Labels
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
    axes1 = axes1.flatten()

    label_info = [
        (s == 1, "NETC", "Reds"),
        (s == 2, "SNFH", "Greens"),
        (s == 3, "ET", "Blues"),
    ]

    for ax, (lbl_mask, title, cmap) in zip(axes1[:3], label_info):
        ax.imshow(bg_norm.T, origin="lower", cmap="gray")
        display = np.where(lbl_mask, 1.0, np.nan)
        ax.imshow(display.T, origin="lower", cmap=cmap, vmin=0, vmax=1, alpha=0.7)
        ax.set_title(title)
        ax.axis("off")

    combined_display = np.where(s > 0, s, np.nan)
    axes1[3].imshow(bg_norm.T, origin="lower", cmap="gray")
    axes1[3].imshow(combined_display.T, origin="lower", cmap="tab10", vmin=1, vmax=10, alpha=0.7)
    axes1[3].set_title("Combinadas")
    axes1[3].axis("off")

    fig1.tight_layout()
    output_path_labels = cfg.data_analysis_dir / "segmentation_labels.svg"
    fig1.savefig(output_path_labels, format='svg')
    plt.close(fig1)

    # 5. Figure Generation: Classic BraTS Clinical Regions
    fig2 = plt.figure(figsize=(10, 10))
    gs = fig2.add_gridspec(2, 4)

    ax_et = fig2.add_subplot(gs[0, 0:2])  
    ax_tc = fig2.add_subplot(gs[0, 2:4])  
    ax_wt = fig2.add_subplot(gs[1, 1:3])  

    axes2 = [ax_et, ax_tc, ax_wt]

    et_mask = (s == 3)                  
    tc_mask = (s == 1) | (s == 3)       
    wt_mask = (s > 0)                   

    regions_info = [
        (et_mask, "ET", "Blues"),
        (tc_mask, "TC", "Purples"),
        (wt_mask, "WT", "Oranges"),
    ]

    for ax, (lbl_mask, title, cmap) in zip(axes2, regions_info):
        ax.imshow(bg_norm.T, origin="lower", cmap="gray")
        display = np.where(lbl_mask, 1.0, np.nan)
        ax.imshow(display.T, origin="lower", cmap=cmap, vmin=0, vmax=1, alpha=0.7)
        ax.set_title(title)
        ax.axis("off")

    fig2.tight_layout()
    output_path_regions = cfg.data_analysis_dir / "segmentation_regions.svg"
    fig2.savefig(output_path_regions, format='svg')
    plt.close(fig2)

if __name__ == "__main__":
    main()