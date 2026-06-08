"""Analyze and visualize the presence, specific combinations, and conditional volume of tumor subregions.

This script acts as a standalone tool. It iterates through the training dataset,
loads the NIfTI segmentation masks, and calculates the presence of each subregion.
It generates three separate figures:
1. Individual label presence rates.
2. The specific topological combinations present in the dataset.
3. The volume distribution of each region excluding zero-volume cases.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import configs.config as cfg
from src.utils.nifti_io import load_volume
from src.utils.plot_utils import setup_plotting_environment

def main() -> None:
    # 1. Configuration and Setup
    setup_plotting_environment(output_dir=cfg.data_analysis_dir, use_seaborn=True,
                               font_size=18, title_size=20)

    VOXEL_VOL_MM3 = 1.0  
    LABELS = {1: "NETC", 2: "SNFH", 3: "ET"}
    
    records = []

    # 2. Data Extraction Loop
    for case_dir in sorted(cfg.train_data_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        try:
            seg_vol = load_volume(case_dir, "seg").astype(np.uint8)
        except FileNotFoundError:
            continue
        
        record = {"case_id": case_dir.name}
        present_labels = []

        for label_id, label_name in LABELS.items():
            voxels = int(np.sum(seg_vol == label_id))
            vol_mm3 = voxels * VOXEL_VOL_MM3
            
            record[f"vol_mm3_{label_name}"] = vol_mm3
            
            has_label = voxels > 0
            record[f"has_{label_name}"] = has_label
            
            if has_label:
                present_labels.append(label_name)

        if not present_labels:
            record["combination"] = "Ninguna (Fondo)"
        else:
            record["combination"] = " + ".join(present_labels)

        records.append(record)

    if not records:
        print("Error: No segmentation files found to analyze.")
        sys.exit(1)

    df = pd.DataFrame(records)
    total_cases = len(df)

    # 3. Statistical Analysis
    presence_stats = {}
    label_names = list(LABELS.values())
    
    for label in label_names:
        presence_count = df[f"has_{label}"].sum()
        presence_stats[label] = (presence_count / total_cases) * 100
        
    print("\nLabel Presence Summary")
    for label, pct in presence_stats.items():
        print(f"{label}: Present in {pct:.1f}% of cases")

    combo_counts = df["combination"].value_counts()
    print("\nSpecific Combinations (Topologies)")
    for combo, count in combo_counts.items():
        print(f"{combo}: {count} cases ({(count/total_cases)*100:.1f}%)")

    # 4. Figure Generation
    # Graph 1: Label Presence
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = sns.barplot(
        x=list(presence_stats.keys()), 
        y=list(presence_stats.values()), 
        palette="viridis"
    )
    ax1.set_title("Presencia de subregiones en el dataset", fontweight="bold", pad=15)
    ax1.set_ylabel("Porcentaje de pacientes (%)", labelpad=15)
    ax1.set_ylim(0, 110)
    for i, val in enumerate(presence_stats.values()):
        ax1.text(i, val + 2, f"{val:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    out_bar = cfg.data_analysis_dir / "label_presence_bar.svg"
    plt.savefig(out_bar, format='svg')
    plt.close(fig1)

    # Graph 2: Exact Combinations
    fig2 = plt.figure(figsize=(10, 6))
    
    ax2 = sns.barplot(
        x=combo_counts.values, 
        y=combo_counts.index, 
        palette="magma"
    )
    ax2.set_title("Combinaciones topológicas presentes en los tumores", fontweight="bold", pad=15)
    ax2.set_xlabel("Número de pacientes")
    ax2.set_ylabel("Combinación de subregiones", labelpad=15)
    
    for p in ax2.patches:
        width = p.get_width()
        ax2.text(width + (total_cases * 0.01), p.get_y() + p.get_height() / 2,
                 f"{int(width)}", ha='left', va='center', fontweight='bold')

    ax2.set_xlim(0, combo_counts.max() * 1.15)

    plt.tight_layout()
    out_combo = cfg.data_analysis_dir / "label_combinations_hbar.svg"
    plt.savefig(out_combo, format='svg')
    plt.close(fig2)

    # Graph 3: Conditional Volume
    vol_data = []
    for label in label_names:
        present_vols = df.loc[df[f"has_{label}"], f"vol_mm3_{label}"]
        for vol in present_vols:
            vol_data.append({"Subregión": label, "Volumen (mm³)": vol})
            
    df_plot = pd.DataFrame(vol_data)

    fig3 = plt.figure(figsize=(8, 6))
    ax3 = sns.boxplot(
        data=df_plot, 
        x="Subregión", 
        y="Volumen (mm³)", 
        palette="viridis",
        showfliers=False
    )
    ax3.set_title("Volumen condicional (cuando está presente)", fontweight="bold", pad=15)
    ax3.set_ylabel("Volumen (mm³)", labelpad=15)

    plt.tight_layout()
    out_box = cfg.data_analysis_dir / "label_conditional_volume_box.svg"
    plt.savefig(out_box, format='svg')
    plt.close(fig3)

if __name__ == "__main__":
    main()