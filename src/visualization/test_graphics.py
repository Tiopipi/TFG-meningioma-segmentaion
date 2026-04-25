"""Generates comparative evaluation plots for segmentation models

This script takes the evaluation results from multiple segmentation
models and generates figures including overall performance, 
subregion-specific Dice scores, Hausdorff Distance comparisons, and an 
efficiency analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import configs.config as cfg

def main() -> None:
    # 1. Configuration and Setup
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["figure.dpi"] = 300

    os.makedirs(cfg.graphs_dir, exist_ok=True)

    # 2. Data Loading
    csv_files = list(cfg.logs_dir.glob("evaluation_results_*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in {cfg.logs_dir}. Make sure you ran the evaluation script.")
        sys.exit(1)

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # 3. Plot Generation

    # Graph 1: Global Comparison (Dice and IoU)
    plt.figure(figsize=(8, 6))
    df_melted_global = data.melt(id_vars="Model", value_vars=["Dice_Global", "IoU_Global"], 
                                var_name="Metric", value_name="Score")

    sns.barplot(data=df_melted_global, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.title("Overall Segmentation Performance")
    plt.ylim(0, 1)
    plt.ylabel("Score (0 - 1)")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(cfg.graphs_dir / "1_overall_performance.svg", format='svg')
    plt.close()

    # Graph 2: Dice Breakdown by Subregions
    plt.figure(figsize=(10, 6))
    df_melted_dice = data.melt(id_vars="Model", value_vars=["Dice_NCR", "Dice_SNFH", "Dice_ET"], 
                            var_name="Subregion", value_name="Dice")

    df_melted_dice['Subregion'] = df_melted_dice['Subregion'].str.replace('Dice_', '')

    sns.barplot(data=df_melted_dice, x="Model", y="Dice", hue="Subregion", palette="mako")
    plt.title("Dice Coefficient by Tumor Subregion")
    plt.ylim(0, 1)
    plt.ylabel("Dice Coefficient")

    plt.legend(title="Subregion (NCR, SNFH, ET)", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(cfg.graphs_dir / "2_dice_by_class.svg", format='svg')
    plt.close()

    # Graph 3: Hausdorff Distance 
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x="Model", y="HD95_Global", palette="rocket")
    plt.title("Global 95% Hausdorff Distance (Lower is Better)")
    plt.ylabel("Millimeters (mm)")

    # Add the numerical value on top of each bar
    for index, row in data.iterrows():
        plt.text(index, row.HD95_Global + 0.5, f"{row.HD95_Global:.1f}", color='black', ha="center")

    plt.tight_layout()
    plt.savefig(cfg.graphs_dir / "3_global_hausdorff.svg", format='svg')
    plt.close()

    # Graph 4: Efficiency vs Performance
    fig, ax1 = plt.subplots(figsize=(10, 6))

    sns.scatterplot(data=data, x="Time_s", y="Dice_Global", hue="Model", s=200, palette="tab10", ax=ax1)

    for i in range(data.shape[0]):
        ax1.text(data["Time_s"][i] + 1, data["Dice_Global"][i], 
                f"{data['Model'][i]}\n({data['VRAM_MB'][i]/1024:.1f} GB)", 
                horizontalalignment='left', size='medium', color='black')

    plt.title("Efficiency Analysis: Accuracy vs. Inference Time")
    plt.xlabel("Inference Time per Patient (Seconds)")
    plt.ylabel("Global Dice")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Model", loc='lower right')
    plt.tight_layout()
    plt.savefig(cfg.graphs_dir / "4_efficiency_vs_accuracy.svg", format='svg')
    plt.close()

if __name__ == "__main__":
    main()