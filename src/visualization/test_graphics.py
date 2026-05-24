"""Generates comparative evaluation plots for segmentation models

This script takes the evaluation results from multiple segmentation
models and generates figures including overall performance, 
subregion-specific Dice scores, Hausdorff Distance comparisons, an 
efficiency analysis, and a comparison with state of the art.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.lines import Line2D
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
    df_melted_global = data.melt(id_vars="Modelo", value_vars=["Dice_Global", "IoU_Global"], 
                                var_name="Metric", value_name="Score")

    sns.barplot(data=df_melted_global, x="Modelo", y="Score", hue="Metric", palette="viridis")
    plt.title("Rendimiento general de la segmentación")
    plt.ylim(0, 1)
    plt.ylabel("Puntuación (0 - 1)")
    plt.legend(title="Métrica")
    plt.tight_layout()
    plt.savefig(cfg.graphs_dir / "1_overall_performance.svg", format='svg')
    plt.close()

    # Graph 2: Dice Breakdown by Subregions
    plt.figure(figsize=(10, 6))
    df_melted_dice = data.melt(id_vars="Modelo", value_vars=["Dice_CLS_NETC", "Dice_CLS_SNFH", "Dice_CLS_ET"], 
                            var_name="Subregion", value_name="Dice")

    df_melted_dice['Subregion'] = df_melted_dice['Subregion'].str.replace('Dice_CLS_', '')

    sns.barplot(data=df_melted_dice, x="Modelo", y="Dice", hue="Subregion", palette="mako")
    plt.title("Coeficiente Dice por subregión del tumor")
    plt.ylim(0, 1)
    plt.ylabel("Coeficiente Dice")

    plt.legend(title="Subregión (NETC, SNFH, ET)", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(cfg.graphs_dir / "2_dice_by_class.svg", format='svg')
    plt.close()

    # Graph 3: Hausdorff Distance 
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x="Modelo", y="HD95_Global", palette="rocket")
    plt.title("Distancia global de Hausdorff al 95% (Menor es mejor)")
    plt.ylabel("Milímetros (mm)")

    for index, row in data.iterrows():
        plt.text(index, row.HD95_Global + 0.5, f"{row.HD95_Global:.1f}", color='black', ha="center")

    plt.tight_layout()
    plt.savefig(cfg.graphs_dir / "3_global_hausdorff.svg", format='svg')
    plt.close()

    # Graph 4: Efficiency vs Performance (3D Scatter Plot)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    models = data["Modelo"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    color_dict = dict(zip(models, colors))

    def draw_3d_balloon(ax, x, y, area, base_color):
        base_rgb = np.array(mcolors.to_rgb(base_color))
        white = np.array([1.0, 1.0, 1.0])
        
        num_layers = 20
        for j in range(num_layers):
            frac = (j / num_layers) ** 2 
            
            layer_color = base_rgb * (1 - frac) + white * frac
            
            layer_size = area * (1 - j / num_layers)
            
            ax.scatter(x, y, s=layer_size, color=layer_color, edgecolor='none', zorder=2)
            
        ax.scatter(x, y, s=area, facecolors='none', edgecolors='black', linewidth=0.8, alpha=0.6, zorder=3)

    min_vram, max_vram = data["VRAM_MB"].min(), data["VRAM_MB"].max()
    min_area, max_area = 300, 3500 

    for i in range(data.shape[0]):
        current_model = data["Modelo"][i]
        x_val = data["Time_s"][i]
        y_val = data["Dice_Global"][i]
        vram_val = data["VRAM_MB"][i]
        
        if max_vram > min_vram:
            area = min_area + (vram_val - min_vram) / (max_vram - min_vram) * (max_area - min_area)
        else:
            area = 1000

        draw_3d_balloon(ax1, x_val, y_val, area, color_dict[current_model])
        
        ax1.text(x_val + 1.5, y_val, 
                f"{current_model}\n({vram_val/1024:.1f} GB)", 
                size=9, color='black', weight='bold', zorder=4)

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[m], 
                              markersize=10, markeredgecolor='black', label=m) for m in models]
    ax1.legend(handles=legend_elements, title="Modelos", loc='lower right')

    ax1.set_title("Análisis de eficiencia: Precisión vs Tiempo inferencia", pad=15)
    ax1.set_xlabel("Tiempo de inferencia (Segundos)")
    ax1.set_ylabel("Coeficiente Dice global")
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)

    x_min, x_max = data["Time_s"].min(), data["Time_s"].max()
    y_min, y_max = data["Dice_Global"].min(), data["Dice_Global"].max()
    
    x_range = x_max - x_min if x_max > x_min else 10
    y_range = y_max - y_min if y_max > y_min else 0.1

    ax1.set_xlim(x_min - (x_range * 0.10), x_max + (x_range * 0.30))
    
    ax1.set_ylim(y_min - (y_range * 0.15), y_max + (y_range * 0.15))

    plt.tight_layout()
    plt.savefig(cfg.graphs_dir / "4_efficiency_vs_accuracy.svg", format='svg')
    plt.close()

    reg_dice_cols = [col for col in data.columns if "Dice_REG_" in col]
    reg_hd95_cols = [col for col in data.columns if "HD95_REG_" in col]

    # Results from state of the art
    sota_data = {
        "Modelo": "Auto3DSeg (SOTA)",
        "Dice_REG_ET": 0.8985,
        "Dice_REG_TC": 0.9035,
        "Dice_REG_WT": 0.8709,
        "HD95_REG_ET": 23.86,
        "HD95_REG_TC": 21.82,
        "HD95_REG_WT": 31.39
    }
    sota_df = pd.DataFrame([sota_data])
    
    data_sota = pd.concat([data, sota_df], ignore_index=True)

    # Graph 5: Dice Breakdown by Regions (WT, TC, ET)
    plt.figure(figsize=(10, 6))
    df_melted_dice_reg = data_sota.melt(id_vars="Modelo", value_vars=reg_dice_cols, 
                            var_name="Region", value_name="Dice")
    df_melted_dice_reg['Region'] = df_melted_dice_reg['Region'].str.replace('Dice_REG_', '')
    sns.barplot(data=df_melted_dice_reg, x="Modelo", y="Dice", hue="Region", palette="crest")
    plt.title("Coeficiente Dice por región estándar frente al Estado del Arte")
    plt.ylim(0, 1)
    plt.ylabel("Coeficiente Dice")
    plt.legend(title="Regiones (SOTA)", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(cfg.graphs_dir / "5_dice_by_region_sota.svg", format='svg')
    plt.close()

    # Graph 6: HD95 Breakdown by Regions (WT, TC, ET)
    plt.figure(figsize=(10, 6))
    df_melted_hd95_reg = data_sota.melt(id_vars="Modelo", value_vars=reg_hd95_cols, 
                            var_name="Region", value_name="HD95")
    df_melted_hd95_reg['Region'] = df_melted_hd95_reg['Region'].str.replace('HD95_REG_', '')
    sns.barplot(data=df_melted_hd95_reg, x="Modelo", y="HD95", hue="Region", palette="flare")
    plt.title("Distancia Hausdorff 95% por región estándar frente al Estado del Arte")
    plt.ylabel("Milímetros (mm)")
    plt.legend(title="Regiones (SOTA)", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(cfg.graphs_dir / "6_hd95_by_region_sota.svg", format='svg')
    plt.close()

if __name__ == "__main__":
    main()