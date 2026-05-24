"""Generate and save learning curve plots for segmentation models.

This script reads training logs from CSV files, calculates the validation 
loss derived from the Dice scores (1.0 - Dice), and generates a single-axis 
line plot comparing average training and validation loss.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import configs.config as cfg

def main() -> None:
    # 1. Configuration and Setup
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["figure.dpi"] = 300

    os.makedirs(cfg.graphs_dir, exist_ok=True)

    models_to_process = {
        cfg.logs_dir / "training_unet3d.csv": "U-Net 3D",
        cfg.logs_dir / "training_segresnet.csv": "SegResNet",
        cfg.logs_dir / "training_swinunetr.csv": "Swin UNETR",
        cfg.logs_dir / "training_segmamba.csv": "SegMamba",
        cfg.logs_dir / "training_segmambav2.csv": "SegMamba V2"
    }

    # 2. Data Processing and Plotting Loop
    for csv_file, model_name in models_to_process.items():
        if not csv_file.exists():
            print(f"File not found: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        df['Validation_Dice'] = pd.to_numeric(df['Validation_Dice'], errors='coerce')

        df['Validation_Loss'] = 1.0 - df['Validation_Dice']

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_xlabel('Época')
        ax.set_ylabel('Valor de Pérdida / Error')

        color_loss = 'tab:red'
        ax.plot(df['Epoch'], df['Avg_Loss'], color=color_loss, linewidth=2, label='DiceLoss (Entrenamiento)')

        color_dice = 'tab:blue'
        df_val = df.dropna(subset=['Validation_Dice'])
        
        if not df_val.empty:
            ax.plot(df_val['Epoch'], df_val['Validation_Loss'], color=color_dice, 
                    marker='o', markersize=8, linewidth=2, linestyle='--', label='1 - Coeficiente Dice (Validación)')

            best_epoch = df_val.loc[df_val['Validation_Loss'].idxmin()]
            
            offset_x = 5 if best_epoch['Epoch'] > 5 else -2
            offset_y = best_epoch['Validation_Loss'] + (best_epoch['Validation_Loss'] * 0.1)
            
            ax.annotate(f'Mejor época\n({best_epoch["Validation_Loss"]:.4f})', 
                        xy=(best_epoch['Epoch'], best_epoch['Validation_Loss']),
                        xytext=(best_epoch['Epoch'] - offset_x, offset_y),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=10, weight='bold')

        ax.legend(loc='upper right')

        # 3. Finalization and Export
        plt.title(f'Curvas de aprendizaje: {model_name}')
        plt.tight_layout()

        clean_filename = model_name.lower().replace(' ', '_').replace('-', '')
        output_path = cfg.graphs_dir / f"learning_curve_{clean_filename}.svg"
        plt.savefig(output_path, format='svg')
        plt.close()

if __name__ == "__main__":
    main()