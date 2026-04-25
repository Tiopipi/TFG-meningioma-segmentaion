"""Generates and saves learning curves for segmentation models.

This script reads training logs from CSV files, extracts the average
training loss and validation Dice scores, and generates a dual-axis
line plot for each model.
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
            print(f"File not found: {csv_file}. Skipping...")
            continue

        print(f" Generating plots for: {model_name}...")

        df = pd.read_csv(csv_file)
        df['Validation_Dice'] = pd.to_numeric(df['Validation_Dice'], errors='coerce')

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_loss = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Average Training Loss', color=color_loss)
        line_loss, = ax1.plot(df['Epoch'], df['Avg_Loss'], color=color_loss, linewidth=2, label='Avg Loss (Train)')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()  
        color_dice = 'tab:blue'
        ax2.set_ylabel('Dice Coefficient (Validation)', color=color_dice)

        df_val = df.dropna(subset=['Validation_Dice'])
        
        if not df_val.empty:
            line_dice, = ax2.plot(df_val['Epoch'], df_val['Validation_Dice'], color=color_dice, 
                                marker='o', markersize=8, linewidth=2, linestyle='--', label='Dice (Validation)')
            ax2.tick_params(axis='y', labelcolor=color_dice)

            best_epoch = df_val.loc[df_val['Validation_Dice'].idxmax()]
            
            offset_x = 5 if best_epoch['Epoch'] > 5 else -2
            
            ax2.annotate(f'Best Epoch\n({best_epoch["Validation_Dice"]:.4f})', 
                        xy=(best_epoch['Epoch'], best_epoch['Validation_Dice']),
                        xytext=(best_epoch['Epoch'] - offset_x, best_epoch['Validation_Dice'] - 0.02),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=10, weight='bold')

            lines = [line_loss, line_dice]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='center right')
        else:
            ax1.legend([line_loss], [line_loss.get_label()], loc='upper right')

        # 3. Finalization and Export
        plt.title(f'Learning Curves: {model_name}')
        plt.tight_layout()

        clean_filename = model_name.lower().replace(' ', '_').replace('-', '')
        output_path = cfg.graphs_dir / f"learning_curve_{clean_filename}.svg"
        plt.savefig(output_path, format='svg')
        plt.close()

if __name__ == "__main__":
    main()