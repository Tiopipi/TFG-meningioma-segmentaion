"""Extract qnd plot MRI spatial resolution satistics.

This script reads clinical data from an excel, extract the x/y resolution
and slice thicknes for the different modalities, and genetates a statistical
summary CSV and a set of boxplots.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import configs.config as cfg
from src.utils.clinical_io import load_clinical_data
from src.utils.plot_utils import setup_plotting_environment


def build_resolution_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Restructure the dataframe to stack modalities for easier plotting."""
    records = []
    modalities = ["T1", "T1c", "T2", "FLAIR"]

    for modality in modalities:
        x_col = f"{modality} x-resolution (mm)"
        y_col = f"{modality} y-resolution (mm)"
        thickness_col = f"{modality} Slice Thickness (mm)"

        if all(col in df.columns for col in [x_col, y_col, thickness_col]):
            temp = df[[x_col, y_col, thickness_col]].copy()
            temp.columns = ["x_resolution", "y_resolution", "slice_thickness"]
            temp["modality"] = modality
            records.append(temp)
        else:
            print(f"Warning: Resolution columns for {modality} not found.")

    if not records:
        print("Error: No resolution data could be extracted.")
        sys.exit(1)

    return pd.concat(records, ignore_index=True)


def save_resolution_summary(resolution_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and save a statistical summary of the resolutions."""
    summary = (
        resolution_df
        .groupby("modality")[["x_resolution", "y_resolution", "slice_thickness"]]
        .agg(["mean", "std", "min", "max"])
        .round(3)
    )

    output_path = cfg.data_analysis_dir / "resolution_summary.csv"
    summary.to_csv(output_path)

    return summary


def plot_resolution_boxplots(resolution_df: pd.DataFrame) -> None:
    """Generate and save boxplots for spatial resolution parameters."""
    modalities = ["T1", "T1c", "T2", "FLAIR"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    variables = [
        ("x_resolution", "Resolución en x (mm)"),
        ("y_resolution", "Resolución en y (mm)"),
        ("slice_thickness", "Grosor de corte (mm)"),
    ]

    for ax, (column, title) in zip(axes, variables):
        data = [
            resolution_df.loc[resolution_df["modality"] == modality, column].dropna()
            for modality in modalities
        ]

        ax.boxplot(data, labels=modalities, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel("mm")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = cfg.data_analysis_dir / "resolution_parameters_boxplots.svg"
    plt.savefig(output_path, format="svg")
    plt.close(fig)

def main() -> None:
    # 1. Configuration and Setup
    setup_plotting_environment(
        output_dir=cfg.data_analysis_dir, 
        font_size=13, 
        title_size=16
    )

    # 2. Data Loading
    df = load_clinical_data(cfg.clinical_data)
    
    # 3. Processing
    resolution_df = build_resolution_dataframe(df)

    # 4. Summary & Plotting
    summary = save_resolution_summary(resolution_df)
    
    print("\nSpatial Resolution Summary (mm)")
    print(summary)
    print("-" * 40)

    plot_resolution_boxplots(resolution_df)

if __name__ == "__main__":
    main()