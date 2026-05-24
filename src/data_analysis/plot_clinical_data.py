"""Generate and save clinical demographic analysis plots

This script reads clinical data from an Excel and generates
a chart for sex distribution, a histogram for age distribution
and a bar chat for meningioma tumor grades.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import configs.config as cfg
from src.utils.clinical_io import load_clinical_data
from src.utils.plot_utils import setup_plotting_environment

def main() -> None:
    # 1. Configuration and Setup
    setup_plotting_environment(
        output_dir=cfg.data_analysis_dir, 
        font_size=14, 
        title_size=16
    )

    # 2. Data Loading
    df = load_clinical_data(cfg.clinical_data)

    # 3. Figure 1: Sex and Age Distribution
    fig_combined, (ax_sex, ax_age) = plt.subplots(1, 2, figsize=(16, 8))
    
    sex_series = df["Sex"].replace({"Female": "Mujer", "Male": "Hombre"})
    sex_all = sex_series.value_counts()

    ax_sex.pie(sex_all, labels=sex_all.index, autopct="%1.1f%%", 
               colors=["lightblue", "orange"], startangle=90)
    ax_sex.set_title("Distribución de sexo", fontweight="bold", pad=20)
    
    ax_age.hist(df["Age"], bins=15, color="lightblue", edgecolor="white")
    ax_age.set_title("Distribución de edad", fontweight="bold", pad=20)
    ax_age.set_xlabel("Edad (años)")
    ax_age.set_ylabel("Número de casos")

    plt.tight_layout()
    
    output_path_combined = cfg.data_analysis_dir / "clinical_demographics_combined.svg"
    fig_combined.savefig(output_path_combined, format='svg')
    plt.close(fig_combined)

    # 4. Figure 2: Meningioma Grade
    fig_grade = plt.figure(figsize=(8, 6))

    grade_counts = pd.to_numeric(df["Grade"], errors="coerce").dropna().astype(int).value_counts().sort_index()

    bars = plt.bar([f"Grado {g}" for g in grade_counts.index], grade_counts.values,
                   color=["lightblue", "orange", "green"][:len(grade_counts)], width=0.5)

    plt.title("Grado del meningioma", fontweight="bold", pad=20)
    plt.ylabel("Número de casos")

    for bar, val in zip(bars, grade_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(val),
                 ha="center", va="bottom", fontweight="bold")

    plt.ylim(0, grade_counts.max() * 1.2)

    plt.tight_layout()
    output_path_grade = cfg.data_analysis_dir / "clinical_analysis_grade.svg"
    fig_grade.savefig(output_path_grade, format='svg')
    plt.close(fig_grade)

if __name__ == "__main__":
    main()