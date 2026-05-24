"""Plotting and visualization configuration utilities.

This module standardizes the visual appearance of all generated plots.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def setup_plotting_environment(output_dir: Path, font_size: int = 14, title_size: int = 16, use_seaborn: bool = False) -> None:
    """Configure matplotlib/seaborn defaults and create the output directory.
    
    Args:
        output_dir: Path object pointing to the directory for saving graphs.
        font_size: Base font size for ticks and axis labels.
        title_size: Font size for plot titles.
        use_seaborn: If True, applies the Seaborn 'whitegrid' theme.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if use_seaborn:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': title_size,
        'figure.dpi': 300
    })