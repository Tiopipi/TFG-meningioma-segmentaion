"""Clinical data input/output utilities."""

import os
import sys
from pathlib import Path

import pandas as pd


def load_clinical_data(filepath: Path) -> pd.DataFrame:
    """Load the clinical data Excel file after verifying its existence.
    
    Args:
        filepath: Path object pointing to the clinical Excel file.
        
    Returns:
        A pandas DataFrame containing the clinical metadata.
        
    Raises:
        SystemExit: If the target file is not found on disk.
    """
    if not os.path.exists(filepath):
        print(f"Error: Clinical data file not found at {filepath}")
        sys.exit(1)
        
    return pd.read_excel(filepath)