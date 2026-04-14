import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
train_data_dir = data_dir / "brats-train-val-2023" / "BraTS-MEN-Train"
splits_file = data_dir / "splits.json"

seed = 33

train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2