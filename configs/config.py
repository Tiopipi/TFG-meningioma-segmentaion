"""Global configuration file for the project

Centralizes all file paths, dataset splits, architectural parameters,
and training hypermarameters.
"""

import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

# Input data
data_dir = base_dir / "data"
train_data_dir = data_dir / "brats-train-val-2023" / "BraTS-MEN-Train"
splits_file = data_dir / "splits.json"

# Output directories
checkpoints_dir = base_dir / "checkpoints"
logs_dir = base_dir / "logs"
graphs_dir = base_dir / "graphs"
cache_dir = base_dir / "cache_brats"

# Seed and Data Splits 
seed = 33

train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# Modalities list
modalities = ["t1n", "t1c", "t2w", "t2f"]

# Number of classes: 0 (Background), 1 (NCR), 2 (SNFH), 3 (ET)
num_classes = 4 

# Standardized spatial size for random crops and sliding window inference
roi_size = (128, 128, 128) 
val_spatial_size = (240, 240, 160)

# Training Hyperparameters
batch_size = 1
num_workers = 4
max_epochs = 50
learning_rate = 1e-4
val_interval = 5
patience = 2