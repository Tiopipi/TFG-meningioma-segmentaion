"""Extract and verify MRI spatial dimensions and voxel spacings.

This script iterates through the training and validation splits to load 
all NIfTI files and extract their spatial shapes and voxel resolutions. 
It verifies consistency across all modalities for each patient and checks 
that the segmentation masks align perfectly with the anatomical MRI scans.
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import configs.config as cfg
from src.utils.nifti_io import load_volume

def main() -> None:
    # 1. Configuration and Setup
    splits = {
        "train": cfg.train_data_dir,
        "validation": getattr(cfg, "val_data_dir", None),
    }

    modalities_val = cfg.modalities
    modalities_train = cfg.modalities + ["seg"]

    records = []

    # 2. Data Processing Loop
    for split_name, split_dir in splits.items():
        if split_dir is None or not split_dir.exists():
            print(f"  [Warning] The directory for {split_name} does not exist at {split_dir}")
            continue

        print(f"  Scanning {split_name} split")
        modalities = modalities_train if split_name == "train" else modalities_val

        for case_dir in sorted(split_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            case_id = case_dir.name
            shapes = {}
            spacings = {}

            for mod in modalities:
                try:
                    data, header = load_volume(case_dir, mod, return_header=True)
                    shapes[mod] = data.shape
                    spacings[mod] = tuple(np.round(header.get_zooms(), 4))
                except FileNotFoundError:
                    shapes[mod] = None
                    spacings[mod] = None

            image_mods = [m for m in modalities if m != "seg"]
            unique_shapes = set(v for m, v in shapes.items() if m in image_mods and v is not None)
            unique_spacings = set(v for m, v in spacings.items() if m in image_mods and v is not None)

            record = {
                "case_id": case_id,
                "split": split_name,
                **{f"shape_{m}": shapes[m] for m in modalities},
                **{f"spacing_{m}": spacings[m] for m in modalities},
                "shapes_consistent": len(unique_shapes) == 1,
                "spacings_consistent": len(unique_spacings) == 1,
            }

            if split_name == "train":
                record["seg_matches_image"] = shapes.get("seg") == shapes.get("t1c")

            records.append(record)

    # 3. Statistical Analysis & Reporting
    df = pd.DataFrame(records)

    for split in ["train", "validation"]:
        sub = df[df["split"] == split]
        if sub.empty:
            continue
            
        print(f"\n{split.capitalize()} Set ({len(sub)} cases)")
        
        print("\nShape consistency (all modalities have the same dimensions):")
        print(sub["shapes_consistent"].value_counts().to_string())
        
        print("\nSpacing consistency (all modalities have the same resolution):")
        print(sub["spacings_consistent"].value_counts().to_string())
        
        print("\nMost common T1c shapes:")
        print(sub["shape_t1c"].value_counts().to_string())
        
        print("\nMost common T1c spacings:")
        print(sub["spacing_t1c"].value_counts().to_string())
        
        if split == "train":
            print("\nDoes segmentation mask (seg) match T1c image?:")
            print(sub["seg_matches_image"].value_counts().to_string())

    anomalies = df[
        ~df["shapes_consistent"] |
        ~df["spacings_consistent"] |
        df.get("seg_matches_image", pd.Series(True, index=df.index)).eq(False)
    ]

    print("\nAnomalies Found")
    if not anomalies.empty:
        df_display = anomalies[["case_id", "split", "shape_t1c", "spacing_t1c", "shapes_consistent"]].copy()
        print(df_display.to_string(index=False))
    else:
        print("No anomalies found. All cases are consistent.")

if __name__ == "__main__":
    main()