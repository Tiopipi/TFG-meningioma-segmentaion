import os
import json
import sys
from typing import Callable, Optional
from monai.data import PersistentDataset
from src.data.transforms import get_train_transforms, get_val_transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import configs.config as cfg

class BraTSDataset:
    """Factory class to create a MONAI PersistentDataset for the  BraTS challenge."""
    
    def __new__(
        cls,
        split: str ="train",
        transform: Optional[Callable] = None
    ) -> PersistentDataset:
        """Creates and returns a PersistentDataset for the specified data split.
        
        Loads the dataset splits from a JSON file, constructs file paths and maps them
        to "image" and "label" keys. It applies the specified transformations and caches
        the processed data locally for faster reloading.

        Args:
            split: The dataset split to load. Defaults to "train".
            transform: Custom transformation pipeline. If None,
            defaults to pre-defined train or validation transforms based on the split.

        Raises:
            FileNotFoundError: If the configuration splits JSON file is not found.
            ValueError: If the requested split is not present in the splits file

        Returns:
           A MONAI dataset instance containing the configured
           data dictionaries, transformations and caching mechanism.
        """
        if not cfg.splits_file.exists():
            raise FileNotFoundError(f"{cfg.splits_file} not found")
            
        with open(cfg.splits_file, 'r') as f:
            splits = json.load(f)
            
        if split not in splits:
            raise ValueError(f"Split '{split}' does not exist")
            
        case_ids = splits[split]
        data_dir = cfg.train_data_dir

        data_dicts = []
        
        for case_id in case_ids:
            case_path = data_dir / case_id
            
            img_paths = [str(case_path / f"{case_id}-{mod}.nii.gz") for mod in cfg.modalities]
            seg_path = str(case_path / f"{case_id}-seg.nii.gz")
            
            data_dicts.append({
                "image": img_paths,
                "label": seg_path,
                "id": case_id
            })

        if transform is None:
            if split == "train":
                transform_pipeline = get_train_transforms()
            else:
                transform_pipeline = get_val_transforms()
        else:
            transform_pipeline = transform

        cache_dir = cfg.cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        return PersistentDataset(
            data=data_dicts,
            transform=transform_pipeline,
            cache_dir=cache_dir
        )