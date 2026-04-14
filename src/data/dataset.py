import os
import json
import sys
from monai.data import PersistentDataset
from src.data.transforms import get_train_transforms, get_val_transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from configs.config import train_data_dir, splits_file

class BraTSDataset:
    def __new__(cls, split="train", transform=None):
        if not splits_file.exists():
            raise FileNotFoundError(f"{splits_file} not found")
            
        with open(splits_file, 'r') as f:
            splits = json.load(f)
            
        if split not in splits:
            raise ValueError(f"Split '{split}' does not exist")
            
        case_ids = splits[split]
        data_dir = train_data_dir

        data_dicts = []
        modalities = ["t1n", "t1c", "t2w", "t2f"]
        
        for case_id in case_ids:
            case_path = data_dir / case_id
            
            img_paths = [str(case_path / f"{case_id}-{mod}.nii.gz") for mod in modalities]
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

        cache_dir = os.path.join(os.getcwd(), "cache_brats")
        os.makedirs(cache_dir, exist_ok=True)

        return PersistentDataset(
            data=data_dicts,
            transform=transform_pipeline,
            cache_dir=cache_dir
        )