import os
import json
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from configs.config import train_data_dir, splits_file, seed, train_ratio, val_ratio

def generate_splits() -> None:
    """Generates train, validation and test splits grouped by patient ID.
    
    Scans the training data directory and groups the cases by patient ID to prevent data leakage 
    across datasets, ensuring all cases from a single patient remain in the same split. The
    patients are shuffled into training, validation and testing sets, saving the resulting 
    ID lists in a JSON file. 
    """
    if not train_data_dir.exists():
        print(f"Error: Folder not found in {train_data_dir}")
        return
    
    case_ids = []
    
    for case in train_data_dir.iterdir():
        if case.is_dir():
            case_ids.append(case.name)

    patients_dict = {}
    
    for case in case_ids:
        clean_case = case.split('.')[0]
        patient_id = clean_case.rsplit('-', 1)[0]
        
        if patient_id not in patients_dict:
            patients_dict[patient_id] = []
            
        patients_dict[patient_id].append(case)
        
    patients = list(patients_dict.keys())    
    patients.sort()
    total_patients = len(patients)

    random.seed(seed)
    random.shuffle(patients)
    
    train_cases = int(total_patients * train_ratio)
    val_cases = train_cases + int(total_patients * val_ratio)
    
    train_patients = patients[:train_cases]
    val_patients = patients[train_cases:val_cases]
    test_patients = patients[val_cases:]
    
    train_ids = []
    for p in train_patients:
        train_ids.extend(patients_dict[p])
        
    val_ids = []
    for p in val_patients:
        val_ids.extend(patients_dict[p])
        
    test_ids = []
    for p in test_patients:
        test_ids.extend(patients_dict[p])
    
    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }
    
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=4)

if __name__ == "__main__":    
    generate_splits()