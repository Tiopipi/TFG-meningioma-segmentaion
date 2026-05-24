import os
import sys
import torch
import numpy as np
from typing import Dict
import nibabel as nib

import warnings
warnings.filterwarnings("ignore")

from monai.inferers import sliding_window_inference

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset import BraTSDataset
from src.models.unet import build_unet3d
from src.models.segmambav2 import build_segmambav2
from src.models.segresnet import build_segresnet
from src.models.swinunetr import build_SwinUNETR
from src.models.segmamba import build_segmamba

import configs.config as cfg


def export_inference_nifti(
    patient_idx: int, 
    models_dict: Dict[str, torch.nn.Module], 
    weights_dict: Dict[str, str]
) -> None:
    """Run inference on a specific patient and export results as NIfTI files.

    This function isolates the T1c modality and the ground truth, runs 
    sliding window inference for each provided model, and saves all volumes 
    using an identity affine matrix for perfect alignment in medical viewers.

    Args:
        patient_idx: Index of the patient in the test dataset.
        models_dict: Dictionary mapping model names to their PyTorch architectures.
        weights_dict: Dictionary mapping model names to their checkpoint paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_test = BraTSDataset(split="test")
    patient_data = dataset_test[patient_idx]
    
    img_tensor = patient_data["image"].unsqueeze(0).to(device)
    
    gt_vol = patient_data["label"].numpy().astype(np.uint8)
    if len(gt_vol.shape) == 4:
        gt_vol = gt_vol[0]
        
    mri_vol = patient_data["image"][1].numpy() 
    
    export_dir = cfg.graphs_dir / "nifti_exports" / f"patient_{patient_idx}"
    os.makedirs(export_dir, exist_ok=True)
    
    affine = np.eye(4)
        
    nib.save(nib.Nifti1Image(mri_vol, affine), export_dir / "00_MRI_T1c.nii.gz")
    nib.save(nib.Nifti1Image(gt_vol, affine), export_dir / "01_Ground_Truth.nii.gz")
    
    with torch.no_grad():
        for name, model in models_dict.items():
            print(f"Generating NIfTI for: {name}...")
            checkpoint = torch.load(weights_dict[name], map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            
            with torch.amp.autocast('cuda'):
                output = sliding_window_inference(
                    inputs=img_tensor, roi_size=cfg.roi_size, sw_batch_size=4, predictor=model, overlap=0.5
                )
            
            pred_vol = torch.argmax(output.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)
            
            clean_name = name.replace(" ", "_").replace("-", "")
            nib.save(nib.Nifti1Image(pred_vol, affine), export_dir / f"02_Pred_{clean_name}.nii.gz")    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {
        "U-Net 3D": build_unet3d().to(device),
        #"SegResNet": build_segresnet().to(device),
        "SegMamba": build_segmamba().to(device),
        "SegMambaV2": build_segmambav2().to(device),
        #"SwinUNETR": build_SwinUNETR().to(device)
    }
    
    weights = {
        "U-Net 3D": cfg.checkpoints_dir / "best_unet3d.pth",
        #"SegResNet": cfg.checkpoints_dir / "best_segresnet.pth",
        "SegMamba": cfg.checkpoints_dir / "best_segmamba.pth",
        "SegMambaV2": cfg.checkpoints_dir / "best_segmambav2.pth",
        #"SwinUNETR": cfg.checkpoints_dir / "best_swinunetr.pth"
    }
    
    export_inference_nifti(patient_idx=34, models_dict=models, weights_dict=weights)