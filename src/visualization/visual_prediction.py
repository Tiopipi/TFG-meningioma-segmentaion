import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from typing import Dict

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

def find_best_slice(gt_volume: np.ndarray) -> int:
    """Find and return the index of the slice with the largest tumor area.
    """
    sum_per_slice = np.sum(gt_volume > 0, axis=(0, 1))
    best_slice = np.argmax(sum_per_slice)
    
    return int(best_slice)

def visualize_inference(
    patient_idx: int, 
    models_dict: Dict[str, torch.nn.Module], 
    weights_dict: Dict[str, str]
) -> None:
    """Generate and save a visual comparision of model predictions for a specific patient.
    
    Loads the requested patient from the test split, indentifies the axial slice with the
    largest tumor area and runs inference across all models. It creates a figure comparing
    the original T1c MRI, the Ground Truth and the predictions of each model.  

    Args:
        patient_idx: The index of the patient in the dataset.
        models_dict: A dictionary mapping model names to instantiated PyTorch models.
        weights_dict: A dictionary mapping model names to their weights paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    dataset_test = BraTSDataset(split="test")
    patient_data = dataset_test[patient_idx]
    
    img_tensor = patient_data["image"].unsqueeze(0).to(device)
    
    gt_vol = patient_data["label"].numpy()
    if len(gt_vol.shape) == 4:
        gt_vol = gt_vol[0]
        
    mri_vol = patient_data["image"][1].numpy() 

    slice_z = find_best_slice(gt_vol)

    slice_mri = np.rot90(mri_vol[:, :, slice_z])
    slice_gt = np.rot90(gt_vol[:, :, slice_z])

    predictions_2d = {}
    
    with torch.no_grad():
        for name, model in models_dict.items():
            print(f"Processing network: {name}...")
            checkpoint = torch.load(weights_dict[name], map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            
            with torch.amp.autocast('cuda'):
                output = sliding_window_inference(
                    inputs=img_tensor, roi_size=cfg.roi_size, sw_batch_size=4, predictor=model, overlap=0.5
                )
            
            pred_vol = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()
            predictions_2d[name] = np.rot90(pred_vol[:, :, slice_z])

    n_models = len(models_dict)
    
    fig = plt.figure(figsize=(5 * n_models, 10), dpi=300)
    
    gs = gridspec.GridSpec(2, n_models * 2)
    
    tumor_colors = ListedColormap(['none', 'red', 'green', 'yellow'])

    total_cols = n_models * 2
    left_pad = (total_cols - 4) // 2

    ax_mri = fig.add_subplot(gs[0, left_pad : left_pad+2])
    ax_mri.imshow(slice_mri, cmap='gray')
    ax_mri.set_title("Original MRI (T1c)", fontsize=18, pad=20)
    ax_mri.axis('off')

    ax_gt = fig.add_subplot(gs[0, left_pad+2 : left_pad+4])
    ax_gt.imshow(slice_mri, cmap='gray')
    mask_gt = np.ma.masked_where(slice_gt == 0, slice_gt)
    ax_gt.imshow(mask_gt, cmap=tumor_colors, alpha=0.7, vmin=0, vmax=3)
    ax_gt.set_title("Ground Truth", fontsize=18, pad=20)
    ax_gt.axis('off')

    for idx, (name, slice_pred) in enumerate(predictions_2d.items()):
        ax_pred = fig.add_subplot(gs[1, idx*2 : idx*2+2])
        ax_pred.imshow(slice_mri, cmap='gray')
        mask_pred = np.ma.masked_where(slice_pred == 0, slice_pred)
        ax_pred.imshow(mask_pred, cmap=tumor_colors, alpha=0.7, vmin=0, vmax=3)
        ax_pred.set_title(f"Prediction: {name}", fontsize=15, pad=15, fontweight='bold')
        ax_pred.axis('off')

    plt.tight_layout()
    
    os.makedirs(cfg.graphs_dir, exist_ok=True)
    save_path = cfg.graphs_dir / f"segmentation_patient_{patient_idx}.svg"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, format='svg')
    plt.close()
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {
        "U-Net 3D": build_unet3d().to(device),
        "SegResNet": build_segresnet().to(device),
        "SegMamba": build_segmamba().to(device),
        "SegMambaV2": build_segmambav2().to(device),
        "SwinUNETR": build_SwinUNETR().to(device)
    }
    
    weights = {
        "U-Net 3D": cfg.checkpoints_dir / "best_unet3d.pth",
        "SegResNet": cfg.checkpoints_dir / "best_segresnet.pth",
        "SegMamba": cfg.checkpoints_dir / "best_segmamba.pth",
        "SegMambaV2": cfg.checkpoints_dir / "best_segmambav2.pth",
        "SwinUNETR": cfg.checkpoints_dir / "best_swinunetr.pth"
    }
    
    visualize_inference(patient_idx=34, models_dict=models, weights_dict=weights)