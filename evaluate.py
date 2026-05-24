import os
import sys
import torch
import csv
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from src.data.dataset import BraTSDataset
from src.models.unet import build_unet3d
from src.models.segresnet import build_segresnet
from src.models.swinunetr import build_SwinUNETR
from src.models.segmamba import build_segmamba
from src.models.segmambav2 import build_segmambav2

import configs.config as cfg

BRATS_CLASSES = [
    {"name": "Non-enhancing Tumor Core", "acronym": "NETC"},
    {"name": "Surrounding T2/FLAIR Hyperintensity", "acronym": "SNFH"},
    {"name": "Enhancing Tumor", "acronym": "ET"}
]

BRATS_REGIONS = [
    {"name": "Enhancing Tumor", "acronym": "ET"},
    {"name": "Tumor Core", "acronym": "TC"},
    {"name": "Whole Tumor", "acronym": "WT"},
]

def convert_to_brats_regions(onehot_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a one-hot tensor of classes (Background, NETC, SNFH, ET) into BraTS regions.
    Returns a tensor with 3 channels: [ET, TC, WT]
    """
    netc = onehot_tensor[1:2] 
    snfh = onehot_tensor[2:3]  
    et = onehot_tensor[3:4] 
    
    tc = torch.logical_or(netc, et).float()
    
    wt = torch.logical_or(tc, snfh).float()
    
    et_region = et.float()
    
    return torch.cat([et_region, tc, wt], dim=0)

def evaluate_model(
    model_name: str,
    model: torch.nn.Module,
    weights_path: str
) -> None:
    """Evaluate a trained segmentation model on the test dataset.
    
    Calculates Dice, IoU, and 95th percentile Hausdorff Distance (HD95) 
    metrics overall and dynamically per official BraTS subregion.

    Args:
        model_name: The display name of the model being evaluated.
        model: The instantiated PyTorch model architecture.
        weights_path: Path to the saved weights to load.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Evaluating {model_name} ---")
    
    test_loader = DataLoader(BraTSDataset(split="test"), batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    dice_metric_cls = DiceMetric(include_background=False, reduction="mean_batch")
    iou_metric_cls = MeanIoU(include_background=False, reduction="mean_batch")
    hd95_metric_cls = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")

    dice_metric_reg = DiceMetric(include_background=True, reduction="mean_batch")
    iou_metric_reg = MeanIoU(include_background=True, reduction="mean_batch")
    hd95_metric_reg = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
    
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg.num_classes)
    post_label = AsDiscrete(to_onehot=cfg.num_classes)

    inference_times = []
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            
            start_time = time.time()
            with torch.amp.autocast('cuda'):
                outputs = sliding_window_inference(
                    inputs=inputs, roi_size=cfg.roi_size, sw_batch_size=4, predictor=model, overlap=0.5
                )
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            if hasattr(outputs, "as_tensor"): outputs = outputs.as_tensor()
            if hasattr(labels, "as_tensor"): labels = labels.as_tensor()
            
            outputs_list = [post_pred(i) for i in decollate_batch(outputs)]
            labels_list = [post_label(i) for i in decollate_batch(labels)]
            
            dice_metric_cls(y_pred=outputs_list, y=labels_list)
            iou_metric_cls(y_pred=outputs_list, y=labels_list)
            hd95_metric_cls(y_pred=outputs_list, y=labels_list)

            outputs_regions = [convert_to_brats_regions(i) for i in outputs_list]
            labels_regions = [convert_to_brats_regions(i) for i in labels_list]

            dice_metric_reg(y_pred=outputs_regions, y=labels_regions)
            iou_metric_reg(y_pred=outputs_regions, y=labels_regions)
            hd95_metric_reg(y_pred=outputs_regions, y=labels_regions)

    mean_dice_cls = torch.nanmean(dice_metric_cls.aggregate()).item()
    mean_iou_cls = torch.nanmean(iou_metric_cls.aggregate()).item()
    mean_hd95_cls = torch.nanmean(hd95_metric_cls.aggregate()).item()
    dice_list_cls = dice_metric_cls.aggregate().tolist()
    iou_list_cls = iou_metric_cls.aggregate().tolist()
    hd95_list_cls = hd95_metric_cls.aggregate().tolist()

    dice_list_reg = dice_metric_reg.aggregate().tolist()
    iou_list_reg = iou_metric_reg.aggregate().tolist()
    hd95_list_reg = hd95_metric_reg.aggregate().tolist()

    mean_time = sum(inference_times) / len(inference_times)
    max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

    print(f"\n--- Global Results (Classes) ({model_name}) ---")
    print(f"Dice: {mean_dice_cls:.4f} | IoU: {mean_iou_cls:.4f} | HD95: {mean_hd95_cls:.4f} mm")
    print(f"Mean Time: {mean_time:.2f} s | Max VRAM: {max_memory_mb:.2f} MB")
    
    print(f"\n--- Breakdown by Class ---")
    for i, class_info in enumerate(BRATS_CLASSES):
        print(f"{class_info['name']:<38}: Dice {dice_list_cls[i]:.4f} | IoU {iou_list_cls[i]:.4f} | HD95 {hd95_list_cls[i]:.4f} mm")

    print(f"\n--- Breakdown by Region (State of the Art comparison) ---")
    for i, reg_info in enumerate(BRATS_REGIONS):
        print(f"{reg_info['name']:<38}: Dice {dice_list_reg[i]:.4f} | IoU {iou_list_reg[i]:.4f} | HD95 {hd95_list_reg[i]:.4f} mm")
    print("\n")
    
    os.makedirs(cfg.logs_dir, exist_ok=True)
    csv_file = cfg.logs_dir / f"evaluation_results_{model_name.lower().replace(' ', '_')}.csv"
    file_exists = csv_file.exists()
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = ["Modelo", "Dice_Global"] + \
                      [f"Dice_CLS_{c['acronym']}" for c in BRATS_CLASSES] + \
                      [f"Dice_REG_{r['acronym']}" for r in BRATS_REGIONS] + \
                      ["IoU_Global"] + \
                      [f"IoU_CLS_{c['acronym']}" for c in BRATS_CLASSES] + \
                      [f"IoU_REG_{r['acronym']}" for r in BRATS_REGIONS] + \
                      ["HD95_Global"] + \
                      [f"HD95_CLS_{c['acronym']}" for c in BRATS_CLASSES] + \
                      [f"HD95_REG_{r['acronym']}" for r in BRATS_REGIONS] + \
                      ["Time_s", "VRAM_MB"]
            writer.writerow(headers)
        
        row_data = [model_name, f"{mean_dice_cls:.4f}"] + \
                   [f"{val:.4f}" for val in dice_list_cls] + \
                   [f"{val:.4f}" for val in dice_list_reg] + \
                   [f"{mean_iou_cls:.4f}"] + \
                   [f"{val:.4f}" for val in iou_list_cls] + \
                   [f"{val:.4f}" for val in iou_list_reg] + \
                   [f"{mean_hd95_cls:.4f}"] + \
                   [f"{val:.4f}" for val in hd95_list_cls] + \
                   [f"{val:.4f}" for val in hd95_list_reg] + \
                   [f"{mean_time:.2f}", f"{max_memory_mb:.2f}"]
        
        writer.writerow(row_data)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unet = build_unet3d().to(device)
    evaluate_model("U-Net 3D", unet, cfg.checkpoints_dir / "best_unet3d.pth")
    del unet
    torch.cuda.empty_cache()
    
    segresnet = build_segresnet().to(device)
    evaluate_model("SegResNet", segresnet, cfg.checkpoints_dir / "best_segresnet.pth")
    del segresnet
    torch.cuda.empty_cache()
    
    swin = build_SwinUNETR().to(device)
    evaluate_model("SwinUNETR", swin, cfg.checkpoints_dir / "best_swinunetr.pth")
    del swin
    torch.cuda.empty_cache()

    mamba = build_segmamba().to(device)
    evaluate_model("SegMamba", mamba, cfg.checkpoints_dir / "best_segmamba.pth")
    del mamba
    torch.cuda.empty_cache()
    
    mambav2 = build_segmambav2().to(device)
    evaluate_model("SegMambaV2", mambav2, cfg.checkpoints_dir / "best_segmambav2.pth")
    del mambav2
    torch.cuda.empty_cache()