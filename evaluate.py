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
    {"name": "Non-enhancing Tumor Core", "acronym": "NCR"},
    {"name": "Surrounding T2/FLAIR Hyperintensity", "acronym": "SNFH"},
    {"name": "Enhancing Tumor", "acronym": "ET"}
]

def evaluate_model(
    model_name: str,
    model: torch.nn.Module,
    weights_path: str
) -> None:
    """Evaluate a trained segmentation model on the test dataset.
    
    Calculates Dice, IoU, and 95th percentile Hausdorff Distance (HD95) 
    metrics overall and dynamically per official BraTS subregion..

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
    
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    iou_metric = MeanIoU(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")
    
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
            
            dice_metric(y_pred=outputs_list, y=labels_list)
            iou_metric(y_pred=outputs_list, y=labels_list)
            hd95_metric(y_pred=outputs_list, y=labels_list)

    dice_per_class = dice_metric.aggregate()
    iou_per_class = iou_metric.aggregate()
    hd95_per_class = hd95_metric.aggregate()

    mean_dice = torch.nanmean(dice_per_class).item()
    mean_iou = torch.nanmean(iou_per_class).item()
    mean_hd95 = torch.nanmean(hd95_per_class).item()

    dice_list = dice_per_class.tolist()
    iou_list = iou_per_class.tolist()
    hd95_list = hd95_per_class.tolist()

    mean_time = sum(inference_times) / len(inference_times)
    max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

    print(f"\n--- Global Results ({model_name}) ---")
    print(f"Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f} | HD95: {mean_hd95:.4f} mm")
    print(f"Mean Time: {mean_time:.2f} s | Max VRAM: {max_memory_mb:.2f} MB")
    
    print(f"\n--- Breakdown by Subregion ---")
    for i, class_info in enumerate(BRATS_CLASSES):
        name = class_info["name"]
        print(f"{name:<38}: Dice {dice_list[i]:.4f} | IoU {iou_list[i]:.4f} | HD95 {hd95_list[i]:.4f} mm")
    print("\n")
    
    os.makedirs(cfg.logs_dir, exist_ok=True)
    csv_file = cfg.logs_dir / f"evaluation_results_{model_name.lower().replace(' ', '_')}.csv"
    file_exists = csv_file.exists()
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = ["Model", "Dice_Global"] + [f"Dice_{c['acronym']}" for c in BRATS_CLASSES] + \
                      ["IoU_Global"] + [f"IoU_{c['acronym']}" for c in BRATS_CLASSES] + \
                      ["HD95_Global"] + [f"HD95_{c['acronym']}" for c in BRATS_CLASSES] + \
                      ["Time_s", "VRAM_MB"]
            writer.writerow(headers)
        
        row_data = [model_name, f"{mean_dice:.4f}"] + [f"{val:.4f}" for val in dice_list] + \
                   [f"{mean_iou:.4f}"] + [f"{val:.4f}" for val in iou_list] + \
                   [f"{mean_hd95:.4f}"] + [f"{val:.4f}" for val in hd95_list] + \
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