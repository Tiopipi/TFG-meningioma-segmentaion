import os
import sys
import torch
import csv
from tqdm import tqdm
import time
from torch.utils.data import DataLoader

from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataset import BraTSDataset
from src.models.unet import build_unet3d
from src.models.segresnet import build_segresnet
from src.models.swinunetr import build_SwinUNETR
from src.models.segmamba import build_segmamba

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")
    
    max_epochs = 50
    val_interval = 5
    patience = 10
    epochs_without_improvement = 0
    best_metric = -1
    
    csv_file = "registro_metricas_segresnet.csv"
    model_file = "best_segresnet.pth"
    
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Avg_Loss", "Learning_Rate", "Time_s", "Max_VRAM_MB", "Validation_Dice"])
    
    train_loader = DataLoader(BraTSDataset(split="train"), batch_size=1, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(BraTSDataset(split="val"), batch_size=1, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    model = build_segmamba().to(device)
    
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    scaler = torch.amp.GradScaler('cuda')

    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    for epoch in range(max_epochs):
        print(f"\n--- Epoch {epoch + 1}/{max_epochs} ---")
        
        epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        model.train()
        epoch_loss = 0
        step = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            step += 1
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        max_memory_mb = 0
        if torch.cuda.is_available():
            max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
        print(f"Average loss: {epoch_loss/step:.4f} | Current LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Time: {epoch_time:.2f} s | Max VRAM: {max_memory_mb:.2f} MB")
        
        scheduler.step()

        dice_csv = "N/A"

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validation"):
                    val_inputs, val_labels = val_batch["image"].to(device), val_batch["label"].to(device)
                    
                    with torch.amp.autocast('cuda'):
                        val_outputs = sliding_window_inference(
                            inputs=val_inputs, roi_size=(128, 128, 128), sw_batch_size=4, predictor=model, overlap=0.5
                        )
                    
                    if hasattr(val_outputs, "as_tensor"):
                        val_outputs = val_outputs.as_tensor()
                    if hasattr(val_labels, "as_tensor"):
                        val_labels = val_labels.as_tensor()
                    
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                print(f"Validation Dice score: {metric:.4f}")
                
                dice_csv = f"{metric:.4f}"

                if metric > best_metric:
                    best_metric = metric
                    epochs_without_improvement = 0
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_metric": best_metric,
                    }, model_file)
                    print("New best model saved.")
                else:
                    epochs_without_improvement += val_interval
                    print(f"No improvement for {epochs_without_improvement} epochs.")
                    
                    if epochs_without_improvement >= patience:
                        print("\n--- EARLY STOPPING ---")
                        print("Stopping training to prevent overfitting.")
                        
                        with open(csv_file, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([epoch + 1, f"{epoch_loss/step:.4f}", f"{scheduler.get_last_lr()[0]:.6f}", f"{epoch_time:.2f}", f"{max_memory_mb:.2f}", dice_csv])
                        return

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{epoch_loss/step:.4f}", 
                f"{scheduler.get_last_lr()[0]:.6f}", 
                f"{epoch_time:.2f}", 
                f"{max_memory_mb:.2f}", 
                dice_csv
            ])

if __name__ == "__main__":
    train_model()