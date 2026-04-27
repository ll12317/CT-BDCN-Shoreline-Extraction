# -*- coding: utf-8 -*-
"""
Training and prediction export script for the baseline CNN-Transformer model.

This script trains ConvTransformerSegNet for binary shoreline segmentation,
saves model checkpoints, records training metrics, and exports predictions
from the best epoch.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# The expected location of this file is:
# CT-BDCN-Shoreline-Extraction/scripts/train_main_models/train_ct.py
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from data import MyDataset
from models.ct import ConvTransformerSegNet


# -----------------------------
# Basic configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 1
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4

# Dataset paths. Change "masks" to "labels" if your label folders use that name.
TRAIN_IMG_DIR = ROOT_DIR / "datasets" / "train" / "images"
TRAIN_MASK_DIR = ROOT_DIR / "datasets" / "train" / "masks"

# Output paths.
RESULT_DIR = ROOT_DIR / "results" / "ct"
PRED_DIR = RESULT_DIR / "best_epoch_predictions"
COMPARE_DIR = RESULT_DIR / "best_epoch_comparisons"
METRICS_DIR = RESULT_DIR / "metrics"

LAST_WEIGHT_PATH = RESULT_DIR / "ct_last.pth"
BEST_WEIGHT_PATH = RESULT_DIR / "ct_best.pth"


def make_output_dirs() -> None:
    """Create output directories if they do not exist."""
    for directory in (RESULT_DIR, PRED_DIR, COMPARE_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_dataloader() -> Tuple[MyDataset, DataLoader]:
    """Build the training dataset and dataloader."""
    train_dataset = MyDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        num_classes=NUM_CLASSES,
        size=IMAGE_SIZE,
        augment=True,
        mode="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    return train_dataset, train_loader


def build_model() -> ConvTransformerSegNet:
    """Build the baseline CNN-Transformer model."""
    model = ConvTransformerSegNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        embed_dim=128,
        num_heads=4,
        transformer_layers=2,
    )
    return model.to(DEVICE)


def load_last_weight_if_available(model: nn.Module) -> None:
    """Resume training from the last checkpoint if it exists."""
    if LAST_WEIGHT_PATH.exists():
        state_dict = torch.load(LAST_WEIGHT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {LAST_WEIGHT_PATH}")
    else:
        print("No previous checkpoint found. Training starts from scratch.")


def calculate_binary_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> Tuple[int, int, int, int, int]:
    """Calculate pixel-level statistics for binary segmentation metrics."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    correct = (preds_flat == masks_flat).sum().item()
    pixels = masks_flat.numel()

    tp = ((preds_flat == 1) & (masks_flat == 1)).sum().item()
    fp = ((preds_flat == 1) & (masks_flat == 0)).sum().item()
    fn = ((preds_flat == 0) & (masks_flat == 1)).sum().item()

    return correct, pixels, tp, fp, fn


def summarize_metrics(
    total_correct: int,
    total_pixels: int,
    total_tp: int,
    total_fp: int,
    total_fn: int,
) -> Dict[str, float]:
    """Convert accumulated pixel statistics into PA, Recall, Precision, F1, and IoU."""
    pa = total_correct / (total_pixels + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    precision = total_tp / (total_tp + total_fp + 1e-6) if (total_tp + total_fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)

    return {
        "PA": pa,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "IoU": iou,
    }


def train(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = NUM_EPOCHS,
) -> Tuple[int, List[Dict[str, float]]]:
    """Train the model and save the best checkpoint according to training loss."""
    history = []
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        total_pixels = 0
        total_correct = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for step, (images, masks) in enumerate(tqdm(data_loader)):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print(f"epoch={epoch + 1}, step={step}, train_loss={loss.item():.6f}")

            running_loss += loss.item() * images.size(0)

            with torch.no_grad():
                correct, pixels, tp, fp, fn = calculate_binary_metrics(logits, masks)
                total_correct += correct
                total_pixels += pixels
                total_tp += tp
                total_fp += fp
                total_fn += fn

        epoch_loss = running_loss / len(data_loader.dataset)
        metric_values = summarize_metrics(
            total_correct=total_correct,
            total_pixels=total_pixels,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"loss={epoch_loss:.6f} "
            f"PA={metric_values['PA']:.4f} "
            f"Recall={metric_values['Recall']:.4f} "
            f"F1={metric_values['F1']:.4f} "
            f"IoU={metric_values['IoU']:.4f}"
        )

        history_item = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            **metric_values,
        }
        history.append(history_item)

        torch.save(model.state_dict(), LAST_WEIGHT_PATH)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_WEIGHT_PATH)
            print(f"Updated best model at epoch {best_epoch} with loss={best_loss:.6f}")

    save_training_logs(history, best_epoch)
    print(f"Training completed. Best epoch = {best_epoch}, best loss = {best_loss:.6f}")

    return best_epoch, history


def save_training_logs(history: List[Dict[str, float]], best_epoch: int) -> None:
    """Save training metrics, curves, and best-epoch information."""
    log_path = METRICS_DIR / "metrics_history.txt"
    with open(log_path, "w", encoding="utf-8") as file:
        file.write("epoch,loss,PA,Recall,Precision,F1,IoU\n")
        for item in history:
            file.write(
                f"{item['epoch']},{item['loss']:.6f},"
                f"{item['PA']:.6f},{item['Recall']:.6f},"
                f"{item['Precision']:.6f},{item['F1']:.6f},"
                f"{item['IoU']:.6f}\n"
            )

    epochs = [item["epoch"] for item in history]
    losses = [item["loss"] for item in history]
    pas = [item["PA"] for item in history]
    recalls = [item["Recall"] for item in history]
    f1s = [item["F1"] for item in history]
    ious = [item["IoU"] for item in history]

    plt.figure()
    plt.plot(epochs, losses, label="Loss")
    plt.plot(epochs, pas, label="PA")
    plt.plot(epochs, recalls, label="Recall")
    plt.plot(epochs, f1s, label="F1")
    plt.plot(epochs, ious, label="IoU")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "metrics_curve.png")
    plt.close()

    best_info_path = METRICS_DIR / "best_epoch.txt"
    with open(best_info_path, "w", encoding="utf-8") as file:
        file.write(f"best_epoch={best_epoch}\n")
        for item in history:
            if item["epoch"] == best_epoch:
                file.write(f"loss={item['loss']:.6f}\n")
                file.write(f"PA={item['PA']:.6f}\n")
                file.write(f"Recall={item['Recall']:.6f}\n")
                file.write(f"Precision={item['Precision']:.6f}\n")
                file.write(f"F1={item['F1']:.6f}\n")
                file.write(f"IoU={item['IoU']:.6f}\n")
                break


def export_best_predictions(model: nn.Module, dataset: MyDataset) -> None:
    """Export prediction masks and side-by-side comparison images using the best checkpoint."""
    if not BEST_WEIGHT_PATH.exists():
        print("Best checkpoint not found. Prediction export is skipped.")
        return

    print("Loading the best checkpoint for prediction export...")
    state_dict = torch.load(BEST_WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    if hasattr(dataset, "augment"):
        dataset.augment = False

    with torch.no_grad():
        for index in range(len(dataset)):
            image, mask = dataset[index]

            if hasattr(dataset, "samples"):
                image_path, _ = dataset.samples[index]
                name = Path(image_path).name
            elif hasattr(dataset, "names"):
                name = f"{dataset.names[index]}.png"
            else:
                name = f"{index:05d}.png"

            input_tensor = image.unsqueeze(0).to(DEVICE)
            logits = model(input_tensor)
            prob = torch.sigmoid(logits)[0, 0]
            pred = (prob > 0.5).float().cpu()

            save_image(pred, PRED_DIR / name)

            vis_image = image.clone()
            if vis_image.shape[0] == 1:
                vis_image = vis_image.repeat(3, 1, 1)

            vis_mask = mask.clone()
            if vis_mask.dim() == 3 and vis_mask.shape[0] == 1:
                vis_mask = vis_mask.repeat(3, 1, 1)
            elif vis_mask.dim() == 2:
                vis_mask = vis_mask.unsqueeze(0).repeat(3, 1, 1)

            vis_pred = pred.unsqueeze(0).repeat(3, 1, 1)
            comparison = torch.cat([vis_image, vis_mask, vis_pred], dim=2)

            save_image(comparison, COMPARE_DIR / name)

    print("Prediction masks and comparison images have been saved.")


def main() -> None:
    """Run model training and prediction export."""
    make_output_dirs()

    print(f"Project root: {ROOT_DIR}")
    print(f"Using device: {DEVICE}")

    train_dataset, train_loader = build_dataloader()
    model = build_model()
    load_last_weight_if_available(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
    )

    export_best_predictions(model, train_dataset)


if __name__ == "__main__":
    main()
