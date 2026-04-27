"""
Training script for the DeepLabV3+ baseline model.

This script trains DeepLabV3+ for binary shoreline segmentation, saves the last
and best model weights, exports prediction maps from the best checkpoint, and
records training metrics for comparison experiments.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import tqdm

# Resolve the repository root when this file is placed under scripts/train_main_models/.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from data import MyDataset
from models.deeplabv3plus import DeepLabV3Plus


# -----------------------------
# Basic settings
# -----------------------------
MODEL_NAME = "deeplabv3plus"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4
NUM_CLASSES = 1
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Repository root: {ROOT_DIR}")
print(f"Using device: {DEVICE}")


# -----------------------------
# Output paths
# -----------------------------
SAVE_DIR = ROOT_DIR / "results" / MODEL_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_PATH = SAVE_DIR / f"{MODEL_NAME}_last.pth"
BEST_WEIGHT_PATH = SAVE_DIR / f"{MODEL_NAME}_best.pth"

BEST_PRED_DIR = SAVE_DIR / "best_epoch_predictions"
COMPARE_DIR = SAVE_DIR / "best_epoch_comparisons"
METRICS_DIR = SAVE_DIR / "metrics"

for output_dir in (BEST_PRED_DIR, COMPARE_DIR, METRICS_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Dataset and dataloader
# -----------------------------
train_dataset = MyDataset(
    mode="train",
    size=IMAGE_SIZE,
    augment=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)


# -----------------------------
# Model, loss, and optimizer
# -----------------------------
model = DeepLabV3Plus(
    num_classes=NUM_CLASSES,
    backbone="resnet50",
    pretrained_backbone=False,
).to(DEVICE)

if WEIGHT_PATH.exists():
    state = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    print("Loaded the last checkpoint successfully.")
else:
    print("No existing checkpoint was found. Training will start from scratch.")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def calculate_batch_statistics(
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> Tuple[int, int, int, int, int]:
    """Calculate pixel-level statistics for binary segmentation metrics."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    correct = (preds_flat == masks_flat).sum().item()
    total_pixels = masks_flat.numel()
    tp = ((preds_flat == 1) & (masks_flat == 1)).sum().item()
    fp = ((preds_flat == 1) & (masks_flat == 0)).sum().item()
    fn = ((preds_flat == 0) & (masks_flat == 1)).sum().item()

    return correct, total_pixels, tp, fp, fn


def calculate_metrics(
    total_correct: int,
    total_pixels: int,
    total_tp: int,
    total_fp: int,
    total_fn: int,
) -> Dict[str, float]:
    """Convert accumulated confusion statistics into evaluation metrics."""
    pa = total_correct / (total_pixels + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    precision = total_tp / (total_tp + total_fp + 1e-6) if (total_tp + total_fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)

    return {
        "PA": pa,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
    }


def train(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 100,
) -> Tuple[int, List[Dict[str, float]]]:
    """Train the model and save the best checkpoint based on training loss."""
    history: List[Dict[str, float]] = []
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

        for step, (images, masks) in enumerate(tqdm.tqdm(data_loader)):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print(f"Epoch {epoch + 1}, step {step}, train_loss={loss.item():.6f}")

            running_loss += loss.item() * images.size(0)

            with torch.no_grad():
                correct, pixels, tp, fp, fn = calculate_batch_statistics(logits, masks)
                total_correct += correct
                total_pixels += pixels
                total_tp += tp
                total_fp += fp
                total_fn += fn

        epoch_loss = running_loss / len(data_loader.dataset)
        metrics = calculate_metrics(total_correct, total_pixels, total_tp, total_fp, total_fn)

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"loss={epoch_loss:.6f} PA={metrics['PA']:.4f} "
            f"Recall={metrics['Recall']:.4f} F1={metrics['F1']:.4f} "
            f"IoU={metrics['IoU']:.4f}"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "PA": metrics["PA"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "IoU": metrics["IoU"],
        }
        history.append(epoch_record)

        torch.save(model.state_dict(), WEIGHT_PATH)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_WEIGHT_PATH)
            print(f"Updated best model at epoch {best_epoch} with loss={best_loss:.6f}")

    save_training_history(history, best_epoch, best_loss)
    print(f"Training completed. Best epoch = {best_epoch}, best loss = {best_loss:.6f}")
    return best_epoch, history


def save_training_history(
    history: List[Dict[str, float]],
    best_epoch: int,
    best_loss: float,
) -> None:
    """Save metric logs, the best-epoch summary, and the training curve."""
    log_path = METRICS_DIR / "metrics_history.txt"
    with open(log_path, "w", encoding="utf-8") as file:
        file.write("epoch,loss,PA,Precision,Recall,F1,IoU\n")
        for record in history:
            file.write(
                f"{record['epoch']},{record['loss']:.6f},"
                f"{record['PA']:.6f},{record['Precision']:.6f},"
                f"{record['Recall']:.6f},{record['F1']:.6f},"
                f"{record['IoU']:.6f}\n"
            )

    epochs = [record["epoch"] for record in history]
    losses = [record["loss"] for record in history]
    pas = [record["PA"] for record in history]
    recalls = [record["Recall"] for record in history]
    f1s = [record["F1"] for record in history]
    ious = [record["IoU"] for record in history]

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
        file.write(f"best_loss={best_loss:.6f}\n")
        for record in history:
            if record["epoch"] == best_epoch:
                file.write(f"PA={record['PA']:.6f}\n")
                file.write(f"Precision={record['Precision']:.6f}\n")
                file.write(f"Recall={record['Recall']:.6f}\n")
                file.write(f"F1={record['F1']:.6f}\n")
                file.write(f"IoU={record['IoU']:.6f}\n")
                break


def export_best_predictions(
    model: nn.Module,
    dataset: MyDataset,
    best_weight_path: Path,
) -> None:
    """Export prediction maps and image-label-prediction comparison panels."""
    if not best_weight_path.exists():
        print("Best weight file not found. Prediction export is skipped.")
        return

    print("Loading the best model for prediction export.")
    state = torch.load(best_weight_path, map_location=DEVICE)
    model.load_state_dict(state)
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
            probability = torch.sigmoid(logits)[0, 0]
            prediction = (probability > 0.5).float().cpu()

            save_image(prediction, BEST_PRED_DIR / name)

            visible_image = image.clone()
            if visible_image.shape[0] == 1:
                visible_image = visible_image.repeat(3, 1, 1)

            visible_mask = mask.clone()
            if visible_mask.dim() == 3 and visible_mask.shape[0] == 1:
                visible_mask = visible_mask.repeat(3, 1, 1)
            elif visible_mask.dim() == 2:
                visible_mask = visible_mask.unsqueeze(0).repeat(3, 1, 1)

            visible_prediction = prediction.unsqueeze(0).repeat(3, 1, 1)
            comparison_panel = torch.cat(
                [visible_image, visible_mask, visible_prediction],
                dim=2,
            )
            save_image(comparison_panel, COMPARE_DIR / name)

    print("Best predictions and comparison images were saved successfully.")


if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)
    export_best_predictions(model, train_dataset, BEST_WEIGHT_PATH)
