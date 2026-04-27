"""
Training script for the U-Net shoreline extraction baseline.

This script loads the shoreline dataset, trains the U-Net model, records
training metrics, saves the last and best weights, and exports prediction
examples from the best checkpoint.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import tqdm

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[1]
print("script_dir:", CURRENT_DIR)
print("root_dir:", ROOT_DIR)

sys.path.append(str(ROOT_DIR))

from data import MyDataset
from models.unet import UNet


# -----------------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", DEVICE)

NUM_CLASSES = 1
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

RESULT_DIR = ROOT_DIR / "results" / "unet"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_PATH = RESULT_DIR / "unet_last.pth"
BEST_WEIGHT_PATH = RESULT_DIR / "unet_best.pth"

BEST_PRED_DIR = RESULT_DIR / "best_epoch_predictions"
COMPARE_DIR = RESULT_DIR / "best_epoch_comparisons"
METRICS_DIR = RESULT_DIR / "metrics"

for directory in (BEST_PRED_DIR, COMPARE_DIR, METRICS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Data, model, loss, and optimizer
# -----------------------------------------------------------------------------
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

model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)

if WEIGHT_PATH.exists():
    state = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    print("Loaded previous last checkpoint.")
else:
    print("No previous checkpoint found. Training starts from scratch.")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def calculate_binary_metrics(preds: torch.Tensor, masks: torch.Tensor) -> dict:
    """Calculate PA, recall, precision, F1, and IoU for binary segmentation."""
    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    correct = (preds_flat == masks_flat).sum().item()
    total = masks_flat.numel()

    tp = ((preds_flat == 1) & (masks_flat == 1)).sum().item()
    fp = ((preds_flat == 1) & (masks_flat == 0)).sum().item()
    fn = ((preds_flat == 0) & (masks_flat == 1)).sum().item()

    pa = correct / (total + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn + 1e-6)

    return {
        "PA": pa,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "IoU": iou,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Correct": correct,
        "Total": total,
    }


def train(model: nn.Module, data_loader: DataLoader, criterion, optimizer, num_epochs: int = 100):
    """Train the U-Net model and save training logs and checkpoints."""
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

        for step, (images, masks) in enumerate(tqdm.tqdm(data_loader)):
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
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                batch_metrics = calculate_binary_metrics(preds, masks)

                total_pixels += batch_metrics["Total"]
                total_correct += batch_metrics["Correct"]
                total_tp += batch_metrics["TP"]
                total_fp += batch_metrics["FP"]
                total_fn += batch_metrics["FN"]

        epoch_loss = running_loss / len(data_loader.dataset)
        pa = total_correct / (total_pixels + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        precision = total_tp / (total_tp + total_fp + 1e-6) if (total_tp + total_fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0
        iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"loss={epoch_loss:.6f} PA={pa:.4f} Recall={recall:.4f} "
            f"Precision={precision:.4f} F1={f1:.4f} IoU={iou:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "PA": pa,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "IoU": iou,
        })

        torch.save(model.state_dict(), WEIGHT_PATH)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_WEIGHT_PATH)
            print(f"Updated best model at epoch {best_epoch} with loss={best_loss:.6f}")

    save_training_logs(history, best_epoch, best_loss)
    print(f"Training finished. Best epoch = {best_epoch}, best loss = {best_loss:.6f}")
    return best_epoch, history


def save_training_logs(history: list, best_epoch: int, best_loss: float) -> None:
    """Save metric history, metric curve, and best-epoch information."""
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
    precisions = [item["Precision"] for item in history]
    f1s = [item["F1"] for item in history]
    ious = [item["IoU"] for item in history]

    plt.figure()
    plt.plot(epochs, losses, label="Loss")
    plt.plot(epochs, pas, label="PA")
    plt.plot(epochs, recalls, label="Recall")
    plt.plot(epochs, precisions, label="Precision")
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
        for item in history:
            if item["epoch"] == best_epoch:
                file.write(f"PA={item['PA']:.6f}\n")
                file.write(f"Recall={item['Recall']:.6f}\n")
                file.write(f"Precision={item['Precision']:.6f}\n")
                file.write(f"F1={item['F1']:.6f}\n")
                file.write(f"IoU={item['IoU']:.6f}\n")
                break


def export_best_predictions(model: nn.Module, dataset: MyDataset, best_weight_path: Path) -> None:
    """Export binary predictions and image-label-prediction comparison panels."""
    if not best_weight_path.exists():
        print("Best checkpoint not found. Prediction export skipped.")
        return

    print("Loading the best model for prediction export...")
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
                stem = dataset.names[index]
                name = f"{stem}.png"
            else:
                name = f"{index:05d}.png"

            inputs = image.unsqueeze(0).to(DEVICE)
            logits = model(inputs)
            probability = torch.sigmoid(logits)[0, 0]
            prediction = (probability > 0.5).float().cpu()

            save_image(prediction, BEST_PRED_DIR / name)

            visual_image = image.clone()
            if visual_image.shape[0] == 1:
                visual_image = visual_image.repeat(3, 1, 1)

            visual_mask = mask.clone()
            if visual_mask.dim() == 3 and visual_mask.shape[0] == 1:
                visual_mask = visual_mask.repeat(3, 1, 1)
            elif visual_mask.dim() == 2:
                visual_mask = visual_mask.unsqueeze(0).repeat(3, 1, 1)

            visual_prediction = prediction.unsqueeze(0).repeat(3, 1, 1)
            comparison = torch.cat([visual_image, visual_mask, visual_prediction], dim=2)
            save_image(comparison, COMPARE_DIR / name)

    print("Best predictions and comparison images have been saved.")


if __name__ == "__main__":
    best_epoch, history = train(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)
    export_best_predictions(model, train_dataset, BEST_WEIGHT_PATH)
