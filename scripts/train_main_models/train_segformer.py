"""
Training script for the SegFormer baseline.

This script trains the SegFormer-style shoreline extraction model, saves
the latest and best model weights, records training metrics, and exports
prediction examples using the best checkpoint.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from data import MyDataset
from models.segformer import SegFormerCoast


# -----------------------------
# Device configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# Path configuration
# -----------------------------
TRAIN_IMAGE_DIR = ROOT_DIR / "datasets" / "train" / "images"
TRAIN_MASK_DIR = ROOT_DIR / "datasets" / "train" / "masks"

SAVE_DIR = ROOT_DIR / "results" / "segformer"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_PATH = SAVE_DIR / "segformer_last.pth"
BEST_WEIGHT_PATH = SAVE_DIR / "segformer_best.pth"

BEST_PRED_DIR = SAVE_DIR / "best_epoch_predictions"
COMPARE_DIR = SAVE_DIR / "best_epoch_comparisons"
METRICS_DIR = SAVE_DIR / "metrics"

for directory in (BEST_PRED_DIR, COMPARE_DIR, METRICS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Training configuration
# -----------------------------
NUM_CLASSES = 2
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4
NUM_WORKERS = 0


def build_dataloader() -> tuple:
    """Build the training dataset and data loader."""
    train_dataset = MyDataset(
        image_dir=str(TRAIN_IMAGE_DIR),
        mask_dir=str(TRAIN_MASK_DIR),
        num_classes=1,
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


def build_model() -> nn.Module:
    """Create the SegFormer baseline model."""
    model = SegFormerCoast(
        num_classes=NUM_CLASSES,
        in_channels=3,
    )
    return model.to(device)


def load_last_checkpoint(model: nn.Module) -> None:
    """Load the latest checkpoint if it exists."""
    if WEIGHT_PATH.exists():
        state = torch.load(WEIGHT_PATH, map_location=device)
        model.load_state_dict(state)
        print("Last checkpoint loaded successfully.")
    else:
        print("No existing checkpoint was found. Training will start from scratch.")


def calculate_binary_metrics(preds: torch.Tensor, masks: torch.Tensor) -> dict:
    """Calculate binary segmentation metrics from prediction and mask tensors."""
    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    total_correct = (preds_flat == masks_flat).sum().item()
    total_pixels = masks_flat.numel()

    tp = ((preds_flat == 1) & (masks_flat == 1)).sum().item()
    fp = ((preds_flat == 1) & (masks_flat == 0)).sum().item()
    fn = ((preds_flat == 0) & (masks_flat == 1)).sum().item()

    return {
        "correct": total_correct,
        "pixels": total_pixels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def summarize_epoch_metrics(
    total_correct: int,
    total_pixels: int,
    total_tp: int,
    total_fp: int,
    total_fn: int,
) -> dict:
    """Summarize pixel-level metrics for one epoch."""
    pa = total_correct / (total_pixels + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    precision = (
        total_tp / (total_tp + total_fp + 1e-6)
        if (total_tp + total_fp) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall + 1e-6)
        if (precision + recall) > 0
        else 0.0
    )
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
) -> tuple:
    """Train the model and save training logs and checkpoints."""
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

        for batch_index, (images, masks) in enumerate(tqdm.tqdm(data_loader)):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            logits_fg = logits[:, 1:2, ...]
            loss = criterion(logits_fg, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 5 == 0:
                print(
                    f"Epoch {epoch + 1}, batch {batch_index}, "
                    f"loss={loss.item():.6f}"
                )

            running_loss += loss.item() * images.size(0)

            with torch.no_grad():
                probs = torch.sigmoid(logits_fg)
                preds = (probs > 0.5).float()
                batch_metrics = calculate_binary_metrics(preds, masks)

                total_correct += batch_metrics["correct"]
                total_pixels += batch_metrics["pixels"]
                total_tp += batch_metrics["tp"]
                total_fp += batch_metrics["fp"]
                total_fn += batch_metrics["fn"]

        epoch_loss = running_loss / len(data_loader.dataset)
        metrics = summarize_epoch_metrics(
            total_correct,
            total_pixels,
            total_tp,
            total_fp,
            total_fn,
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"loss={epoch_loss:.6f} "
            f"PA={metrics['PA']:.4f} "
            f"Recall={metrics['Recall']:.4f} "
            f"F1={metrics['F1']:.4f} "
            f"IoU={metrics['IoU']:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "PA": metrics["PA"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "IoU": metrics["IoU"],
        })

        torch.save(model.state_dict(), WEIGHT_PATH)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_WEIGHT_PATH)
            print(
                f"Best model updated at epoch {best_epoch} "
                f"(loss={best_loss:.6f})."
            )

    save_training_logs(history, best_epoch)
    print(
        f"Training finished. Best epoch = {best_epoch}, "
        f"best loss = {best_loss:.6f}."
    )

    return best_epoch, history


def save_training_logs(history: list, best_epoch: int) -> None:
    """Save training metrics and curves."""
    log_path = METRICS_DIR / "metrics_history.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,loss,PA,Precision,Recall,F1,IoU\n")
        for item in history:
            f.write(
                f"{item['epoch']},{item['loss']:.6f},"
                f"{item['PA']:.6f},{item['Precision']:.6f},"
                f"{item['Recall']:.6f},{item['F1']:.6f},"
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
    with open(best_info_path, "w", encoding="utf-8") as f:
        f.write(f"best_epoch={best_epoch}\n")
        for item in history:
            if item["epoch"] == best_epoch:
                f.write(f"loss={item['loss']:.6f}\n")
                f.write(f"PA={item['PA']:.6f}\n")
                f.write(f"Precision={item['Precision']:.6f}\n")
                f.write(f"Recall={item['Recall']:.6f}\n")
                f.write(f"F1={item['F1']:.6f}\n")
                f.write(f"IoU={item['IoU']:.6f}\n")
                break


def export_best_predictions(
    model: nn.Module,
    dataset: MyDataset,
    best_weight_path: Path,
) -> None:
    """Export prediction masks and input-label-prediction comparison images."""
    if not best_weight_path.exists():
        print("Best checkpoint was not found. Prediction export was skipped.")
        return

    print("Loading the best model for prediction export.")
    state = torch.load(best_weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
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

            input_tensor = image.unsqueeze(0).to(device)
            outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            logits_fg = logits[:, 1:2, ...]
            prob = torch.sigmoid(logits_fg)[0, 0]
            pred = (prob > 0.5).float().cpu()

            save_image(pred, BEST_PRED_DIR / name)

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

    print("Prediction masks and comparison images were saved.")


if __name__ == "__main__":
    train_dataset, train_loader = build_dataloader()
    model = build_model()
    load_last_checkpoint(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_epoch, history = train(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
    )

    export_best_predictions(model, train_dataset, BEST_WEIGHT_PATH)
