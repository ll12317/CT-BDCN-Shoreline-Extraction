# -*- coding: utf-8 -*-
"""
Training and validation script for CT-BDCN shoreline extraction.

This script trains the CT-BDCN model, saves the latest and best weights,
exports prediction examples, and reports both region-level metrics and
morphology-based boundary-band metrics.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import MyDataset
from models.ct_bdcn import EdgeEnhancedSegNet_UNetBDCN


# -------------------------
# Basic configuration
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 1
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4
THRESHOLD = 0.5
BOUNDARY_BAND_K = 5

LAMBDA_EDGE = 0.2
LAMBDA_SIDE = 0.3
GAMMA_EDGE_IOU = 1.0
BOUNDARY_WEIGHT_ALPHA = 2.0

TRAIN_IMAGE_DIR = ROOT_DIR / "datasets" / "train" / "images"
TRAIN_MASK_DIR = ROOT_DIR / "datasets" / "train" / "masks"
VAL_IMAGE_DIR = ROOT_DIR / "datasets" / "val" / "images"
VAL_MASK_DIR = ROOT_DIR / "datasets" / "val" / "masks"

RESULT_DIR = ROOT_DIR / "results" / "ct_bdcn"
PRED_DIR = RESULT_DIR / "best_epoch_predictions"
COMPARE_DIR = RESULT_DIR / "best_epoch_comparisons"
METRICS_DIR = RESULT_DIR / "metrics"

LAST_WEIGHT_PATH = RESULT_DIR / "ct_bdcn_last.pth"
BEST_WEIGHT_PATH = RESULT_DIR / "ct_bdcn_best.pth"

for directory in (RESULT_DIR, PRED_DIR, COMPARE_DIR, METRICS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# -------------------------
# Boundary-band utilities
# -------------------------
def get_edge_band(gt_mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Construct a boundary band from a binary mask by dilation and erosion."""
    pad = k // 2
    dilate = F.max_pool2d(gt_mask, kernel_size=k, stride=1, padding=pad)
    erode = 1.0 - F.max_pool2d(1.0 - gt_mask, kernel_size=k, stride=1, padding=pad)
    band = (dilate - erode) > 0.5
    return band.float()


def edge_iou_loss(seg_logits: torch.Tensor, masks: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Calculate soft IoU loss within the morphology-based boundary band."""
    probs = torch.sigmoid(seg_logits)
    with torch.no_grad():
        band = get_edge_band(masks, k=k)

    pred_edge = probs * band
    gt_edge = masks * band

    intersection = (pred_edge * gt_edge).sum()
    union = (pred_edge + gt_edge - pred_edge * gt_edge).sum() + 1e-6
    iou_edge = intersection / union
    return 1.0 - iou_edge


def update_binary_metric_counts(preds: torch.Tensor, masks: torch.Tensor, counts: dict) -> None:
    """Update accumulated binary segmentation counts."""
    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    counts["correct"] += (preds_flat == masks_flat).sum().item()
    counts["pixels"] += masks_flat.numel()
    counts["tp"] += ((preds_flat == 1) & (masks_flat == 1)).sum().item()
    counts["fp"] += ((preds_flat == 1) & (masks_flat == 0)).sum().item()
    counts["fn"] += ((preds_flat == 0) & (masks_flat == 1)).sum().item()


def calculate_metrics(counts: dict) -> dict:
    """Calculate PA, recall, precision, F1, and IoU from accumulated counts."""
    pa = counts["correct"] / (counts["pixels"] + 1e-6)
    recall = counts["tp"] / (counts["tp"] + counts["fn"] + 1e-6)
    precision = counts["tp"] / (counts["tp"] + counts["fp"] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = counts["tp"] / (counts["tp"] + counts["fp"] + counts["fn"] + 1e-6)
    return {
        "PA": pa,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "IoU": iou,
    }


def empty_metric_counts() -> dict:
    """Create an empty metric counter dictionary."""
    return {"pixels": 0, "correct": 0, "tp": 0, "fp": 0, "fn": 0}


# -------------------------
# Training and validation
# -------------------------
def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion_edge: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
) -> tuple[float, dict]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    counts = empty_metric_counts()

    for step, (images, masks) in enumerate(tqdm.tqdm(data_loader, desc=f"Train {epoch}")):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        seg_logits, edge_logits = model(images)
        edge_gt = get_edge_band(masks, k=BOUNDARY_BAND_K)

        weight_map = 1.0 + BOUNDARY_WEIGHT_ALPHA * edge_gt
        loss_seg = F.binary_cross_entropy_with_logits(seg_logits, masks, weight=weight_map)
        loss_edge_main = criterion_edge(edge_logits, edge_gt)

        loss_side = torch.tensor(0.0, device=DEVICE)
        side_maps = getattr(model, "_last_side_maps", None)
        if side_maps is not None:
            edge_gt_half = F.interpolate(edge_gt, size=side_maps[0].shape[-2:], mode="nearest")
            for side_map in side_maps:
                loss_side = loss_side + criterion_edge(side_map, edge_gt_half)

        loss_edge_total = loss_edge_main + LAMBDA_SIDE * loss_side
        loss_edge_iou = edge_iou_loss(seg_logits, masks, k=BOUNDARY_BAND_K)
        loss = loss_seg + LAMBDA_EDGE * loss_edge_total + GAMMA_EDGE_IOU * loss_edge_iou

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(
                f"epoch={epoch}, step={step}, loss={loss.item():.6f}, "
                f"seg={loss_seg.item():.6f}, edge_main={loss_edge_main.item():.6f}, "
                f"edge_side={loss_side.item():.6f}, edge_iou_loss={loss_edge_iou.item():.6f}"
            )

        running_loss += loss.item() * images.size(0)

        with torch.no_grad():
            probs = torch.sigmoid(seg_logits)
            preds = (probs > THRESHOLD).float()
            update_binary_metric_counts(preds, masks, counts)

    epoch_loss = running_loss / len(data_loader.dataset)
    metrics = calculate_metrics(counts)
    return epoch_loss, metrics


def validate(model: nn.Module, data_loader: DataLoader, criterion_edge: nn.Module) -> tuple[float, dict]:
    """Validate the model on the validation dataset."""
    model.eval()
    running_loss = 0.0
    counts = empty_metric_counts()

    with torch.no_grad():
        for images, masks in tqdm.tqdm(data_loader, desc="Validate"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            seg_logits, edge_logits = model(images)
            edge_gt = get_edge_band(masks, k=BOUNDARY_BAND_K)

            loss_seg = F.binary_cross_entropy_with_logits(seg_logits, masks)
            loss_edge_main = criterion_edge(edge_logits, edge_gt)
            loss_edge_iou = edge_iou_loss(seg_logits, masks, k=BOUNDARY_BAND_K)
            loss = loss_seg + LAMBDA_EDGE * loss_edge_main + GAMMA_EDGE_IOU * loss_edge_iou

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(seg_logits)
            preds = (probs > THRESHOLD).float()
            update_binary_metric_counts(preds, masks, counts)

    val_loss = running_loss / len(data_loader.dataset)
    metrics = calculate_metrics(counts)
    return val_loss, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion_edge: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
) -> tuple[int, list[dict]]:
    """Run the full training process and save the best model by validation loss."""
    history = []
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion_edge,
            optimizer,
            epoch,
        )
        val_loss, val_metrics = validate(model, val_loader, criterion_edge)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"val_PA={val_metrics['PA']:.4f}, val_Recall={val_metrics['Recall']:.4f}, "
            f"val_Precision={val_metrics['Precision']:.4f}, val_F1={val_metrics['F1']:.4f}, "
            f"val_IoU={val_metrics['IoU']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_PA": train_metrics["PA"],
                "train_Recall": train_metrics["Recall"],
                "train_Precision": train_metrics["Precision"],
                "train_F1": train_metrics["F1"],
                "train_IoU": train_metrics["IoU"],
                "val_PA": val_metrics["PA"],
                "val_Recall": val_metrics["Recall"],
                "val_Precision": val_metrics["Precision"],
                "val_F1": val_metrics["F1"],
                "val_IoU": val_metrics["IoU"],
            }
        )

        torch.save(model.state_dict(), LAST_WEIGHT_PATH)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_WEIGHT_PATH)
            print(f"Updated best model at epoch {best_epoch} with val_loss={best_loss:.6f}")

    save_training_history(history, best_epoch)
    print(f"Training finished. Best epoch={best_epoch}, best val_loss={best_loss:.6f}")
    return best_epoch, history


# -------------------------
# Logging and visualization
# -------------------------
def save_training_history(history: list[dict], best_epoch: int) -> None:
    """Save metric history, training curves, and best-epoch information."""
    log_path = METRICS_DIR / "metrics_history.txt"
    header = (
        "epoch,train_loss,val_loss,train_PA,train_Recall,train_Precision,"
        "train_F1,train_IoU,val_PA,val_Recall,val_Precision,val_F1,val_IoU\n"
    )
    with open(log_path, "w", encoding="utf-8") as file:
        file.write(header)
        for item in history:
            file.write(
                f"{item['epoch']},{item['train_loss']:.6f},{item['val_loss']:.6f},"
                f"{item['train_PA']:.6f},{item['train_Recall']:.6f},{item['train_Precision']:.6f},"
                f"{item['train_F1']:.6f},{item['train_IoU']:.6f},"
                f"{item['val_PA']:.6f},{item['val_Recall']:.6f},{item['val_Precision']:.6f},"
                f"{item['val_F1']:.6f},{item['val_IoU']:.6f}\n"
            )

    epochs = [item["epoch"] for item in history]
    plt.figure()
    plt.plot(epochs, [item["train_loss"] for item in history], label="Train loss")
    plt.plot(epochs, [item["val_loss"] for item in history], label="Validation loss")
    plt.plot(epochs, [item["val_PA"] for item in history], label="Validation PA")
    plt.plot(epochs, [item["val_Recall"] for item in history], label="Validation Recall")
    plt.plot(epochs, [item["val_F1"] for item in history], label="Validation F1")
    plt.plot(epochs, [item["val_IoU"] for item in history], label="Validation IoU")
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
                for key, value in item.items():
                    file.write(f"{key}={value}\n")
                break


# -------------------------
# Prediction export and evaluation
# -------------------------
def export_best_predictions(model: nn.Module, dataset: MyDataset, best_weight_path: Path) -> None:
    """Export binary prediction masks and side-by-side comparison images."""
    if not best_weight_path.exists():
        print("Best weight file was not found. Prediction export skipped.")
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

            inp = image.unsqueeze(0).to(DEVICE)
            seg_logits, _ = model(inp)
            prob = torch.sigmoid(seg_logits)[0, 0]
            pred = (prob > THRESHOLD).float().cpu()

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

    print("Prediction masks and comparison images were saved.")


def evaluate_boundary_metrics(model: nn.Module, dataset: MyDataset, best_weight_path: Path, k: int = 5) -> dict:
    """Evaluate region-level and boundary-band metrics using the best model."""
    if not best_weight_path.exists():
        print("Best weight file was not found. Evaluation skipped.")
        return {}

    print("Loading the best model for evaluation.")
    state = torch.load(best_weight_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    if hasattr(dataset, "augment"):
        dataset.augment = False

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    region_counts = empty_metric_counts()
    edge_counts = empty_metric_counts()

    with torch.no_grad():
        for images, masks in tqdm.tqdm(loader, desc="Evaluate"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            seg_logits, _ = model(images)
            probs = torch.sigmoid(seg_logits)
            preds = (probs > THRESHOLD).float()

            update_binary_metric_counts(preds, masks, region_counts)

            band = get_edge_band(masks, k=k)
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)
            band_flat = band.view(-1)

            valid = band_flat == 1
            if valid.sum().item() == 0:
                continue

            preds_edge = preds_flat[valid]
            masks_edge = masks_flat[valid]
            update_binary_metric_counts(preds_edge, masks_edge, edge_counts)

    region_metrics = calculate_metrics(region_counts)
    edge_metrics = calculate_metrics(edge_counts)

    results = {
        **region_metrics,
        "PA_edge": edge_metrics["PA"],
        "Recall_edge": edge_metrics["Recall"],
        "Precision_edge": edge_metrics["Precision"],
        "F1_edge": edge_metrics["F1"],
        "IoU_edge": edge_metrics["IoU"],
    }

    print(f"weight={best_weight_path.name}")
    for key in [
        "PA",
        "Recall",
        "Precision",
        "F1",
        "IoU",
        "PA_edge",
        "Recall_edge",
        "Precision_edge",
        "F1_edge",
        "IoU_edge",
    ]:
        print(f"{key}={results[key]:.6f}")

    metrics_path = METRICS_DIR / "final_validation_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(f"weight={best_weight_path.name}\n")
        for key, value in results.items():
            file.write(f"{key}={value:.6f}\n")

    return results


# -------------------------
# Main workflow
# -------------------------
def main() -> None:
    """Run CT-BDCN training, prediction export, and validation evaluation."""
    print(f"Using device: {DEVICE}")
    print(f"Project root: {ROOT_DIR}")

    train_dataset = MyDataset(
        image_dir=str(TRAIN_IMAGE_DIR),
        mask_dir=str(TRAIN_MASK_DIR),
        num_classes=NUM_CLASSES,
        size=IMAGE_SIZE,
        augment=True,
        mode="train",
    )
    val_dataset = MyDataset(
        image_dir=str(VAL_IMAGE_DIR),
        mask_dir=str(VAL_MASK_DIR),
        num_classes=NUM_CLASSES,
        size=IMAGE_SIZE,
        augment=False,
        mode="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = EdgeEnhancedSegNet_UNetBDCN(
        in_channels=3,
        num_classes=NUM_CLASSES,
        embed_dim=128,
        num_heads=4,
        transformer_layers=2,
    ).to(DEVICE)

    if LAST_WEIGHT_PATH.exists():
        try:
            state = torch.load(LAST_WEIGHT_PATH, map_location=DEVICE)
            model.load_state_dict(state)
            print("Loaded the existing latest weight file.")
        except RuntimeError as error:
            print("The existing weight file does not match the current model. Training starts from scratch.")
            print(f"Detail: {error}")
    else:
        print("No existing weight file was found. Training starts from scratch.")

    criterion_edge = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion_edge=criterion_edge,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
    )

    export_best_predictions(model, val_dataset, BEST_WEIGHT_PATH)
    evaluate_boundary_metrics(model, val_dataset, BEST_WEIGHT_PATH, k=BOUNDARY_BAND_K)


if __name__ == "__main__":
    main()
