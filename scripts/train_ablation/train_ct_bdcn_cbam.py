"""
Training and evaluation script for the CT-BDCN-CBAM ablation model.

This script trains the CT-BDCN-CBAM variant, saves the last and best model
weights, exports prediction examples, and reports both region-level and
boundary-band evaluation metrics.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from data import MyDataset
from models.ablation_ct_bdcn_cbam import EdgeEnhancedSegNetUNetBDCNCBAM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_CLASSES = 1
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4

LAMBDA_EDGE = 0.2
LAMBDA_SIDE = 0.3
GAMMA_EDGE_IOU = 1.0
BOUNDARY_WEIGHT_ALPHA = 2.0
EDGE_BAND_KERNEL = 5

TRAIN_IMG_DIR = ROOT_DIR / "datasets" / "train" / "images"
TRAIN_MASK_DIR = ROOT_DIR / "datasets" / "train" / "masks"
VAL_IMG_DIR = ROOT_DIR / "datasets" / "val" / "images"
VAL_MASK_DIR = ROOT_DIR / "datasets" / "val" / "masks"

SAVE_DIR = ROOT_DIR / "results" / "ablation_ct_bdcn_cbam"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_PATH = SAVE_DIR / "ct_bdcn_cbam_last.pth"
BEST_WEIGHT_PATH = SAVE_DIR / "ct_bdcn_cbam_best.pth"
BEST_PRED_DIR = SAVE_DIR / "best_epoch_predictions"
COMPARE_DIR = SAVE_DIR / "best_epoch_comparisons"
METRICS_DIR = SAVE_DIR / "metrics"

for directory in (BEST_PRED_DIR, COMPARE_DIR, METRICS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def build_dataset(split: str, augment: bool) -> MyDataset:
    """Create a dataset for the requested split."""
    if split == "train":
        image_dir = TRAIN_IMG_DIR
        mask_dir = TRAIN_MASK_DIR
    elif split == "val":
        image_dir = VAL_IMG_DIR
        mask_dir = VAL_MASK_DIR
    else:
        raise ValueError("split must be 'train' or 'val'.")

    return MyDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        mode=split,
        size=IMAGE_SIZE,
        augment=augment,
    )


def get_edge_band(gt_mask: torch.Tensor, k: int = EDGE_BAND_KERNEL) -> torch.Tensor:
    """Construct a morphology-based boundary band from a binary mask."""
    pad = k // 2
    dilate = F.max_pool2d(gt_mask, kernel_size=k, stride=1, padding=pad)
    erode = 1.0 - F.max_pool2d(1.0 - gt_mask, kernel_size=k, stride=1, padding=pad)
    band = (dilate - erode) > 0.5
    return band.float()


def edge_iou_loss(seg_logits: torch.Tensor, masks: torch.Tensor, k: int = EDGE_BAND_KERNEL) -> torch.Tensor:
    """Calculate soft IoU loss within the boundary band."""
    probs = torch.sigmoid(seg_logits)
    with torch.no_grad():
        band = get_edge_band(masks, k=k)

    pred_edge = probs * band
    gt_edge = masks * band

    intersection = (pred_edge * gt_edge).sum()
    union = (pred_edge + gt_edge - pred_edge * gt_edge).sum() + 1e-6
    return 1.0 - intersection / union


def calculate_metrics(preds: torch.Tensor, masks: torch.Tensor) -> Dict[str, float]:
    """Calculate binary segmentation metrics from prediction and mask tensors."""
    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    total_pixels = masks_flat.numel()
    correct = (preds_flat == masks_flat).sum().item()
    tp = ((preds_flat == 1) & (masks_flat == 1)).sum().item()
    fp = ((preds_flat == 1) & (masks_flat == 0)).sum().item()
    fn = ((preds_flat == 0) & (masks_flat == 1)).sum().item()

    pa = correct / (total_pixels + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
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
        "correct": correct,
        "pixels": total_pixels,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_edge: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0

    total_pixels = 0
    total_correct = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for step, (images, masks) in enumerate(tqdm.tqdm(loader)):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        seg_logits, edge_logits = model(images)
        edge_gt = get_edge_band(masks, k=EDGE_BAND_KERNEL)

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
        loss_edge_iou = edge_iou_loss(seg_logits, masks, k=EDGE_BAND_KERNEL)
        loss = loss_seg + LAMBDA_EDGE * loss_edge_total + GAMMA_EDGE_IOU * loss_edge_iou

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(
                f"{epoch + 1}-{step}-loss={loss.item():.6f} "
                f"(seg={loss_seg.item():.6f}, "
                f"edge_main={loss_edge_main.item():.6f}, "
                f"edge_side={loss_side.item():.6f}, "
                f"edgeIoU={loss_edge_iou.item():.6f})"
            )

        running_loss += loss.item() * images.size(0)

        with torch.no_grad():
            probs = torch.sigmoid(seg_logits)
            preds = (probs > 0.5).float()
            batch_metrics = calculate_metrics(preds, masks)
            total_correct += batch_metrics["correct"]
            total_pixels += batch_metrics["pixels"]
            total_tp += batch_metrics["TP"]
            total_fp += batch_metrics["FP"]
            total_fn += batch_metrics["FN"]

    epoch_loss = running_loss / len(loader.dataset)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    precision = total_tp / (total_tp + total_fp + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    pa = total_correct / (total_pixels + 1e-6)

    return {
        "loss": epoch_loss,
        "PA": pa,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "IoU": iou,
    }


def train(model: nn.Module, train_loader: DataLoader, criterion_edge: nn.Module, optimizer: optim.Optimizer) -> Tuple[int, List[Dict[str, float]]]:
    """Run the complete training process."""
    history = []
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(NUM_EPOCHS):
        epoch_metrics = train_one_epoch(model, train_loader, criterion_edge, optimizer, epoch)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"loss={epoch_metrics['loss']:.6f} "
            f"PA={epoch_metrics['PA']:.4f} "
            f"Recall={epoch_metrics['Recall']:.4f} "
            f"F1={epoch_metrics['F1']:.4f} "
            f"IoU={epoch_metrics['IoU']:.4f}"
        )

        history.append({"epoch": epoch + 1, **epoch_metrics})
        torch.save(model.state_dict(), WEIGHT_PATH)

        if epoch_metrics["loss"] < best_loss:
            best_loss = epoch_metrics["loss"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_WEIGHT_PATH)
            print(f"*** Updated best model at epoch {best_epoch} (loss={best_loss:.6f})")

    save_training_history(history, best_epoch)
    print(f"Training complete. Best epoch = {best_epoch}, best loss = {best_loss:.6f}")
    return best_epoch, history


def save_training_history(history: List[Dict[str, float]], best_epoch: int) -> None:
    """Save metric logs and training curves."""
    log_path = METRICS_DIR / "metrics_history.txt"
    with open(log_path, "w", encoding="utf-8") as file:
        file.write("epoch,loss,PA,Recall,Precision,F1,IoU\n")
        for item in history:
            file.write(
                f"{item['epoch']},{item['loss']:.6f},"
                f"{item['PA']:.6f},{item['Recall']:.6f},"
                f"{item['Precision']:.6f},{item['F1']:.6f},{item['IoU']:.6f}\n"
            )

    epochs = [item["epoch"] for item in history]
    plt.figure()
    plt.plot(epochs, [item["loss"] for item in history], label="Loss")
    plt.plot(epochs, [item["PA"] for item in history], label="PA")
    plt.plot(epochs, [item["Recall"] for item in history], label="Recall")
    plt.plot(epochs, [item["Precision"] for item in history], label="Precision")
    plt.plot(epochs, [item["F1"] for item in history], label="F1")
    plt.plot(epochs, [item["IoU"] for item in history], label="IoU")
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


def export_best_predictions(model: nn.Module, dataset: Dataset, best_weight_path: Path) -> None:
    """Export binary predictions and side-by-side comparison images."""
    if not best_weight_path.exists():
        print("Best weight file not found. Prediction export skipped.")
        return

    print("Loading best model for prediction export...")
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
            seg_logits, _ = model(input_tensor)
            prob = torch.sigmoid(seg_logits)[0, 0]
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

    print("Prediction and comparison images saved.")


def evaluate_edge_metrics(model: nn.Module, dataset: Dataset, best_weight_path: Path, k: int = EDGE_BAND_KERNEL) -> None:
    """Evaluate region-level and boundary-band metrics using the best model."""
    if not best_weight_path.exists():
        print("Best weight file not found. Evaluation skipped.")
        return

    print("Loading best model for evaluation...")
    state = torch.load(best_weight_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    if hasattr(dataset, "augment"):
        dataset.augment = False

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    totals = {"pixels": 0, "correct": 0, "TP": 0, "FP": 0, "FN": 0}
    edge_totals = {"pixels": 0, "correct": 0, "TP": 0, "FP": 0, "FN": 0}

    with torch.no_grad():
        for images, masks in tqdm.tqdm(loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            seg_logits, _ = model(images)
            preds = (torch.sigmoid(seg_logits) > 0.5).float()

            metrics = calculate_metrics(preds, masks)
            for key in totals:
                source_key = key if key in ("pixels", "correct") else key
                totals[key] += metrics[source_key]

            band = get_edge_band(masks, k=k)
            band_flat = band.view(-1)
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)
            preds_edge = preds_flat[band_flat == 1]
            masks_edge = masks_flat[band_flat == 1]

            if preds_edge.numel() == 0:
                continue

            edge_metrics = calculate_metrics(preds_edge, masks_edge)
            for key in edge_totals:
                source_key = key if key in ("pixels", "correct") else key
                edge_totals[key] += edge_metrics[source_key]

    overall = summarize_totals(totals)
    boundary = summarize_totals(edge_totals)

    output_lines = [
        f"weight={best_weight_path.name}",
        f"PA={overall['PA']:.6f}",
        f"Recall={overall['Recall']:.6f}",
        f"Precision={overall['Precision']:.6f}",
        f"F1={overall['F1']:.6f}",
        f"IoU={overall['IoU']:.6f}",
        f"PA_edge={boundary['PA']:.6f}",
        f"Recall_edge={boundary['Recall']:.6f}",
        f"Precision_edge={boundary['Precision']:.6f}",
        f"F1_edge={boundary['F1']:.6f}",
        f"IoU_edge={boundary['IoU']:.6f}",
    ]

    for line in output_lines:
        print(line)

    with open(METRICS_DIR / "evaluation_metrics.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(output_lines) + "\n")


def summarize_totals(totals: Dict[str, float]) -> Dict[str, float]:
    """Convert accumulated confusion statistics into evaluation metrics."""
    pa = totals["correct"] / (totals["pixels"] + 1e-6)
    recall = totals["TP"] / (totals["TP"] + totals["FN"] + 1e-6)
    precision = totals["TP"] / (totals["TP"] + totals["FP"] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = totals["TP"] / (totals["TP"] + totals["FP"] + totals["FN"] + 1e-6)
    return {"PA": pa, "Recall": recall, "Precision": precision, "F1": f1, "IoU": iou}


def main() -> None:
    """Run training, prediction export, and metric evaluation."""
    train_dataset = build_dataset(split="train", augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    if VAL_IMG_DIR.is_dir() and VAL_MASK_DIR.is_dir():
        eval_dataset = build_dataset(split="val", augment=False)
    else:
        print("Validation dataset not found. Evaluation will use the training dataset.")
        eval_dataset = train_dataset

    model = EdgeEnhancedSegNetUNetBDCNCBAM(
        in_channels=3,
        num_classes=NUM_CLASSES,
        embed_dim=128,
        num_heads=4,
        transformer_layers=2,
    ).to(DEVICE)

    if WEIGHT_PATH.exists():
        try:
            state = torch.load(WEIGHT_PATH, map_location=DEVICE)
            model.load_state_dict(state)
            print("Loaded previous last-epoch weights.")
        except RuntimeError as error:
            print("Existing weights do not match the current model. Training will start from scratch.")
            print(f"Details: {error}")
    else:
        print("No previous weights found. Training will start from scratch.")

    criterion_edge = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, criterion_edge, optimizer)
    export_best_predictions(model, eval_dataset, BEST_WEIGHT_PATH)
    evaluate_edge_metrics(model, eval_dataset, BEST_WEIGHT_PATH)


if __name__ == "__main__":
    main()
