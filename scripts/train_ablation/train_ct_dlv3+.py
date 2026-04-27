"""
Training script for the CT-DLV3+ ablation variant.

This script trains a CNN-Transformer segmentation model with an ASPP-style
edge branch for shoreline extraction. It saves the last model, the best model,
training logs, metric curves, prediction masks, and comparison images.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from data import MyDataset
from models.ablation_ct_dlv3plus import EdgeEnhancedSegNetASPPEdge


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

RESULT_DIR = ROOT_DIR / "results" / "ablation_ct_dlv3plus"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LAST_WEIGHT_PATH = RESULT_DIR / "ct_dlv3plus_last.pth"
BEST_WEIGHT_PATH = RESULT_DIR / "ct_dlv3plus_best.pth"

PRED_DIR = RESULT_DIR / "best_epoch_predictions"
COMPARE_DIR = RESULT_DIR / "best_epoch_comparisons"
METRICS_DIR = RESULT_DIR / "metrics"

for directory in (PRED_DIR, COMPARE_DIR, METRICS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TRAIN_IMAGE_DIR = ROOT_DIR / "datasets" / "train" / "images"
TRAIN_MASK_DIR = ROOT_DIR / "datasets" / "train" / "masks"

NUM_CLASSES = 1
BATCH_SIZE = 4
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4
EDGE_LOSS_WEIGHT = 0.2


def get_edge_band(gt_mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Construct a boundary band from a binary mask using dilation and erosion."""
    pad = k // 2
    dilated = F.max_pool2d(gt_mask, kernel_size=k, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - gt_mask, kernel_size=k, stride=1, padding=pad)
    band = (dilated - eroded) > 0.5
    return band.float()


def compute_binary_metrics(preds: torch.Tensor, masks: torch.Tensor):
    """Compute PA, recall, precision, F1 score, and IoU for binary masks."""
    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    correct = (preds_flat == masks_flat).sum().item()
    total = masks_flat.numel()

    tp = ((preds_flat == 1) & (masks_flat == 1)).sum().item()
    fp = ((preds_flat == 1) & (masks_flat == 0)).sum().item()
    fn = ((preds_flat == 0) & (masks_flat == 1)).sum().item()

    return correct, total, tp, fp, fn


def build_model() -> nn.Module:
    """Create the CT-DLV3+ ablation model."""
    model = EdgeEnhancedSegNetASPPEdge(
        in_channels=3,
        num_classes=NUM_CLASSES,
        embed_dim=128,
        num_heads=4,
        transformer_layers=2,
    )
    return model.to(DEVICE)


def build_dataloader():
    """Create the training dataset and dataloader."""
    train_dataset = MyDataset(
        image_dir=str(TRAIN_IMAGE_DIR),
        mask_dir=str(TRAIN_MASK_DIR),
        mode="train",
        size=(256, 256),
        augment=True,
    )

    data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    return train_dataset, data_loader


def load_last_weight_if_available(model: nn.Module) -> None:
    """Resume training from the last saved weight if it is compatible."""
    if not LAST_WEIGHT_PATH.exists():
        print("No existing last weight was found. Training starts from scratch.")
        return

    try:
        state = torch.load(LAST_WEIGHT_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        print("Loaded the existing last weight successfully.")
    except RuntimeError as error:
        print("Existing weights are incompatible with the current model. Training starts from scratch.")
        print("Detail:", error)


def train(
    model: nn.Module,
    data_loader: DataLoader,
    criterion_seg: nn.Module,
    criterion_edge: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = NUM_EPOCHS,
):
    """Train the model and save model weights and metric logs."""
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

        for batch_idx, (images, masks) in enumerate(tqdm.tqdm(data_loader)):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            seg_logits, edge_logits = model(images)

            edge_gt = get_edge_band(masks, k=5)
            loss_seg = criterion_seg(seg_logits, masks)
            loss_edge = criterion_edge(edge_logits, edge_gt)
            loss = loss_seg + EDGE_LOSS_WEIGHT * loss_edge

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                print(
                    f"Epoch {epoch + 1}, batch {batch_idx}, "
                    f"loss={loss.item():.6f}, "
                    f"seg={loss_seg.item():.6f}, "
                    f"edge={loss_edge.item():.6f}"
                )

            running_loss += loss.item() * images.size(0)

            with torch.no_grad():
                probs = torch.sigmoid(seg_logits)
                preds = (probs > 0.5).float()
                correct, total, tp, fp, fn = compute_binary_metrics(preds, masks)

                total_correct += correct
                total_pixels += total
                total_tp += tp
                total_fp += fp
                total_fn += fn

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

        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "PA": pa,
                "Recall": recall,
                "Precision": precision,
                "F1": f1,
                "IoU": iou,
            }
        )

        torch.save(model.state_dict(), LAST_WEIGHT_PATH)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_WEIGHT_PATH)
            print(f"Updated best model at epoch {best_epoch} with loss={best_loss:.6f}")

    save_training_logs(history, best_epoch, best_loss)
    print(f"Training finished. Best epoch = {best_epoch}, best loss = {best_loss:.6f}")
    return best_epoch, history


def save_training_logs(history: list, best_epoch: int, best_loss: float) -> None:
    """Save metric history, metric curves, and best-epoch information."""
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
    """Export prediction masks and image-mask-prediction comparison panels."""
    if not best_weight_path.exists():
        print("Best weight file was not found. Prediction export is skipped.")
        return

    print("Loading the best model for prediction export.")
    state = torch.load(best_weight_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    if hasattr(dataset, "augment"):
        dataset.augment = False

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, mask = dataset[idx]

            if hasattr(dataset, "samples"):
                image_path, _ = dataset.samples[idx]
                name = Path(image_path).name
            elif hasattr(dataset, "names"):
                name = f"{dataset.names[idx]}.png"
            else:
                name = f"{idx:05d}.png"

            input_tensor = image.unsqueeze(0).to(DEVICE)
            seg_logits, _ = model(input_tensor)

            prob = torch.sigmoid(seg_logits)[0, 0]
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

    print("Prediction masks and comparison images were saved.")


def main() -> None:
    train_dataset, data_loader = build_dataloader()
    model = build_model()
    load_last_weight_if_available(model)

    criterion_seg = nn.BCEWithLogitsLoss()
    criterion_edge = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(
        model=model,
        data_loader=data_loader,
        criterion_seg=criterion_seg,
        criterion_edge=criterion_edge,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
    )
    export_best_predictions(model, train_dataset, BEST_WEIGHT_PATH)


if __name__ == "__main__":
    main()
