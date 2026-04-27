# -*- coding: utf-8 -*-
"""
Pure validation script for EdgeEnhancedSegNet_UNetBDCN.

This script only handles validation:
1. Load validation images and masks from val/images and val/labels.
2. Load the trained checkpoint.
3. Save binary segmentation predictions.
4. Save image/mask/prediction comparison panels.
5. Compute full-image metrics, edge-band metrics, and optional edge-head metrics.

The network definition is intentionally not included here. Keep the model architecture in
edge_enhanced_segnet_unetbdcn.py and import EdgeEnhancedSegNet_UNetBDCN from that file.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import torchvision.transforms.functional as TF


# ================== Path configuration ==================
current_dir = Path(__file__).resolve().parent
print("current_dir:", current_dir)

VAL_IMG_DIR = current_dir / "val" / "images"
VAL_MASK_DIR = current_dir / "val" / "labels"

RESULT_DIR = current_dir / "result"
BEST_WEIGHT_PATH = RESULT_DIR / "cntrans+unet alpha=2_2.0best.pth"

PRED_DIR = RESULT_DIR / "cntrans+unet alpha=2_2.0val_preds"
CMP_DIR = RESULT_DIR / "cntrans+unet alpha=2_2.0val_compare"
EDGE_PRED_DIR = RESULT_DIR / "cntrans+unet alpha=2_2.0val_edgepreds"
METRICS_DIR = RESULT_DIR / "cntrans+unet alpha=2_2.0metrics"
METRICS_TXT = METRICS_DIR / "cntrans+unet alpha=2_2.0_metrics_val.txt"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 1
NUM_WORKERS = 0
THRESHOLD = 0.5
EDGE_BAND_KERNEL = 5
SAVE_EDGE_PREDICTIONS = True

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
CMP_DIR.mkdir(parents=True, exist_ok=True)
EDGE_PRED_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ================== Model import ==================
sys.path.append(str(current_dir))
try:
    from edge_enhanced_segnet_unetbdcn import EdgeEnhancedSegNet_UNetBDCN
except ImportError as exc:
    raise ImportError(
        "Cannot import EdgeEnhancedSegNet_UNetBDCN. "
        "Please put the network definition in edge_enhanced_segnet_unetbdcn.py "
        "under the same directory as this validation script."
    ) from exc


# ================== Device configuration ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)


# ================== Dataset ==================
class SimpleSegDataset(Dataset):
    """
    Dataset for direct validation from image and mask folders.

    Images are loaded as RGB tensors. Masks are loaded as single-channel tensors
    and binarized to 0/1. The default image size must match the training setting.
    """

    def __init__(self, img_dir: Path, mask_dir: Path, size=(256, 256)):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.size = size

        if not self.img_dir.exists():
            raise RuntimeError(f"Image directory does not exist: {self.img_dir}")
        if not self.mask_dir.exists():
            raise RuntimeError(f"Mask directory does not exist: {self.mask_dir}")

        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        self.img_paths = [
            p for p in sorted(self.img_dir.iterdir())
            if p.is_file() and p.suffix.lower() in exts
        ]
        if not self.img_paths:
            raise RuntimeError(f"No image files were found in: {self.img_dir}")

        self.mask_paths = []
        for img_path in self.img_paths:
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists():
                raise RuntimeError(f"Missing mask for {img_path.name}: {mask_path}")
            self.mask_paths.append(mask_path)

        print(f"[SimpleSegDataset] Found {len(self.img_paths)} validation images.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.size is not None:
            img = img.resize(self.size, Image.BILINEAR)
            mask = mask.resize(self.size, Image.NEAREST)

        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return img, mask, img_path.name


# ================== Model builder ==================
def build_model():
    """Build the same model architecture used during training."""
    model = EdgeEnhancedSegNet_UNetBDCN(
        in_channels=3,
        num_classes=1,
        embed_dim=128,
        num_heads=4,
        transformer_layers=2,
    ).to(device)
    return model


# ================== Metric helpers ==================
def get_edge_band(gt_mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Build a binary boundary band from the ground-truth mask.

    Args:
        gt_mask: Tensor with shape [B, 1, H, W], values in {0, 1}.
        k: Morphological kernel size. A larger value gives a wider band.

    Returns:
        Tensor with shape [B, 1, H, W], where 1 indicates the boundary band.
    """
    pad = k // 2
    dilate = F.max_pool2d(gt_mask, kernel_size=k, stride=1, padding=pad)
    erode = 1.0 - F.max_pool2d(1.0 - gt_mask, kernel_size=k, stride=1, padding=pad)
    band = (dilate - erode) > 0.5
    return band.float()


def compute_binary_metrics(tp, fp, fn, correct, pixels):
    pa = correct / (pixels + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall + 1e-6)
        if (precision + recall) > 0 else 0.0
    )
    iou = tp / (tp + fp + fn + 1e-6)
    return pa, recall, precision, f1, iou


def update_confusion(pred_flat: torch.Tensor, target_flat: torch.Tensor):
    correct = (pred_flat == target_flat).sum().item()
    pixels = target_flat.numel()
    tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
    fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
    fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()
    return tp, fp, fn, correct, pixels


# ================== Validation ==================
def evaluate():
    print("\n=== Starting validation for EdgeEnhancedSegNet_UNetBDCN ===")

    if not BEST_WEIGHT_PATH.exists():
        print("Error: checkpoint not found:", BEST_WEIGHT_PATH)
        return

    print("Loading checkpoint:", BEST_WEIGHT_PATH)

    model = build_model()
    state = torch.load(BEST_WEIGHT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dataset = SimpleSegDataset(
        img_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        size=IMAGE_SIZE,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    total_tp = total_fp = total_fn = 0
    total_correct = total_pixels = 0

    edge_total_tp = edge_total_fp = edge_total_fn = 0
    edge_total_correct = edge_total_pixels = 0

    head_tp = head_fp = head_fn = 0
    head_total_correct = head_total_pixels = 0

    with torch.no_grad():
        for idx, (img, mask, name) in enumerate(loader):
            img = img.to(device)
            mask = mask.to(device)

            seg_logits, edge_logits = model(img)

            seg_probs = torch.sigmoid(seg_logits)
            seg_preds = (seg_probs > THRESHOLD).float()

            preds_flat = seg_preds.view(-1)
            masks_flat = mask.view(-1)

            tp, fp, fn, correct, pixels = update_confusion(preds_flat, masks_flat)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_correct += correct
            total_pixels += pixels

            band = get_edge_band(mask, k=EDGE_BAND_KERNEL)
            band_flat = band.view(-1) > 0.5
            p_edge = preds_flat[band_flat]
            g_edge = masks_flat[band_flat]

            if p_edge.numel() > 0:
                tp_e, fp_e, fn_e, correct_e, pixels_e = update_confusion(p_edge, g_edge)
                edge_total_tp += tp_e
                edge_total_fp += fp_e
                edge_total_fn += fn_e
                edge_total_correct += correct_e
                edge_total_pixels += pixels_e

            edge_probs = torch.sigmoid(edge_logits)
            edge_preds = (edge_probs > THRESHOLD).float()
            edge_preds_flat = edge_preds.view(-1)
            band_flat_all = band.view(-1)

            tp_h, fp_h, fn_h, correct_h, pixels_h = update_confusion(edge_preds_flat, band_flat_all)
            head_tp += tp_h
            head_fp += fp_h
            head_fn += fn_h
            head_total_correct += correct_h
            head_total_pixels += pixels_h

            filename = name[0]
            pred_map = seg_preds[0, 0].cpu()
            save_image(pred_map.unsqueeze(0), PRED_DIR / filename)

            if SAVE_EDGE_PREDICTIONS:
                edge_map = edge_probs[0, 0].cpu().clamp(0, 1)
                save_image(edge_map.unsqueeze(0), EDGE_PRED_DIR / filename)

            vis_img = img[0].cpu()
            if vis_img.shape[0] == 1:
                vis_img = vis_img.repeat(3, 1, 1)

            vis_mask = mask[0].cpu()
            if vis_mask.dim() == 3 and vis_mask.shape[0] == 1:
                vis_mask = vis_mask.repeat(3, 1, 1)
            elif vis_mask.dim() == 2:
                vis_mask = vis_mask.unsqueeze(0).repeat(3, 1, 1)

            vis_pred = pred_map.unsqueeze(0).repeat(3, 1, 1)
            triplet = torch.cat([vis_img, vis_mask, vis_pred], dim=2)
            save_image(triplet, CMP_DIR / filename)

            if (idx + 1) % 50 == 0:
                print(f"  Validated {idx + 1}/{len(loader)} images...")

    pa, recall, precision, f1, iou = compute_binary_metrics(
        total_tp, total_fp, total_fn, total_correct, total_pixels
    )
    pa_edge, recall_edge, precision_edge, f1_edge, iou_edge = compute_binary_metrics(
        edge_total_tp, edge_total_fp, edge_total_fn, edge_total_correct, edge_total_pixels
    )
    pa_head, recall_head, precision_head, f1_head, iou_head = compute_binary_metrics(
        head_tp, head_fp, head_fn, head_total_correct, head_total_pixels
    )

    print("\n=== Segmentation metrics on validation set ===")
    print(f"PA        = {pa:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"Precision = {precision:.4f}")
    print(f"F1        = {f1:.4f}")
    print(f"IoU       = {iou:.4f}")

    print("\n--- Edge-band segmentation metrics ---")
    print(f"PA_edge(seg)        = {pa_edge:.4f}")
    print(f"Recall_edge(seg)    = {recall_edge:.4f}")
    print(f"Precision_edge(seg) = {precision_edge:.4f}")
    print(f"F1_edge(seg)        = {f1_edge:.4f}")
    print(f"IoU_edge(seg)       = {iou_edge:.4f}")

    print("\n--- Edge-head metrics against the boundary band ---")
    print(f"PA_head(edge)        = {pa_head:.4f}")
    print(f"Recall_head(edge)    = {recall_head:.4f}")
    print(f"Precision_head(edge) = {precision_head:.4f}")
    print(f"F1_head(edge)        = {f1_head:.4f}")
    print(f"IoU_head(edge)       = {iou_head:.4f}")

    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(f"weight={BEST_WEIGHT_PATH.name}\n")
        f.write(f"PA={pa:.6f}\n")
        f.write(f"Recall={recall:.6f}\n")
        f.write(f"Precision={precision:.6f}\n")
        f.write(f"F1={f1:.6f}\n")
        f.write(f"IoU={iou:.6f}\n")
        f.write(f"PA_edge(seg)={pa_edge:.6f}\n")
        f.write(f"Recall_edge(seg)={recall_edge:.6f}\n")
        f.write(f"Precision_edge(seg)={precision_edge:.6f}\n")
        f.write(f"F1_edge(seg)={f1_edge:.6f}\n")
        f.write(f"IoU_edge(seg)={iou_edge:.6f}\n")
        f.write(f"PA_head(edge)={pa_head:.6f}\n")
        f.write(f"Recall_head(edge)={recall_head:.6f}\n")
        f.write(f"Precision_head(edge)={precision_head:.6f}\n")
        f.write(f"F1_head(edge)={f1_head:.6f}\n")
        f.write(f"IoU_head(edge)={iou_head:.6f}\n")

    print("Metrics saved to:", METRICS_TXT)
    print("Predictions saved to:", PRED_DIR)
    print("Comparison panels saved to:", CMP_DIR)
    if SAVE_EDGE_PREDICTIONS:
        print("Edge predictions saved to:", EDGE_PRED_DIR)


if __name__ == "__main__":
    evaluate()
