# -*- coding: utf-8 -*-
"""
Pure validation script for EdgeEnhancedSegNet_UNetEdge.

This script only handles validation:
1. Load validation images and masks from val/images and val/labels.
2. Load the trained model weights.
3. Save binary segmentation predictions.
4. Save image/mask/prediction comparison maps.
5. Compute full-image segmentation metrics.
6. Compute edge-band segmentation metrics.
7. Compute edge-head metrics against the edge-band target.

The network definition is intentionally not included here. Put the network file in the
same directory as this script and make sure it exposes EdgeEnhancedSegNet_UNetEdge.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF


# ================== Path configuration ==================
current_dir = Path(__file__).resolve().parent
print("current_dir:", current_dir)

VAL_IMG_DIR = current_dir / "val" / "images"
VAL_MASK_DIR = current_dir / "val" / "labels"

OUT_ROOT = current_dir / "result"
BEST_WEIGHT_PATH = OUT_ROOT / "cntrans+unet_best.pth"

PRED_DIR = OUT_ROOT / "cntrans+unet_val_preds"
CMP_DIR = OUT_ROOT / "cntrans+unet_val_compare"
EDGE_PRED_DIR = OUT_ROOT / "cntrans+unet_val_edgepreds"
METRICS_TXT = OUT_ROOT / "cntrans+unet_metrics_val.txt"

for directory in (OUT_ROOT, PRED_DIR, CMP_DIR, EDGE_PRED_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ================== Model import ==================
# Keep the network definition in a separate file.
# If your network file has another name, only change this import line.
sys.path.append(str(current_dir))
from edge_enhanced_segnet_unet_edge import EdgeEnhancedSegNet_UNetEdge


# ================== Device ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)


def build_model() -> torch.nn.Module:
    """Build the same model architecture used during training."""
    model = EdgeEnhancedSegNet_UNetEdge(
        in_channels=3,
        num_classes=1,
        embed_dim=128,
        num_heads=4,
        transformer_layers=2,
    ).to(device)
    return model


class SimpleValDataset(Dataset):
    """Validation dataset built directly from image and mask folders."""

    def __init__(self, img_dir: Path, mask_dir: Path, size: Tuple[int, int] = (256, 256)):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.size = size
        self.exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

        if not self.img_dir.exists():
            raise RuntimeError(f"Image directory does not exist: {self.img_dir}")
        if not self.mask_dir.exists():
            raise RuntimeError(f"Mask directory does not exist: {self.mask_dir}")

        self.img_paths = [
            p for p in sorted(self.img_dir.iterdir())
            if p.is_file() and p.suffix.lower() in self.exts
        ]
        if not self.img_paths:
            raise RuntimeError(f"No image files were found in: {self.img_dir}")

        self.mask_paths: List[Path] = []
        for img_path in self.img_paths:
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                self.mask_paths.append(mask_path)
                continue

            matched_mask = None
            for ext in self.exts:
                candidate = self.mask_dir / f"{img_path.stem}{ext}"
                if candidate.exists():
                    matched_mask = candidate
                    break

            if matched_mask is None:
                raise RuntimeError(
                    f"No matching mask was found for {img_path.name} in {self.mask_dir}"
                )
            self.mask_paths.append(matched_mask)

        print(
            f"[SimpleValDataset] images={len(self.img_paths)} "
            f"img_dir={self.img_dir} mask_dir={self.mask_dir}"
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.size is not None:
            img = img.resize(self.size, Image.BILINEAR)
            mask = mask.resize(self.size, Image.NEAREST)

        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t, img_path.name


def get_edge_band(gt_mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Create a binary edge band from the ground-truth mask."""
    pad = k // 2
    dilate = F.max_pool2d(gt_mask, kernel_size=k, stride=1, padding=pad)
    erode = 1.0 - F.max_pool2d(1.0 - gt_mask, kernel_size=k, stride=1, padding=pad)
    band = (dilate - erode) > 0.5
    return band.float()


def load_weights_strict(model: torch.nn.Module, weight_path: Path) -> None:
    """Load weights and handle common state_dict wrappers."""
    state = torch.load(weight_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)


def compute_metrics(tp: float, fp: float, fn: float, correct: float, total: float):
    """Compute PA, recall, precision, F1, and IoU."""
    pa = correct / (total + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn + 1e-6)
    return pa, recall, precision, f1, iou


def evaluate(k_edge: int = 5, thr_seg: float = 0.5, thr_edge: float = 0.5) -> None:
    print("\n=== VAL EVAL: cntrans+unet ===")

    if not BEST_WEIGHT_PATH.exists():
        raise RuntimeError(f"Best weight file was not found: {BEST_WEIGHT_PATH}")

    model = build_model()
    load_weights_strict(model, BEST_WEIGHT_PATH)
    model.to(device)
    model.eval()
    print("Loaded weight:", BEST_WEIGHT_PATH.name)

    dataset = SimpleValDataset(VAL_IMG_DIR, VAL_MASK_DIR, size=(256, 256))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    total_pixels = 0.0
    total_correct = 0.0
    total_tp = total_fp = total_fn = 0.0

    edge_total_pixels = 0.0
    edge_total_correct = 0.0
    edge_tp = edge_fp = edge_fn = 0.0

    head_total_pixels = 0.0
    head_total_correct = 0.0
    head_tp = head_fp = head_fn = 0.0

    with torch.no_grad():
        for idx, (img, mask, name) in enumerate(loader):
            img = img.to(device)
            mask = mask.to(device)

            seg_logits, edge_logits = model(img)

            if seg_logits.shape[-2:] != mask.shape[-2:]:
                seg_logits = F.interpolate(seg_logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            if edge_logits.shape[-2:] != mask.shape[-2:]:
                edge_logits = F.interpolate(edge_logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)

            seg_prob = torch.sigmoid(seg_logits)
            seg_pred = (seg_prob > thr_seg).float()

            preds_flat = seg_pred.view(-1)
            masks_flat = mask.view(-1)

            total_correct += (preds_flat == masks_flat).sum().item()
            total_pixels += masks_flat.numel()
            total_tp += ((preds_flat == 1) & (masks_flat == 1)).sum().item()
            total_fp += ((preds_flat == 1) & (masks_flat == 0)).sum().item()
            total_fn += ((preds_flat == 0) & (masks_flat == 1)).sum().item()

            band = get_edge_band(mask, k=k_edge)
            band_flat = band.view(-1) > 0.5
            p_edge = preds_flat[band_flat]
            g_edge = masks_flat[band_flat]

            if p_edge.numel() > 0:
                edge_total_correct += (p_edge == g_edge).sum().item()
                edge_total_pixels += g_edge.numel()
                edge_tp += ((p_edge == 1) & (g_edge == 1)).sum().item()
                edge_fp += ((p_edge == 1) & (g_edge == 0)).sum().item()
                edge_fn += ((p_edge == 0) & (g_edge == 1)).sum().item()

            edge_prob = torch.sigmoid(edge_logits)
            edge_pred = (edge_prob > thr_edge).float()
            edge_pred_flat = edge_pred.view(-1)
            band_flat_all = band.view(-1)

            head_total_correct += (edge_pred_flat == band_flat_all).sum().item()
            head_total_pixels += band_flat_all.numel()
            head_tp += ((edge_pred_flat == 1) & (band_flat_all == 1)).sum().item()
            head_fp += ((edge_pred_flat == 1) & (band_flat_all == 0)).sum().item()
            head_fn += ((edge_pred_flat == 0) & (band_flat_all == 1)).sum().item()

            filename = name[0]
            pred_map = seg_pred[0, 0].cpu()
            save_image(pred_map.unsqueeze(0), PRED_DIR / filename)

            edge_map = edge_prob[0, 0].cpu().clamp(0, 1)
            save_image(edge_map.unsqueeze(0), EDGE_PRED_DIR / filename)

            vis_img = img[0].cpu()
            vis_mask = mask[0].cpu().repeat(3, 1, 1)
            vis_pred = pred_map.unsqueeze(0).repeat(3, 1, 1)
            triplet = torch.cat([vis_img, vis_mask, vis_pred], dim=2)
            save_image(triplet, CMP_DIR / filename)

            if (idx + 1) % 50 == 0:
                print(f"  Validated {idx + 1}/{len(loader)} images...")

    pa, recall, precision, f1, iou = compute_metrics(total_tp, total_fp, total_fn, total_correct, total_pixels)

    if edge_total_pixels > 0:
        pa_edge, recall_edge, precision_edge, f1_edge, iou_edge = compute_metrics(
            edge_tp, edge_fp, edge_fn, edge_total_correct, edge_total_pixels
        )
    else:
        pa_edge = recall_edge = precision_edge = f1_edge = iou_edge = 0.0

    pa_head, recall_head, precision_head, f1_head, iou_head = compute_metrics(
        head_tp, head_fp, head_fn, head_total_correct, head_total_pixels
    )

    print("\n=== Eval result (VAL) seg ===")
    print(f"PA        = {pa:.6f}")
    print(f"Recall    = {recall:.6f}")
    print(f"Precision = {precision:.6f}")
    print(f"F1        = {f1:.6f}")
    print(f"IoU       = {iou:.6f}")

    print("\n--- Edge-band metrics for segmentation ---")
    print(f"k_edge         = {k_edge}")
    print(f"PA_edge        = {pa_edge:.6f}")
    print(f"Recall_edge    = {recall_edge:.6f}")
    print(f"Precision_edge = {precision_edge:.6f}")
    print(f"F1_edge        = {f1_edge:.6f}")
    print(f"IoU_edge       = {iou_edge:.6f}")

    print("\n--- Edge-head metrics against the edge-band target ---")
    print(f"PA_head        = {pa_head:.6f}")
    print(f"Recall_head    = {recall_head:.6f}")
    print(f"Precision_head = {precision_head:.6f}")
    print(f"F1_head        = {f1_head:.6f}")
    print(f"IoU_head       = {iou_head:.6f}")

    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(f"weight={BEST_WEIGHT_PATH.name}\n")
        f.write(f"PA={pa:.6f}\n")
        f.write(f"Recall={recall:.6f}\n")
        f.write(f"Precision={precision:.6f}\n")
        f.write(f"F1={f1:.6f}\n")
        f.write(f"IoU={iou:.6f}\n")
        f.write(f"k_edge={k_edge}\n")
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
    print("Preds saved to:", PRED_DIR)
    print("Compare saved to:", CMP_DIR)
    print("Edge preds saved to:", EDGE_PRED_DIR)


if __name__ == "__main__":
    evaluate(k_edge=5, thr_seg=0.5, thr_edge=0.5)
