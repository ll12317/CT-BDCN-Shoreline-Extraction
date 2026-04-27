# -*- coding: utf-8 -*-
"""Segmentation metric utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .iou_edge import get_edge_band


def compute_prf_iou(tp: float, fp: float, fn: float, correct: float, total: float):
    """Compute PA, recall, precision, F1, and IoU from accumulated counts."""
    pa = correct / (total + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall + 1e-6)
        if (precision + recall) > 0
        else 0.0
    )
    iou = tp / (tp + fp + fn + 1e-6)
    return pa, recall, precision, f1, iou


@dataclass
class MetricResult:
    """Container for common binary segmentation metrics."""

    pa: float
    recall: float
    precision: float
    f1: float
    iou: float

    def as_dict(self):
        return {
            "PA": self.pa,
            "Recall": self.recall,
            "Precision": self.precision,
            "F1": self.f1,
            "IoU": self.iou,
        }


class BinarySegmentationMeter:
    """Accumulate binary segmentation metrics over multiple batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_pixels = 0.0
        self.total_correct = 0.0
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update_from_predictions(self, preds: torch.Tensor, masks: torch.Tensor):
        """
        Update counts from binary predictions and binary masks.

        Args:
            preds: Binary prediction tensor with values 0/1.
            masks: Binary ground-truth tensor with values 0/1.
        """
        preds_flat = preds.view(-1)
        masks_flat = masks.view(-1)

        self.total_correct += (preds_flat == masks_flat).sum().item()
        self.total_pixels += masks_flat.numel()

        self.tp += ((preds_flat == 1) & (masks_flat == 1)).sum().item()
        self.fp += ((preds_flat == 1) & (masks_flat == 0)).sum().item()
        self.fn += ((preds_flat == 0) & (masks_flat == 1)).sum().item()

    def compute(self) -> MetricResult:
        pa, recall, precision, f1, iou = compute_prf_iou(
            self.tp,
            self.fp,
            self.fn,
            self.total_correct,
            self.total_pixels,
        )
        return MetricResult(pa, recall, precision, f1, iou)


class EdgeBandSegmentationMeter:
    """Accumulate segmentation metrics only within the ground-truth boundary band."""

    def __init__(self, k_edge: int = 5):
        self.k_edge = k_edge
        self.meter = BinarySegmentationMeter()

    def reset(self):
        self.meter.reset()

    def update_from_predictions(self, preds: torch.Tensor, masks: torch.Tensor):
        band = get_edge_band(masks, k=self.k_edge)
        band_flat = band.view(-1) > 0.5

        preds_flat = preds.view(-1)
        masks_flat = masks.view(-1)

        preds_edge = preds_flat[band_flat]
        masks_edge = masks_flat[band_flat]

        if preds_edge.numel() > 0:
            self.meter.update_from_predictions(preds_edge, masks_edge)

    def compute(self) -> MetricResult:
        return self.meter.compute()


def logits_to_binary_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert logits to binary predictions."""
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


def update_full_and_edge_metrics(
    seg_logits: torch.Tensor,
    masks: torch.Tensor,
    full_meter: BinarySegmentationMeter,
    edge_meter: EdgeBandSegmentationMeter,
    threshold: float = 0.5,
):
    """Update full-image and edge-band meters from segmentation logits."""
    preds = logits_to_binary_predictions(seg_logits, threshold=threshold)
    full_meter.update_from_predictions(preds, masks)
    edge_meter.update_from_predictions(preds, masks)
    return preds
