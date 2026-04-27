# -*- coding: utf-8 -*-
"""Boundary-band construction and edge-focused IoU loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_edge_band(gt_mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Build a binary boundary band from a ground-truth mask.

    Args:
        gt_mask: Binary mask tensor with shape [B, 1, H, W].
        k: Morphological kernel size. A larger value creates a wider boundary band.

    Returns:
        Binary boundary-band tensor with shape [B, 1, H, W].
    """
    pad = k // 2
    dilate = F.max_pool2d(gt_mask, kernel_size=k, stride=1, padding=pad)
    erode = 1.0 - F.max_pool2d(1.0 - gt_mask, kernel_size=k, stride=1, padding=pad)
    band = (dilate - erode) > 0.5
    return band.float()


def edge_iou_loss(seg_logits: torch.Tensor, masks: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Compute soft IoU loss within the boundary band.

    Args:
        seg_logits: Segmentation logits with shape [B, 1, H, W].
        masks: Binary ground-truth masks with shape [B, 1, H, W].
        k: Boundary-band kernel size.

    Returns:
        Loss value equal to 1 - soft boundary IoU.
    """
    probs = torch.sigmoid(seg_logits)
    with torch.no_grad():
        band = get_edge_band(masks, k=k)

    pred_edge = probs * band
    gt_edge = masks * band

    intersection = (pred_edge * gt_edge).sum()
    union = (pred_edge + gt_edge - pred_edge * gt_edge).sum() + 1e-6
    iou_edge = intersection / union

    return 1.0 - iou_edge
