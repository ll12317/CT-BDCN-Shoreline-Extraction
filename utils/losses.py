# -*- coding: utf-8 -*-
"""Loss functions for edge-enhanced binary segmentation."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .iou_edge import edge_iou_loss, get_edge_band


def weighted_bce_with_logits(
    seg_logits: torch.Tensor,
    masks: torch.Tensor,
    edge_gt: torch.Tensor,
    alpha: float = 2.0,
) -> torch.Tensor:
    """
    Compute weighted BCE loss for segmentation.

    Boundary pixels receive a larger weight through the boundary band.
    """
    weight_map = 1.0 + alpha * edge_gt
    return F.binary_cross_entropy_with_logits(seg_logits, masks, weight=weight_map)


def side_supervision_loss(
    side_maps: Optional[Sequence[torch.Tensor]],
    edge_gt: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute deep supervision loss for multi-scale side maps."""
    if side_maps is None:
        return torch.tensor(0.0, device=edge_gt.device)

    loss = torch.tensor(0.0, device=edge_gt.device)
    edge_gt_half = F.interpolate(edge_gt, size=side_maps[0].shape[-2:], mode="nearest")

    for side_map in side_maps:
        loss = loss + criterion(side_map, edge_gt_half)

    return loss


def compute_edge_enhanced_loss(
    model: nn.Module,
    seg_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    masks: torch.Tensor,
    criterion_edge: nn.Module,
    alpha: float = 2.0,
    lambda_edge: float = 0.2,
    lambda_side: float = 0.3,
    gamma_edge_iou: float = 1.0,
    k_edge: int = 5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the total training loss for edge-enhanced segmentation.

    The total loss contains:
        1. Boundary-weighted segmentation BCE loss.
        2. Edge-head BCE loss.
        3. Multi-scale side-map deep supervision loss.
        4. Soft boundary IoU loss.
    """
    edge_gt = get_edge_band(masks, k=k_edge)

    loss_seg = weighted_bce_with_logits(
        seg_logits=seg_logits,
        masks=masks,
        edge_gt=edge_gt,
        alpha=alpha,
    )

    loss_edge_main = criterion_edge(edge_logits, edge_gt)

    side_maps = getattr(model, "_last_side_maps", None)
    loss_side = side_supervision_loss(
        side_maps=side_maps,
        edge_gt=edge_gt,
        criterion=criterion_edge,
    )

    loss_edge_total = loss_edge_main + lambda_side * loss_side
    loss_edge_iou = edge_iou_loss(seg_logits, masks, k=k_edge)

    total_loss = loss_seg + lambda_edge * loss_edge_total + gamma_edge_iou * loss_edge_iou

    loss_items = {
        "total": total_loss.detach(),
        "seg": loss_seg.detach(),
        "edge_main": loss_edge_main.detach(),
        "edge_side": loss_side.detach(),
        "edge_iou": loss_edge_iou.detach(),
    }

    return total_loss, loss_items
