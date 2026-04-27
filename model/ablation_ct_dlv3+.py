"""
Ablation network definition for the CT-DLV3+ variant.

This module defines a CNN-Transformer segmentation network with an
ASPP-style edge branch. It is intended for the ablation experiments in
the CT-BDCN shoreline extraction project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution, batch normalization, and ReLU activation."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = 1, d: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPPEdgeBranch(nn.Module):
    """
    ASPP-style edge branch.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output edge-feature channels.
    rates : tuple
        Dilation rates used in the ASPP branches.

    Input
    -----
    f1 : torch.Tensor
        Shallow feature map with shape [B, 64, H/2, W/2].

    Output
    ------
    edge_feat : torch.Tensor
        Edge feature map with shape [B, out_ch, H/2, W/2].
    """

    def __init__(self, in_ch: int = 64, out_ch: int = 16, rates=(1, 2, 4, 8)):
        super().__init__()
        inter = 32

        self.branch1x1 = ConvBNReLU(in_ch, inter, k=1, p=0, d=1)

        self.branches_dilated = nn.ModuleList(
            [ConvBNReLU(in_ch, inter, k=3, p=r, d=r) for r in rates]
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, inter, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
        )

        total_ch = inter * (1 + len(rates) + 1)
        self.project = nn.Sequential(
            ConvBNReLU(total_ch, inter, k=1, p=0),
            nn.Conv2d(inter, out_ch, kernel_size=1),
        )

    def forward(self, f1: torch.Tensor) -> torch.Tensor:
        h, w = f1.shape[-2:]

        features = [self.branch1x1(f1)]
        features.extend([branch(f1) for branch in self.branches_dilated])

        pooled = self.global_pool(f1)
        pooled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
        features.append(pooled)

        edge_feat = self.project(torch.cat(features, dim=1))
        return edge_feat


class EdgeEnhancedSegNetASPPEdge(nn.Module):
    """
    CNN-Transformer segmentation network with an ASPP-style edge branch.

    This model is used as the CT-DLV3+ ablation variant. It keeps the
    CNN-Transformer backbone and replaces the BDCN-style boundary branch
    with an ASPP-style edge branch.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dim: int = 128,
        num_heads: int = 4,
        transformer_layers: int = 2,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        self.edge_branch = ASPPEdgeBranch(
            in_ch=64,
            out_ch=16,
            rates=(1, 2, 4, 8),
        )

        self.edge_gate = nn.Conv2d(16, 1, kernel_size=1, bias=True)
        self.edge_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        self.up1 = nn.ConvTranspose2d(embed_dim, 64, 2, stride=2)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(64 + 64 + 16, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)

        batch_size, channels, height, width = f2.shape
        tokens = f2.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        f2_trans = tokens.transpose(1, 2).view(batch_size, channels, height, width)

        edge_feat = self.edge_branch(f1)
        gate = torch.sigmoid(self.edge_gate(edge_feat))
        edge_feat_gated = edge_feat * (0.5 + gate)
        edge_map = self.edge_out(edge_feat_gated)

        up1 = self.up1(f2_trans)
        fusion = torch.cat([up1, f1, edge_feat_gated], dim=1)
        x = self.fuse_conv(fusion)
        x = self.up2(x)
        seg_logits = self.final_conv(x)

        return seg_logits, edge_map


# Backward-compatible alias for the original class name.
EdgeEnhancedSegNet_ASPPEdge = EdgeEnhancedSegNetASPPEdge
