"""
Ablation model definition for the CT-UNet variant used in shoreline extraction.

This file defines a CNN-Transformer segmentation network with a U-Net-style
edge branch. It is intended for the ablation experiments of the CT-BDCN
shoreline extraction project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution followed by batch normalization and ReLU activation."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = 1, d: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetEdgeBranch(nn.Module):
    """
    U-Net-style edge branch.

    Input:
        f1: Tensor with shape [B, 64, H/2, W/2].

    Output:
        edge_feat: Tensor with shape [B, out_ch, H/2, W/2].
    """

    def __init__(self, in_ch: int = 64, mid_ch: int = 32, out_ch: int = 16):
        super().__init__()
        self.enc1 = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch),
            ConvBNReLU(mid_ch, mid_ch),
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            ConvBNReLU(mid_ch, mid_ch),
            ConvBNReLU(mid_ch, mid_ch),
        )
        self.up = nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBNReLU(mid_ch + mid_ch, mid_ch),
            ConvBNReLU(mid_ch, mid_ch),
        )
        self.out_conv = nn.Conv2d(mid_ch, out_ch, kernel_size=1)

    def forward(self, f1: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(f1)
        x2 = self.pool(x1)
        x2 = self.enc2(x2)

        up = self.up(x2)
        if up.shape[-2:] != x1.shape[-2:]:
            up = F.interpolate(up, size=x1.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([up, x1], dim=1)
        x = self.dec1(x)
        edge_feat = self.out_conv(x)
        return edge_feat


class EdgeEnhancedSegNetUNetEdge(nn.Module):
    """
    CNN-Transformer segmentation network with a U-Net-style edge branch.

    This model corresponds to the CT-UNet ablation variant. It keeps the
    CNN-Transformer backbone and replaces the boundary branch with a compact
    U-Net-style edge branch.
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.edge_branch = UNetEdgeBranch(in_ch=64, mid_ch=32, out_ch=16)
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

        b, c, h, w = f2.shape
        tokens = f2.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        f2_trans = tokens.transpose(1, 2).contiguous().view(b, c, h, w)

        edge_feat = self.edge_branch(f1)
        gate = torch.sigmoid(self.edge_gate(edge_feat))
        edge_feat_gated = edge_feat * (0.5 + gate)
        edge_map = self.edge_out(edge_feat_gated)

        up1 = self.up1(f2_trans)
        if up1.shape[-2:] != f1.shape[-2:]:
            up1 = F.interpolate(up1, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        fusion = torch.cat([up1, f1, edge_feat_gated], dim=1)
        x = self.fuse_conv(fusion)
        x = self.up2(x)
        seg_logits = self.final_conv(x)

        return seg_logits, edge_map


EdgeEnhancedSegNet_UNetEdge = EdgeEnhancedSegNetUNetEdge
