"""
CT-BDCN-CBAM ablation model for shoreline extraction.

This file contains only the network architecture used in the ablation
experiment. It defines a CNN-Transformer segmentation backbone with a
U-Net-BDCN-style multi-scale boundary branch and CBAM attention modules.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution, batch normalization, and ReLU activation."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = 1, d: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChannelAttention(nn.Module):
    """Channel attention module used in CBAM."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module used in CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7.")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class UNetBDCNEdgeBranch(nn.Module):
    """
    U-Net-BDCN-style multi-scale boundary branch.

    Args:
        in_ch: Number of input channels.
        c1: Number of channels in the first encoder stage.
        c2: Number of channels in the second encoder stage and bottleneck.
        out_ch: Number of output boundary feature channels.

    Returns:
        edge_feat: Boundary feature map at H/2 resolution.
        side1: Side-output edge map at H/2 resolution.
        side2_up: Side-output edge map upsampled from H/4 to H/2.
        side3_up: Side-output edge map upsampled from H/8 to H/2.
    """

    def __init__(self, in_ch: int = 64, c1: int = 32, c2: int = 64, out_ch: int = 16):
        super().__init__()

        self.enc1_1 = ConvBNReLU(in_ch, c1)
        self.enc1_2 = ConvBNReLU(c1, c1)
        self.side1 = nn.Conv2d(c1, 1, kernel_size=1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = ConvBNReLU(c1, c2)
        self.enc2_2 = ConvBNReLU(c2, c2)
        self.side2 = nn.Conv2d(c2, 1, kernel_size=1)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = ConvBNReLU(c2, c2)
        self.enc3_2 = ConvBNReLU(c2, c2)
        self.side3 = nn.Conv2d(c2, 1, kernel_size=1)

        self.up2 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        self.dec2_1 = ConvBNReLU(c2 + c2, c2)
        self.dec2_2 = ConvBNReLU(c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1_1 = ConvBNReLU(c1 + c1, c1)
        self.dec1_2 = ConvBNReLU(c1, c1)

        self.edge_fuse = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.out_feat = nn.Conv2d(c1 + 16, out_ch, kernel_size=1)

    def forward(self, f1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, h_half, w_half = f1.shape

        e1 = self.enc1_1(f1)
        e1 = self.enc1_2(e1)
        side1 = self.side1(e1)

        x = self.pool1(e1)
        e2 = self.enc2_1(x)
        e2 = self.enc2_2(e2)
        side2 = self.side2(e2)

        x = self.pool2(e2)
        e3 = self.enc3_1(x)
        e3 = self.enc3_2(e3)
        side3 = self.side3(e3)

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2_1(d2)
        d2 = self.dec2_2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_1(d1)
        d1 = self.dec1_2(d1)

        side2_up = F.interpolate(side2, size=(h_half, w_half), mode="bilinear", align_corners=False)
        side3_up = F.interpolate(side3, size=(h_half, w_half), mode="bilinear", align_corners=False)

        side_cat = torch.cat([side1, side2_up, side3_up], dim=1)
        edge_multi = self.edge_fuse(side_cat)
        edge_feat = self.out_feat(torch.cat([d1, edge_multi], dim=1))

        return edge_feat, side1, side2_up, side3_up


class EdgeEnhancedSegNetUNetBDCNCBAM(nn.Module):
    """
    CT-BDCN-CBAM ablation network for shoreline extraction.

    The model combines a CNN-Transformer backbone, a U-Net-BDCN-style
    boundary branch, gated boundary enhancement, residual boundary refinement,
    and CBAM attention modules placed after the shallow feature block and the
    Transformer-enhanced feature block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dim: int = 128,
        num_heads: int = 4,
        transformer_layers: int = 2,
        cbam_reduction: int = 16,
        cbam_spatial_kernel: int = 7,
    ):
        super().__init__()
        self._last_side_maps = None

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.cbam_f1 = CBAM(64, reduction=cbam_reduction, spatial_kernel=cbam_spatial_kernel)
        self.cbam_f2 = CBAM(embed_dim, reduction=cbam_reduction, spatial_kernel=cbam_spatial_kernel)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.edge_branch = UNetBDCNEdgeBranch(in_ch=64, c1=32, c2=64, out_ch=16)
        self.edge_gate = nn.Conv2d(16, 1, kernel_size=1, bias=True)
        self.edge_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        self.up1 = nn.ConvTranspose2d(embed_dim, 64, kernel_size=2, stride=2)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(64 + 64 + 16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.high_fuse = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.edge_refine = nn.Sequential(
            nn.Conv2d(num_classes + 1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f0 = self.conv0(x)

        f1 = self.conv1(f0)
        f1 = self.cbam_f1(f1)

        f2 = self.conv2(f1)
        batch_size, channels, height, width = f2.shape
        tokens = f2.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        f2_trans = tokens.transpose(1, 2).contiguous().view(batch_size, channels, height, width)
        f2_trans = self.cbam_f2(f2_trans)

        edge_feat, side1, side2_up, side3_up = self.edge_branch(f1)
        self._last_side_maps = (side1, side2_up, side3_up)

        gate = torch.sigmoid(self.edge_gate(edge_feat))
        edge_feat_gated = edge_feat * (0.5 + gate)
        edge_map = self.edge_out(edge_feat_gated)

        up1 = self.up1(f2_trans)
        fusion = torch.cat([up1, f1, edge_feat_gated], dim=1)
        x = self.fuse_conv(fusion)
        x = self.up2(x)
        x = self.high_fuse(torch.cat([x, f0], dim=1))

        seg_logits = self.final_conv(x)
        refine_input = torch.cat([seg_logits, edge_map], dim=1)
        edge_residual = self.edge_refine(refine_input)
        seg_logits = seg_logits + edge_residual

        return seg_logits, edge_map


# Backward-compatible alias for older training scripts.
EdgeEnhancedSegNet_UNetBDCN = EdgeEnhancedSegNetUNetBDCNCBAM
