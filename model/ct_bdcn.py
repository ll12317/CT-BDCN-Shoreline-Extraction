# -*- coding: utf-8 -*-
"""
Model definition for CT-BDCN shoreline extraction.

This file contains the network components used by the CT-BDCN model,
including the CNN-Transformer backbone, the U-Net-BDCN-style multi-scale
boundary branch, gated boundary enhancement, and boundary residual refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution, batch normalization, and ReLU activation block."""

    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1):
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

    def forward(self, x):
        return self.block(x)


class UNetBDCN_EdgeBranch(nn.Module):
    """
    U-Net-BDCN-style multi-scale boundary branch.

    Parameters
    ----------
    in_ch : int
        Number of input channels from the shallow encoder feature.
    c1 : int
        Number of channels in the first boundary encoding stage.
    c2 : int
        Number of channels in the deeper boundary encoding stages.
    out_ch : int
        Number of output boundary feature channels.

    Input
    -----
    f1 : torch.Tensor
        Shallow feature map with shape [B, 64, H/2, W/2].

    Output
    ------
    edge_feat : torch.Tensor
        Fused boundary feature map with shape [B, out_ch, H/2, W/2].
    side1, side2_up, side3_up : torch.Tensor
        Multi-scale side-output boundary maps upsampled to H/2 and W/2.
    """

    def __init__(self, in_ch=64, c1=32, c2=64, out_ch=16):
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

    def forward(self, f1):
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

        side2_up = F.interpolate(
            side2,
            size=(h_half, w_half),
            mode="bilinear",
            align_corners=False,
        )
        side3_up = F.interpolate(
            side3,
            size=(h_half, w_half),
            mode="bilinear",
            align_corners=False,
        )

        side_cat = torch.cat([side1, side2_up, side3_up], dim=1)
        edge_multi = self.edge_fuse(side_cat)
        feat_cat = torch.cat([d1, edge_multi], dim=1)
        edge_feat = self.out_feat(feat_cat)

        return edge_feat, side1, side2_up, side3_up


class EdgeEnhancedSegNet_UNetBDCN(nn.Module):
    """
    CT-BDCN network for boundary-enhanced shoreline extraction.

    The model uses a CNN encoder for local spatial features, a Transformer
    bottleneck for global contextual modeling, and a U-Net-BDCN-style boundary
    branch to improve shoreline localization. Boundary features are injected
    into the decoder through gated enhancement and residual refinement.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        embed_dim=128,
        num_heads=4,
        transformer_layers=2,
    ):
        super().__init__()
        self._last_side_maps = None

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
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

        self.edge_branch = UNetBDCN_EdgeBranch(
            in_ch=64,
            c1=32,
            c2=64,
            out_ch=16,
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

        self.high_fuse = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        self.edge_refine = nn.Sequential(
            nn.Conv2d(num_classes + 1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

    def forward(self, x):
        f0 = self.conv0(x)
        f1 = self.conv1(f0)
        f2 = self.conv2(f1)

        batch_size, channels, h, w = f2.shape
        tokens = f2.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        f2_trans = tokens.transpose(1, 2).view(batch_size, channels, h, w)

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
