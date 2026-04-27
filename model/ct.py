# -*- coding: utf-8 -*-
"""
Baseline CNN-Transformer segmentation network for shoreline extraction.

This file defines the ConvTransformerSegNet model used as the CT baseline
in the CT-BDCN shoreline extraction project.
"""

import torch
import torch.nn as nn


class ConvTransformerSegNet(nn.Module):
    """CNN-Transformer network for binary shoreline segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dim: int = 128,
        num_heads: int = 4,
        transformer_layers: int = 2,
    ):
        super().__init__()

        # CNN encoder: downsample the input to 1/2 and then 1/4 resolution.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
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

        # Transformer bottleneck for global contextual modeling.
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

        # Decoder with skip connection from the shallow CNN feature.
        self.up1 = nn.ConvTranspose2d(embed_dim, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation."""

        # CNN feature extraction.
        f1 = self.conv1(x)   # [B, 64, H/2, W/2]
        f2 = self.conv2(f1)  # [B, embed_dim, H/4, W/4]

        # Convert the feature map into a token sequence for the Transformer.
        batch_size, channels, height, width = f2.shape
        tokens = f2.flatten(2).transpose(1, 2)  # [B, L, C]
        tokens = self.transformer(tokens)
        f2_trans = tokens.transpose(1, 2).view(batch_size, channels, height, width)

        # Decode and fuse shallow features.
        up1 = self.up1(f2_trans)               # [B, 64, H/2, W/2]
        fused = torch.cat([up1, f1], dim=1)
        x = self.dec_conv1(fused)
        x = self.up2(x)                        # [B, 32, H, W]
        seg_logits = self.final_conv(x)        # [B, num_classes, H, W]

        return seg_logits
