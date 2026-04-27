"""
SegFormer-style model for shoreline extraction.

This file contains the network definition used by the SegFormer baseline
in the CT-BDCN shoreline extraction project. The implementation provides
a lightweight SegFormer-style encoder with overlapping patch embedding,
Transformer encoder blocks, and an MLP decoder head for dense prediction.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding implemented with a convolution layer."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm(x_flat)
        x = x_flat.transpose(1, 2).reshape(x.shape[0], -1, h, w)
        return x, h, w


class MixFFN(nn.Module):
    """Feed-forward network with depthwise convolution for local context."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, _ = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(b, -1, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientSelfAttention(nn.Module):
    """Self-attention with optional spatial reduction."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape

        q = self.q(x)
        q = q.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_2d = x.transpose(1, 2).reshape(b, c, h, w)
            x_2d = self.sr(x_2d)
            x_reduced = x_2d.flatten(2).transpose(1, 2)
            x_reduced = self.norm(x_reduced)
        else:
            x_reduced = x

        kv = self.kv(x_reduced)
        kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    """Transformer block used in the SegFormer-style encoder."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class SegFormerEncoder(nn.Module):
    """Hierarchical SegFormer-style encoder."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: List[int] = None,
        num_heads: List[int] = None,
        depths: List[int] = None,
        sr_ratios: List[int] = None,
    ) -> None:
        super().__init__()

        if embed_dims is None:
            embed_dims = [32, 64, 160, 256]
        if num_heads is None:
            num_heads = [1, 2, 5, 8]
        if depths is None:
            depths = [2, 2, 2, 2]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]

        self.patch_embed1 = OverlapPatchEmbed(
            in_channels, embed_dims[0], kernel_size=7, stride=4, padding=3
        )
        self.patch_embed2 = OverlapPatchEmbed(
            embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1
        )
        self.patch_embed3 = OverlapPatchEmbed(
            embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1
        )
        self.patch_embed4 = OverlapPatchEmbed(
            embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1
        )

        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], sr_ratios[0])
            for _ in range(depths[0])
        ])
        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], sr_ratios[1])
            for _ in range(depths[1])
        ])
        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], sr_ratios[2])
            for _ in range(depths[2])
        ])
        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], sr_ratios[3])
            for _ in range(depths[3])
        ])

        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    @staticmethod
    def _run_stage(
        x: torch.Tensor,
        patch_embed: nn.Module,
        blocks: nn.ModuleList,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        x, h, w = patch_embed(x)
        b, c, _, _ = x.shape
        x_tokens = x.flatten(2).transpose(1, 2)

        for block in blocks:
            x_tokens = block(x_tokens, h, w)

        x_tokens = norm(x_tokens)
        x = x_tokens.transpose(1, 2).reshape(b, c, h, w)
        return x

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self._run_stage(x, self.patch_embed1, self.block1, self.norm1)
        features.append(x)

        x = self._run_stage(x, self.patch_embed2, self.block2, self.norm2)
        features.append(x)

        x = self._run_stage(x, self.patch_embed3, self.block3, self.norm3)
        features.append(x)

        x = self._run_stage(x, self.patch_embed4, self.block4, self.norm4)
        features.append(x)

        return features


class SegFormerHead(nn.Module):
    """MLP decoder head for multi-scale feature fusion."""

    def __init__(
        self,
        in_channels: List[int],
        channels: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(in_channels), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor], output_size) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projected = []

        for feature, proj in zip(features, self.proj_layers):
            x = proj(feature)
            if x.shape[-2:] != target_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(x)

        x = torch.cat(projected, dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return x


class SegFormerCoast(nn.Module):
    """SegFormer-style semantic segmentation network for shoreline extraction."""

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        embed_dims: List[int] = None,
    ) -> None:
        super().__init__()

        if embed_dims is None:
            embed_dims = [32, 64, 160, 256]

        self.encoder = SegFormerEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
        )
        self.decode_head = SegFormerHead(
            in_channels=embed_dims,
            channels=128,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        logits = self.decode_head(features, output_size=input_size)
        return logits
