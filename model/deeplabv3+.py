"""
DeepLabV3+ model definition for shoreline extraction.

This file defines a lightweight DeepLabV3+ implementation with a ResNet-50
backbone. It is used as a comparison model in the CT-BDCN shoreline extraction
project.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

try:
    from torchvision.models import ResNet50_Weights
except ImportError:  # Compatibility with older torchvision versions.
    ResNet50_Weights = None


class ConvBNReLU(nn.Module):
    """Convolution followed by batch normalization and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPPConv(nn.Sequential):
    """Atrous convolution branch used in ASPP."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    """Global image-level pooling branch used in ASPP."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: Optional[tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()
        if atrous_rates is None:
            atrous_rates = (12, 24, 36)

        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        modules.extend(ASPPConv(in_channels, out_channels, rate) for rate in atrous_rates)
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [conv(x) for conv in self.convs]
        return self.project(torch.cat(features, dim=1))


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ for binary or multi-class shoreline segmentation.

    Parameters
    ----------
    num_classes:
        Number of output channels. Use 1 for binary segmentation with
        BCEWithLogitsLoss.
    backbone:
        Backbone name. Currently only "resnet50" is supported.
    pretrained_backbone:
        If True, load ImageNet-pretrained ResNet-50 weights when available.
    output_stride:
        Output stride of the encoder. Supported values are 8 and 16.
    """

    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = "resnet50",
        pretrained_backbone: bool = False,
        output_stride: int = 8,
        aspp_channels: int = 256,
        low_level_channels: int = 48,
    ) -> None:
        super().__init__()
        if backbone.lower() != "resnet50":
            raise ValueError("Only the ResNet-50 backbone is supported in this implementation.")

        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            atrous_rates = (12, 24, 36)
        elif output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
            atrous_rates = (6, 12, 18)
        else:
            raise ValueError("output_stride must be either 8 or 16.")

        weights = None
        if pretrained_backbone:
            if ResNet50_Weights is not None:
                weights = ResNet50_Weights.DEFAULT
            else:
                weights = "IMAGENET1K_V1"

        resnet = resnet50(
            weights=weights,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.aspp = ASPP(2048, out_channels=aspp_channels, atrous_rates=atrous_rates)

        self.low_level_project = nn.Sequential(
            nn.Conv2d(256, low_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            ConvBNReLU(aspp_channels + low_level_channels, 256, kernel_size=3, padding=1),
            ConvBNReLU(256, 256, kernel_size=3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        x = self.stem(x)
        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        high_level_feat = self.layer4(x)

        x = self.aspp(high_level_feat)
        low_level_feat = self.low_level_project(low_level_feat)

        x = F.interpolate(
            x,
            size=low_level_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
