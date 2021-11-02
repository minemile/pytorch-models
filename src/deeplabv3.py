import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
from torch.nn.modules.batchnorm import BatchNorm2d
from src.resnet import ResNet


def build_encoder(name):
    if name == "resnet50":
        encoder = ResnetEncoder([3, 4, 6, 3], 3, 1, dilation=2)
    elif name == "resnet101":
        encoder = ResnetEncoder([3, 4, 23, 3], 3, 1, dilation=2)
    else:
        raise ValueError("No such encoder")
    return encoder


class ResnetEncoder(ResNet):
    def __init__(self, num_blocks, in_channels, num_classes, dilation):
        super().__init__(num_blocks, in_channels, num_classes, dilation=dilation)
        self.out_channels = self.layer4[2].conv3.out_channels

    def forward(self, x):
        out = self.init_stage(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )
        bn = nn.BatchNorm2d(out_channels)
        activation = nn.SiLU(inplace=True)
        super().__init__(conv, bn, activation)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilation_rates=(6, 12, 18)):
        super().__init__()
        self.modules = []

        self.in_channels = in_channels
        self.out_channels = out_channels
        rate1, rate2, rate3 = dilation_rates
        aspp_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.modules.append(aspp_1)
        self.modules.append(ASPPConv(in_channels, out_channels, rate1))
        self.modules.append(ASPPConv(in_channels, out_channels, rate2))
        self.modules.append(ASPPConv(in_channels, out_channels, rate3))
        self.pooling_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels * 5, out_channels=out_channels, kernel_size=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        outs = []
        size = x.shape[2:]
        for module in self.modules:
            outs.append(module(x))
        pool_features = self.pooling_layer(x)
        pool_features = F.interpolate(
            pool_features, size=size, mode="bilinear", align_corners=False
        )
        outs.append(pool_features)
        outs = torch.cat(outs, dim=1)
        outs = self.final_conv(outs)
        return outs


class DeepLabv3(nn.Module):
    def __init__(self, encoder_name, num_classes, out_channels=256, output_stride=16):
        super().__init__()
        self.output_stride = output_stride
        self.encoder = build_encoder(encoder_name)
        self.aspp = ASPP(self.encoder.out_channels, out_channels=out_channels)
        self.logits = nn.Sequential(nn.Conv2d(out_channels, num_classes, kernel_size=1))

    def forward(self, x):
        features = self.encoder(x)
        features = self.aspp(features)
        features = self.logits(features)
        return F.interpolate(
            features, size=x.shape[2:], mode="bilinear", align_corners=False
        )
