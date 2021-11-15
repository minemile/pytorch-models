import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._C import Value
from torch.nn.modules import module
from torch.nn.modules.batchnorm import BatchNorm2d

from pytorch_models_imp.resnet import ResNet


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
        self.out_channels = self.layer4[-1].conv3.out_channels
        self.highres_channels = self.layer1[-1].conv3.out_channels
    
    def get_stages(self):
        return [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        ]

    def forward(self, x):
        stages = self.get_stages()
        init_stage = self.get_init_stage()
        out = init_stage(x)

        feature_maps = []
        feature_maps.append(out)
        for stage in stages:
            out = stage(out)
            feature_maps.append(out)
        return feature_maps
    
    def get_init_stage(self):
        return self.init_stage

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=3):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation,
        )
        bn = nn.BatchNorm2d(out_channels)
        activation = nn.SiLU(inplace=True)
        super().__init__(conv, bn, activation)


class DepthWiseConv2d(nn.Sequential):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)
    
class ASPPDepthWise(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=3):
        sep_conv = DepthWiseConv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=dilation)
        bn = nn.BatchNorm2d(out_channels)
        activation = nn.SiLU(inplace=True)
        super().__init__(sep_conv, bn, activation)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilation_rates=(6, 12, 18), use_separable=True):
        super().__init__()
        self.modules = []

        conv_block = ASPPDepthWise if use_separable else ASPPConv

        self.in_channels = in_channels
        self.out_channels = out_channels
        rate1, rate2, rate3 = dilation_rates
        aspp_1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.modules.append(aspp_1_1)
        self.modules.append(conv_block(in_channels, out_channels, rate1))
        self.modules.append(conv_block(in_channels, out_channels, rate2))
        self.modules.append(conv_block(in_channels, out_channels, rate3))

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

class DeepLabv3PlusDecoder(nn.Module):
    def __init__(self, in_channels, high_res_channels, out_channels=256, dilation_rates=(6, 12, 18)):
        super().__init__()
        self.aspp = ASPP(in_channels, out_channels, dilation_rates)
        self.up_4x = nn.UpsamplingBilinear2d(scale_factor=4)

        high_res_output = 48
        self.downscale = nn.Sequential(
            nn.Conv2d(in_channels=high_res_channels, out_channels=high_res_output, kernel_size=1),
            nn.BatchNorm2d(high_res_output),
            nn.SiLU(inplace=True)
        )

        self.conv1 = ASPPConv(high_res_output + out_channels, out_channels=out_channels, kernel_size=3)
        self.conv2 = ASPPConv(out_channels, out_channels=out_channels, kernel_size=3)

    def forward(self, features, conv2_features):
        aspp_features = self.aspp(features)
        aspp_features = self.up_4x(aspp_features)
        high_res_features = self.downscale(conv2_features)
        out = torch.cat((aspp_features, high_res_features), dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class DeepLabv3(nn.Module):
    def __init__(self, encoder_name, num_classes, out_channels=256, output_stride=16):
        super().__init__()
        self.output_stride = output_stride
        self.encoder = build_encoder(encoder_name)
        # self.aspp = ASPP(self.encoder.out_channels, out_channels=out_channels)
        self.decoder = DeepLabv3PlusDecoder(self.encoder.out_channels, high_res_channels=self.encoder.highres_channels, out_channels=out_channels)
        self.logits = nn.Sequential(nn.Conv2d(out_channels, num_classes, kernel_size=1))

    def forward(self, x):
        features = self.encoder(x)

        features = self.decoder(features[-1], features[1])

        features = self.logits(features)
        return F.interpolate(
            features, size=x.shape[2:], mode="bilinear", align_corners=False
        )
