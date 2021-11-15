from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU6, SiLU
from torch.nn.modules.batchnorm import BatchNorm2d


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        padding: int = 0,
        dilation: int = 1,
    ):
        padding = (kernel_size[0] - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(out_planes),
            activation_layer(inplace=True),
        )


class SEBlock(nn.Module):
    def __init__(self, in_ch, r=4):
        super().__init__()
        self.in_ch = in_ch
        self.r = r
        self.exitation_ch = self.in_ch // self.r
        self.fc1 = nn.Conv2d(in_ch, self.exitation_ch, 1)
        self.activation = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(self.exitation_ch, in_ch, 1)
        self.hard_sigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, 1)
        out = self.activation(self.fc1(out))
        out = self.hard_sigmoid(self.fc2(out))
        return x * out


class MobileNetLayerConfig:
    def __init__(self, kernel_size, exp_size, out_ch, use_se, activation, stride):
        assert activation in ["HS", "RE"]
        assert stride in [1, 2]

        self.exp_size = exp_size
        self.out_ch = out_ch
        self.use_se = use_se
        self.activation = activation
        self.kernel_size = kernel_size
        self.stride = stride


class InvertedResidualV3(nn.Module):
    def __init__(
        self, in_ch, out_ch, activation, kernel_size, inner_ch, use_se, stride
    ):
        super().__init__()
        self.inner_ch = inner_ch
        self.use_residual = stride == 1 and in_ch == out_ch
        self.activation = nn.ReLU if activation == "RE" else nn.SiLU
        self.kernel_size = (kernel_size, kernel_size)
        self.out_ch = out_ch
        self.use_se = use_se

        if self.use_se:
            self.se_block = SEBlock(in_ch=self.inner_ch)

        if in_ch != inner_ch:
            self.up_channel_conv = ConvBNActivation(
                in_planes=in_ch,
                out_planes=self.inner_ch,
                kernel_size=(1, 1),
                activation_layer=self.activation,
            )
        else:
            self.up_channel_conv = nn.Identity()
        self.depthwise_conv = ConvBNActivation(
            in_planes=self.inner_ch,
            out_planes=self.inner_ch,
            kernel_size=self.kernel_size,
            activation_layer=self.activation,
            groups=self.inner_ch,
            stride=stride,
        )
        self.linear = ConvBNActivation(
            in_planes=self.inner_ch,
            out_planes=out_ch,
            kernel_size=(1, 1),
            activation_layer=nn.Identity,
        )

    def forward(self, x):
        out = self.up_channel_conv(x)
        out = self.depthwise_conv(out)
        if self.use_se:
            out = self.se_block(out)
        out = self.linear(out)
        if self.use_residual:
            indentity = x
            return out + indentity
        return out


class MobileNetV3(nn.Module):
    def __init__(
        self,
        layers_config: MobileNetLayerConfig,
        in_channels,
        num_classes,
        last_channel,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_out_channels = 16
        self.layers_config = layers_config
        activation = nn.SiLU

        self.init_conv = ConvBNActivation(
            in_planes=in_channels,
            out_planes=self.init_out_channels,
            activation_layer=activation,
            kernel_size=(3, 3),
            stride=2,
        )
        in_ch = self.init_out_channels

        blocks = []
        for config in self.layers_config:
            first_layer = InvertedResidualV3(
                in_ch=in_ch,
                out_ch=config.out_ch,
                activation=config.activation,
                kernel_size=config.kernel_size,
                use_se=config.use_se,
                inner_ch=config.exp_size,
                stride=config.stride,
            )
            blocks.append(first_layer)
            in_ch = config.out_ch
        self.blocks = nn.Sequential(*blocks)

        self.before_pool_ch = 6 * layers_config[-1].out_ch
        self.before_pool = ConvBNActivation(
            in_planes=in_ch,
            out_planes=self.before_pool_ch,
            kernel_size=(1, 1),
            activation_layer=activation,
        )

        self.last_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=self.before_pool_ch,
                out_channels=last_channel,
                kernel_size=(1, 1),
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=last_channel, out_channels=num_classes, kernel_size=(1, 1)
            ),
        )

    def forward(self, x):
        out = self.init_conv(x)
        out = self.blocks(out)
        out = self.before_pool(out)
        out = self.last_layers(out)
        out = torch.flatten(out, 1)
        return out


def mobilenet_v3_small(in_channels, num_classes):
    layer_configs = []
    layer_configs.append(MobileNetLayerConfig(3, 16, 16, True, "RE", 2))
    layer_configs.append(MobileNetLayerConfig(3, 72, 24, False, "RE", 2))
    layer_configs.append(MobileNetLayerConfig(3, 88, 24, False, "RE", 1))
    layer_configs.append(MobileNetLayerConfig(5, 96, 40, True, "HS", 2))
    layer_configs.append(MobileNetLayerConfig(5, 240, 40, True, "HS", 1))
    layer_configs.append(MobileNetLayerConfig(5, 240, 40, True, "HS", 1))
    layer_configs.append(MobileNetLayerConfig(5, 120, 48, True, "HS", 1))
    layer_configs.append(MobileNetLayerConfig(5, 144, 48, True, "HS", 1))
    layer_configs.append(MobileNetLayerConfig(5, 288, 96, True, "HS", 2))
    layer_configs.append(MobileNetLayerConfig(5, 576, 96, True, "HS", 1))
    layer_configs.append(MobileNetLayerConfig(5, 576, 96, True, "HS", 1))
    return MobileNetV3(layer_configs, in_channels, num_classes, last_channel=1024)


def mobilenet_v3_large(in_channels, num_classes):
    layer_configs = []
    layer_configs.append(MobileNetLayerConfig(3, 16, 16, False, "RE", 1))
    layer_configs.append(MobileNetLayerConfig(3, 64, 24, False, "RE", 2))
    layer_configs.append(MobileNetLayerConfig(3, 72, 24, False, "RE", 1))
    layer_configs.append(MobileNetLayerConfig(5, 72, 40, True, "RE", 2))
    layer_configs.append(MobileNetLayerConfig(5, 120, 40, True, "RE", 1))
    layer_configs.append(MobileNetLayerConfig(5, 120, 40, True, "RE", 1))
    layer_configs.append(MobileNetLayerConfig(3, 240, 80, False, "HS", 2))
    layer_configs.append(MobileNetLayerConfig(3, 200, 80, False, "HS", 1))
    layer_configs.append(MobileNetLayerConfig(3, 184, 80, False, "HS", 1))
    layer_configs.append(MobileNetLayerConfig(3, 184, 80, False, "HS", 1))
    layer_configs.append(MobileNetLayerConfig(3, 480, 112, True, "HS", 1))
    layer_configs.append(MobileNetLayerConfig(3, 672, 112, True, "HS", 1))
    layer_configs.append(MobileNetLayerConfig(5, 672, 160, True, "HS", 2))
    layer_configs.append(MobileNetLayerConfig(5, 960, 160, True, "HS", 1))
    layer_configs.append(MobileNetLayerConfig(5, 960, 160, True, "HS", 1))
    return MobileNetV3(layer_configs, in_channels, num_classes, last_channel=1280)