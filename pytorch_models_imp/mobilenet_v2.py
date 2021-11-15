import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU6
from torch.nn.modules.batchnorm import BatchNorm2d

mobile_net_config = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, expansion, stride):
        super(InvertedResidual, self).__init__()
        self.expansion = expansion
        self.inner_ch = in_ch * expansion
        self.use_residual = stride == 1 and in_ch == out_ch
        if self.expansion != 1:
            self.init_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch, out_channels=self.inner_ch, kernel_size=(1, 1)
                ),
                nn.BatchNorm2d(self.inner_ch),
                nn.ReLU6(inplace=True),
            )
        else:
            self.init_layer = nn.Identity()

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.inner_ch,
                out_channels=self.inner_ch,
                kernel_size=(3, 3),
                groups=self.inner_ch,
                padding=1,
                stride=stride,
            ),
            nn.BatchNorm2d(self.inner_ch),
            nn.ReLU6(inplace=True),
        )
        self.linear = nn.Sequential(
            nn.Conv2d(
                in_channels=self.inner_ch, out_channels=out_ch, kernel_size=(1, 1)
            ),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.init_layer(x)
        out = self.depthwise_conv(out)
        out = self.linear(out)
        if self.use_residual:
            indentity = x
            return out + indentity
        return out


class MobileNetV2(nn.Module):
    def __init__(self, layers_config, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_out_channels = 32
        self.last_out_channels = 1280
        self.layers_config = layers_config
        assert len(layers_config[0]) == 4

        self.init_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.init_out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=2,
        )
        self.init_bn = nn.BatchNorm2d(self.init_out_channels)
        self.init_activation = nn.ReLU6(inplace=True)
        self.init_block = nn.Sequential(
            self.init_conv, self.init_bn, self.init_activation
        )

        in_ch = self.init_out_channels
        blocks = []
        for t, c, n, s in self.layers_config:
            first_layer = InvertedResidual(in_ch, c, t, s)
            blocks.append(first_layer)
            for _ in range(n - 1):
                layer = InvertedResidual(c, c, t, 1)
                blocks.append(layer)
            in_ch = c
        self.blocks = nn.Sequential(*blocks)

        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=self.last_out_channels,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(self.last_out_channels),
            nn.ReLU6(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.last_out_channels,
                out_channels=num_classes,
                kernel_size=(1, 1),
            ),
        )

    def forward(self, x):
        out = self.init_block(x)
        out = self.blocks(out)
        out = self.final_conv(out)
        out = self.classifier(out)
        out = torch.flatten(out, 1)
        return out

