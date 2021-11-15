import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(
        in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1), stride=stride
    )


def conv3x3(in_ch, out_ch, padding=1, stride=1, dilation=1):
    return nn.Conv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=(3, 3),
        padding=padding,
        stride=stride,
        dilation=dilation
    )


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, inter_channels, stride=1, dilation=1, padding=1, downsample_identity=None):
        super(BottleNeck, self).__init__()
        self.conv1 = conv1x1(in_channels, inter_channels)
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.conv2 = conv3x3(inter_channels, inter_channels, stride=stride, dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm2d(inter_channels)

        self.conv3 = conv1x1(inter_channels, inter_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(inter_channels * self.expansion)

        self.downsample_identity = downsample_identity
        self.stride = stride

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample_identity is not None:
            identity = self.downsample_identity(identity)
        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    expansion = 4

    def __init__(self, num_blocks, in_channels, num_classes, dilation=1):
        super().__init__()
        self.out_channels = 64
        self.num_classes = num_classes
        self.dilation = dilation

        self.init_stage = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.out_channels,
                kernel_size=(7, 7),
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

        self.layer1 = self._make_layer(
            num_blocks=num_blocks[0], in_channels=64, inter_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            num_blocks=num_blocks[1], in_channels=64 * self.expansion, inter_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            num_blocks=num_blocks[2], in_channels=128 * self.expansion, inter_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            num_blocks=num_blocks[3], in_channels=256 * self.expansion, inter_channels=512, stride=2, dilation=dilation
        )

    def _make_layer(self, num_blocks, in_channels, inter_channels, stride, padding=1, dilation=1):
        layers = []
        if dilation > 1:
            padding = dilation
            stride = 1

        if stride != 1 or in_channels != inter_channels * self.expansion:
            downsample_identity = nn.Sequential(
                conv1x1(in_ch=in_channels, out_ch=inter_channels * self.expansion, stride=stride),
                nn.BatchNorm2d(inter_channels * self.expansion),
            )
            layer = BottleNeck(
                in_channels=in_channels,
                inter_channels=inter_channels,
                stride=stride,
                padding=padding,
                dilation=dilation,
                downsample_identity=downsample_identity,
            )
            layers.append(layer)
            num_blocks -= 1

        in_channels = inter_channels * 4
        for i in range(num_blocks):
            layer = BottleNeck(in_channels=in_channels, inter_channels=inter_channels, dilation=dilation, padding=padding)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init_stage(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet50(in_channels, num_classes):
    return ResNet([3, 4, 6, 3], in_channels, num_classes)


def resnet101(in_channels, num_classes):
    return ResNet([3, 4, 23, 3], in_channels, num_classes)


def resnet152(in_channels, num_classes):
    return ResNet([3, 8, 36, 3], in_channels, num_classes)
