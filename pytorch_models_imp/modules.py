import torch.nn as nn

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation_layer: str = None,
        padding: int = 0,
        dilation: int = 1,
        dimension: int = 2,
    ):
        if dimension == 2:
            norm_layer = nn.BatchNorm2d(out_planes)
            conv_layer = nn.Conv2d
        elif dimension == 1:
            norm_layer = nn.BatchNorm1d(out_planes)
            conv_layer = nn.Conv1d
        else:
            raise ValueError(f"Invalid dimension: {dimension}")
        padding = (kernel_size - 1) // 2
        if activation_layer is None:
            activation_layer = nn.SiLU
        super().__init__(
            conv_layer(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            ),
            norm_layer,
            activation_layer(inplace=True),
        )