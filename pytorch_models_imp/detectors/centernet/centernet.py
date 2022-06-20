import torch
import torch.nn.functional as F
from pytorch_models_imp.detectors.centernet.utils import (
    get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat)
from pytorch_models_imp.modules import ConvBNActivation
from torch import nn
from torchvision.models import resnet18, resnet50


class ResnetEncoder(nn.Module):
    def __init__(self, encoder_name) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        if self.encoder_name == "resnet18":
            self.resnet = resnet18(pretrained=True)
        elif self.encoder_name == "resnet50":
            self.resnet = resnet50(pretrained=True)
        self.init_layers = nn.Sequential(*list(self.resnet.children())[:4])
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

    @property
    def output_channels(self):
        if self.encoder_name == "resnet18":
            return self.resnet.layer4[-1].conv2.out_channels
        else:
            return self.resnet.layer4[-1].conv3.out_channels

    def forward(self, x):
        x = self.init_layers(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, inplanes):
        super(Decoder, self).__init__()
        # backbone output: [b, 2048, _h, _w]
        self.inplanes = inplanes
        self.deconv_layers = self._make_deconv_layer(
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

    @property
    def output_channels(self):
        return self.deconv_layers[-3].out_channels

    def _make_deconv_layer(self, num_filters, num_kernels):
        layers = []
        for i in range(len(num_filters)):
            kernel = num_kernels[i]
            padding = 0 if kernel == 2 else 1
            output_padding = 1 if kernel == 3 else 0
            planes = num_filters[i]

            layers.append(
                ConvBNActivation(
                    in_planes=self.inplanes,
                    out_planes=planes,
                    kernel_size=3,
                    padding=1,
                    activation_layer=nn.ReLU,
                )
            )

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class Head(nn.Module):
    def __init__(self, in_channels, num_classes=80):
        super(Head, self).__init__()

        self.cls_head = self._make_head(in_channels, num_classes)
        self.wh_head = self._make_head(in_channels, 2)
        self.reg_head = self._make_head(in_channels, 2)

    def _make_head(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x):
        hm = self.cls_head(x).sigmoid()
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        return hm, wh, offset


class CenterNet(nn.Module):
    def __init__(self, encoder_name, num_classes=20):
        super().__init__()
        self.backbone = ResnetEncoder(encoder_name)
        self.upsample = Decoder(self.backbone.output_channels)
        self.head = Head(self.upsample.output_channels, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feat = feats[-1]
        return self.head(self.upsample(feat))

    @torch.no_grad()
    def detect(self, images, k=100):
        center_heatmap_preds, wh_preds, offset_preds = self(images)

        result_list = []
        for i in range(len(images)):
            result_list.append(self._get_bboxes_single(
                center_heatmap_preds[i:i+1],
                wh_preds[i:i+1],
                offset_preds[i:i+1],
                k,
                images.shape
            ))
        return result_list
    
    def _get_bboxes_single(self,
        center_heatmap_pred,
        wh_pred,
        offset_pred,
        k,
        img_shape
    ):
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            img_shape,
            k=k,
            kernel=3)
        return batch_det_bboxes, batch_labels


    def decode_heatmap(self,
        center_heatmap_pred,
        wh_pred,
        offset_pred,
        img_shape,
        k=100,
        kernel=3
    ):
        height, width = center_heatmap_pred.shape[2:]
        _, _, inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)

        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)

        return batch_bboxes, batch_topk_labels
