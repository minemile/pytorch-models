import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial


def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss


class Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=1.0):
        super().__init__()
        self.focal_loss = gaussian_focal_loss
        self.loss_wh = partial(F.l1_loss, reduction="none")
        self.loss_offset = partial(F.l1_loss, reduction="none")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.eps = torch.finfo(torch.float32).eps

    def forward(self, pred, targets):
        center_heatmap_pred = pred[0]
        wh_pred = pred[1]
        offset_pred = pred[2]

        center_heatmap_target = targets["center_heatmap_target"]
        wh_target = targets["wh_target"]
        offset_target = targets["offset_target"]
        wh_offset_target_weight = targets["wh_offset_target_weight"]

        avg_factor = max(1, (center_heatmap_target == 1).sum())

        loss_center_heatmap = self.focal_loss(
            center_heatmap_pred, center_heatmap_target
        ).sum() / (self.eps + avg_factor)

        loss_wh = (wh_offset_target_weight * self.loss_wh(wh_pred, wh_target)).sum() / (
            self.eps + 2 * avg_factor
        )

        loss_offset = (
            wh_offset_target_weight * self.loss_offset(offset_pred, offset_target)
        ).sum() / (self.eps + 2 * avg_factor)

        return dict(
            loss_center_heatmap=self.alpha * loss_center_heatmap.mean(),
            loss_wh=self.beta * loss_wh.mean(),
            loss_offset=self.gamma * loss_offset.mean(),
        )
