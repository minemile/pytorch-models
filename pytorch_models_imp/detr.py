import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment



class DetrLoss(nn.Module):
    def __init__(self, matcher, num_classes, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        assert "logits" in outputs, "No logits were found in the outputs"
        src_logits = outputs["logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = nn.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "bboxes" in outputs, "No predicted boxes found in outputs"
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["bboxes"][idx]
        target_boxes = torch.cat([t["bboxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = nn.functional.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        return losses


    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes
        }
        assert loss in loss_map, f"Loss {loss} not supported"
        return loss_map[loss](outputs, targets, indices, num_boxes)

class HungarianMatcher(nn.Module):

    @torch.no_grad()
    def forward(self, predictions, targets):
        # prediction {"labels": [...], "bboxes": np.ndarray}
        # targets: dict with "labels" and "bboxes" as lists

        bs, num_queries = predictions["logits"].shape[:2]

        output_probs = predictions['logits'].flatten(0, 1).softmax(-1)
        output_bboxes = predictions['bboxes'].flatten(0, 1)


        gt_labels = torch.cat([target['labels'] for target in targets])
        gt_bboxes = torch.cat([target['bboxes'] for target in targets])


        class_cost = -output_probs[:, gt_labels]
        bbox_loss = torch.cdist(output_bboxes, gt_bboxes)

        matrix_loss = bbox_loss + class_cost
        matrix_loss = matrix_loss.view(bs, num_queries, -1)

        sizes = [len(v["bboxes"]) for v in targets]

        indxes = []
        for i, cost_part in enumerate(matrix_loss.split(sizes, -1)):
            i_match, j_match = linear_sum_assignment(cost_part[i])
            indxes.append((torch.as_tensor(i_match, dtype=torch.int64), torch.as_tensor(j_match, dtype=torch.int64)))
        return indxes


class PositionEncoder(nn.Module):
    def __init__(self, max_hw=50, hidden_dim=256) -> None:
        super().__init__()
        self.row_encoding = nn.Embedding(max_hw, hidden_dim // 2)
        self.col_encoding = nn.Embedding(max_hw, hidden_dim // 2)

    def forward(self, x):
        b, c, h, w = x.shape

        rows_embs = self.row_encoding(torch.arange(0, h, device=x.device))
        rows_embs = rows_embs.unsqueeze(1).repeat(1, 1, w, 1)

        cols_embs = self.col_encoding(torch.arange(0, w, device=x.device))
        cols_embs = cols_embs.unsqueeze(0).repeat(1, h, 1, 1)

        embds = torch.cat((rows_embs, cols_embs), dim=-1).permute(0, 3, 1, 2)
        return embds


class MLPBoxDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DETR(nn.Module):
    def __init__(
        self,
        num_classes,
        num_queries,
        hidden_dim,
        n_head,
        num_encoder_layers,
        num_decoder_layers,
        dropout,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(*models.resnet50(pretrained=True).children())[:-2]
        self.encoder_downsampler = nn.Conv2d(
            2048, hidden_dim, kernel_size=1
        )  # 2048 resnet
        self.position_encoder = PositionEncoder(hidden_dim=hidden_dim)

        self.object_queries = nn.Parameter(torch.randn(num_queries, hidden_dim))

        self.decoder = nn.Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.class_projection = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_projection = MLPBoxDecoder(hidden_dim, hidden_dim, 4)

    def forward(self, x):
        b, c, h, w = x.shape
        features = self.encoder(x)
        features = self.encoder_downsampler(features)

        pos_embds = self.position_encoder(features)

        features += pos_embds
        features = features.flatten(2).permute(0, 2, 1)

        obj_queries = self.object_queries.repeat(b, 1, 1)

        output = self.decoder(features, obj_queries)
        output = self.final_ln(output)

        classes = self.class_projection(output)
        bboxes = self.bbox_projection(output)

        out = {}
        out['logits'] = classes
        out['bboxes'] = torch.sigmoid(bboxes)
        return out


if __name__ == "__main__":
    detr = DETR(30, 256)
    example_input = torch.randn((4, 3, 224, 224))
    output = detr(example_input)
