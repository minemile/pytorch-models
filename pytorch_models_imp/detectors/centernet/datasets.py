import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from pytorch_models_imp.detectors.centernet.utils import (gaussian_radius,
                                                          gen_gaussian_target)
from torch.utils.data import Dataset
from torchvision import transforms


def get_bbox_from_annotation(object):
    x_min = int(object.find("xmin").text)
    y_min = int(object.find("ymin").text)
    x_max = int(object.find("xmax").text)
    y_max = int(object.find("ymax").text)
    return [x_min, y_min, x_max, y_max]


def get_objects_from_folder(folder):
    return sorted(
        [
            object_name.strip(".xml")
            for object_name in os.listdir(folder)
            if not object_name.startswith(".")
        ]
    )


class VOCDataset(Dataset):
    def __init__(self, root, augmentations=None):
        self.root = root
        self.xml_annotations = os.path.join(self.root, "Annotations")
        self.images_folder = os.path.join(self.root, "JPEGImages")
        self.objects = get_objects_from_folder(self.xml_annotations)
        self.class_names = [
            # "__background__",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        self.augmentations = augmentations
        self.down_stride = 4
        self.num_classes = len(self.class_names)

    def cut_dataset_by(self, index_start, index_end):
        self.objects = self.objects[index_start:index_end]

    def parse_annotations(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_name = root.find("filename").text
        objects = root.findall("object")
        bboxes = []
        labels = []
        for obj in objects:
            bboxes.append(get_bbox_from_annotation(obj.find("bndbox")))
            label_name = obj.find("name").text
            if label_name == "__background__":
                raise ValueError(f"Got background at {xml_path}")
            labels.append(self.class_names.index(label_name))
        return image_name, bboxes, labels

    def __len__(self):
        return len(self.objects)

    def _get_image(self, path):
        if not os.path.exists(path):
            raise ValueError(f"No such image at path: {path}")
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Can't read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index):
        object_id = self.objects[index]
        xml_annotation = os.path.join(self.xml_annotations, object_id + ".xml")
        image_name, bboxes, labels = self.parse_annotations(xml_annotation)

        image_path = os.path.join(self.images_folder, image_name)
        image = self._get_image(image_path)

        if self.augmentations is not None:
            image, bboxes, labels = self.transform(image, bboxes, labels)

        bboxes = torch.from_numpy(bboxes)

        h, w, c = image.shape
        feat_h, feat_w = h // self.down_stride, w // self.down_stride

        center_heatmap_target = torch.zeros((self.num_classes, feat_h, feat_w))
        wh_target = torch.zeros((2, feat_h, feat_w))
        offset_target = torch.zeros([2, feat_h, feat_w])
        wh_offset_target_weight = torch.zeros([2, feat_h, feat_w])

        center_x = (bboxes[:, [0]] + bboxes[:, [2]]) / (2 * self.down_stride)
        center_y = (bboxes[:, [1]] + bboxes[:, [3]]) / (2 * self.down_stride)
        gt_centers = torch.cat((center_x, center_y), dim=1)

        for i, ct in enumerate(gt_centers):
            ctx_int, cty_int = ct.int()
            ctx, cty = ct
            scale_box_h = (bboxes[i][3] - bboxes[i][1]) / self.down_stride
            scale_box_w = (bboxes[i][2] - bboxes[i][0]) / self.down_stride

            radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
            radius = max(0, int(radius))
            ind = labels[i]
            gen_gaussian_target(center_heatmap_target[ind], [ctx_int, cty_int], radius)

            wh_target[0, cty_int, ctx_int] = scale_box_w
            wh_target[1, cty_int, ctx_int] = scale_box_h

            offset_target[0, cty_int, ctx_int] = ctx - ctx_int
            offset_target[1, cty_int, ctx_int] = cty - cty_int

            wh_offset_target_weight[:, cty_int, ctx_int] = 1

        # avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight,
            gt_labels=torch.from_numpy(labels)
        )

        image = transforms.ToTensor()(image)
        return image, target_result

    def collate_fn(self, batch):
        images = []
        center_heatmap_targets = []
        wh_targets = []
        offset_targets = []
        wh_offset_target_weights = []
        gt_labels = []
        for image, targets in batch:
            images.append(image)
            center_heatmap_targets.append(targets["center_heatmap_target"])
            wh_targets.append(targets["wh_target"])
            offset_targets.append(targets["offset_target"])
            wh_offset_target_weights.append(targets["wh_offset_target_weight"])
            gt_labels.append(targets['gt_labels'])

        images = torch.stack(images)
        center_heatmap_targets = torch.stack(center_heatmap_targets)
        wh_targets = torch.stack(wh_targets)
        offset_targets = torch.stack(offset_targets)
        wh_offset_target_weights = torch.stack(wh_offset_target_weights)

        targets = dict(
            center_heatmap_target=center_heatmap_targets,
            wh_target=wh_targets,
            offset_target=offset_targets,
            wh_offset_target_weight=wh_offset_target_weights,
            gt_labels=gt_labels
        )
        return images, targets

    def transform(self, image, bboxes, labels):
        output = self.augmentations(image=image, bboxes=bboxes, class_labels=labels)
        return (
            output["image"],
            np.array(output["bboxes"], dtype="float32"),
            np.array(output["class_labels"], dtype="int"),
        )
