from pytorch_models_imp.detectors.utils import intersection_over_union, mean_average_precision, nms
import torch
import pytest


class TestIOU():
    def test_diff_bboxes(self):
        t1_box1 = torch.tensor([[1, 1, 3, 3]])
        t1_box2 = torch.tensor([[2, 0, 4, 2]])
        assert intersection_over_union(t1_box1, t1_box2, box_format="corners")[
            0, 0] == pytest.approx(1/7, 0.0001)

    def test_equal_bboxes(self):
        box = torch.tensor([[1, 2, 3, 4]])
        assert intersection_over_union(
            box, box)[0, 0] == pytest.approx(1, 0.0001)

    def test_multiple_bboxes(self):
        t12_bboxes1 = torch.tensor(
            [
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 3, 2],
            ]
        )
        t12_bboxes2 = torch.tensor(
            [
                [3, 0, 5, 2],
                [3, 0, 5, 2],
                [0, 3, 2, 5],
                [2, 0, 5, 2],
                [1, 1, 3, 3],
                [1, 1, 3, 3],
            ]
        )
        t12_correct_ious = torch.tensor([0, 0, 0, 0, 1 / 7, 0.25])

        assert (intersection_over_union(t12_bboxes1,
                t12_bboxes2, box_format='corners').flatten() - t12_correct_ious <= 0.01).all()


class TestNSM():

    def test_nms(self):
        t1_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [1, 0.8, 0.5, 0.5, 0.2, 0.4],
            [1, 0.7, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        c1_boxes = [[1, 1, 0.5, 0.45, 0.4, 0.5],
                    [1, 0.7, 0.25, 0.35, 0.3, 0.1]]

        result = nms(t1_boxes, threshold=0.2, iou_threshold=7 /
                   20, box_format="midpoint")
        # print(result)
        assert result == c1_boxes


class TestMAP():

    def test_map(self):
        t1_preds = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        t1_targets = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]

        correct_map = 1.0

        mean_avg_prec = mean_average_precision(
            t1_preds,
            t1_targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=1,
        )

        assert mean_avg_prec == pytest.approx(correct_map, 0.001)

