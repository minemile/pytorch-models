import argparse

from torch import Tensor

import albumentations as A
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

from pytorch_models_imp.detectors.centernet.centernet import CenterNet
from pytorch_models_imp.detectors.centernet.datasets import VOCDataset
from pytorch_models_imp.detectors.centernet.loss import Loss
from pytorch_models_imp.detectors.centernet.utils import get_linear_schedule_with_warmup

DATASET_ROOT = "data/VOCdevkit/VOC2012"
cudnn.benchmark = True
IMAGE_SIZE = 512
DEVICE = "cuda:0"
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)
TRAIN_INPUT_SIZE = (IMAGE_SIZE, IMAGE_SIZE)
TEST_INPUT_SIZE = (IMAGE_SIZE, IMAGE_SIZE)
LEARNING_RATE = 0.02
EPOCHS = 300
WEIGHT_DECAY = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=16, type=int, help="Number of batches")
    parser.add_argument("--num_workers", default=3, type=int, help="Number of workers")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device")
    return parser.parse_args()


def get_augmentations(max_size):
    transform = A.Compose(
        [
            # A.Normalize(),
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(
                min_height=max_size, min_width=max_size, border_mode=cv2.BORDER_CONSTANT
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_visibility=0.5, label_fields=["class_labels"]
        ),
    )
    return transform


def create_dataloders(root, max_size, batch_size, num_workers):
    transform = get_augmentations(max_size)

    voc_train = VOCDataset(root, transform)
    voc_train.cut_dataset_by(0, 1000)
    print(voc_train.objects)
    train_dataloader = DataLoader(
        voc_train,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=voc_train.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )

    voc_validation = VOCDataset(root, transform)
    voc_validation.cut_dataset_by(0, 1000)
    val_dataloader = DataLoader(
        voc_validation,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=voc_validation.collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    args = parse_args()

    device = args.device

    train_dataloader, validation_dataloader = create_dataloders(
        DATASET_ROOT, IMAGE_SIZE, args.batch_size, args.num_workers
    )

    center_net = CenterNet(encoder_name='resnet50').to(device)
    losser = Loss().to(device)
    # optimizer = torch.optim.AdamW(
        # center_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    # )
    optimizer = torch.optim.SGD(center_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)
    for epoch in range(EPOCHS):
        train_loss = 0
        validation_loss = 0
        center_net = center_net.train()
        for tr_counter, batch in enumerate(train_dataloader, 1):

            images, targets = batch
            images = images.to(device)

            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

            optimizer.zero_grad()

            output = center_net(images)
            losses = losser(output, targets)
            loss = sum(losses.values())

            loss.backward()


            torch.nn.utils.clip_grad_norm_(center_net.parameters(), max_norm=32)

            # print(list(optimizer.param_groups)[0]['lr'])

            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # print(f"center heatmap: {losses['loss_center_heatmap']:.5f}. loss_wh: {losses['loss_wh']:.5f}. loss_offset: {losses['loss_offset']:.5f}")
        center_net = center_net.eval()
        for val_counter, batch in enumerate(validation_dataloader, 1):
            images, targets = batch
            images = images.to(device)
            
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

            with torch.no_grad():
                output = center_net(images)
                losses = losser(output, targets)
                loss = losses['loss_center_heatmap'] + losses['loss_wh'] + losses['loss_offset']
                validation_loss += loss.item()

        print(f"Epoch: [{epoch}]. Loss: {(train_loss):4f}\{(validation_loss):4f}. LR: {scheduler.get_last_lr()[0]}")

    torch.save(center_net, f"prod_models/centernet_loss_{train_loss}.mdl")
