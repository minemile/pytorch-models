import argparse
import os

import PIL
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
from pytorch_models_imp.datasets.penn_funn import PennFudanDataset
from pytorch_models_imp.detr import DETR, PositionEncoder, HungarianMatcher, DetrLoss

import albumentations as A


cudnn.benchmark = True
IMAGE_SIZE = 224
DEVICE = "cuda:0"
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)
TRAIN_INPUT_SIZE = (224, 224)
TEST_INPUT_SIZE = (224, 224)


# transformer
HEADS = 8
DROPOUT = 0
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_QUERIES = 10
HIDDEN_DIM = 256


USE_TIMM = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--total_steps", type=int, help="Number of steps")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--model_name", type=str, help="Timm model name")
    parser.add_argument(
        "--root", default="./data", type=str, help="Root to save images"
    )
    parser.add_argument("--batch_size", default=512, type=int, help="Number of batches")
    parser.add_argument("--num_workers", default=3, type=int, help="Number of workers")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device")
    return parser.parse_args()

def save_model(name, model):
    # state = {"net": model.state_dict()}
    output_folder = "model_outputs/detection_transforer"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    torch.save(model, os.path.join(output_folder, name))


def create_datasetloader(root, batch_size, num_workers):
    transform_train = A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    transform_test = A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    imgs_list = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))[:100]
    masks_list = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))[:100]

    k = 0.7
    train_part = int(len(imgs_list) * k)
    train_imgs, train_masks = imgs_list[:train_part], masks_list[:train_part]
    val_imgs, val_masks = imgs_list[train_part:], masks_list[train_part:]

    train_dataset = PennFudanDataset(root, train_imgs, train_masks, transform_train)
    val_dataset = PennFudanDataset(root, val_imgs, val_masks, transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
        collate_fn=val_dataset.collate_fn
    )

    return train_loader, val_loader


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    epochs = args.epochs
    total_steps = args.total_steps

    train_loader, val_loader = create_datasetloader(
        args.root, args.batch_size, args.num_workers
    )

    # num_classes = len(train_loader.dataset.classes)
    num_classes = 1

    detr = DETR(num_classes, NUM_QUERIES, HIDDEN_DIM, HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROPOUT)
    matcher = HungarianMatcher()
    losses = DetrLoss(matcher, num_classes, eos_coef=0.1, losses=['labels', 'boxes'])



    optimizer = torch.optim.AdamW(
        detr.parameters(), args.learning_rate, weight_decay=1e-4
    )

    pct_start = 0.05
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     pct_start=pct_start,
    #     total_steps=total_steps,
    #     max_lr=args.learning_rate,
    #     # steps_per_epoch=len(train_loader),
    #     # epochs=args.epochs,
    # )

    # print(
    #     f"There is {total_steps} steps. For warmup there is {total_steps * pct_start} steps"
    # )
    # epochs = int(total_steps / len(train_loader))
    # print(
    #     f"It will take {epochs} epochs for {total_steps} steps with {len(train_loader)} step"
    # )

    detr.to(device)
    # detr = nn.DataParallel(detr)

    for epoch in range(epochs):
        train_metric = 0
        val_metric = 0

        val_accuracy = 0
        val_total = 0

        for train_step, (images, targets) in enumerate(train_loader, 1):
            detr.train()
            images = images.to(device)
            for indx, target in enumerate(targets):
                targets[indx] = {k: v.to(device) for k, v in target.items()}

            out = detr(images)
            optimizer.zero_grad()
            loss = losses(out, targets)
            loss = sum(loss.values())

            torch.nn.utils.clip_grad_norm_(detr.parameters(), max_norm=1)

            train_metric += loss.item()
            loss.backward()
            optimizer.step()

        for val_step, (images, targets) in enumerate(val_loader, 1):
            detr.eval()
            images = images.to(device)
            for indx, target in enumerate(targets):
                targets[indx] = {k: v.to(device) for k, v in target.items()}
            with torch.no_grad():
                out = detr(images)
                loss = losses(out, targets)
                loss = sum(loss.values())

            # predict = out.argmax(dim=1)
            # val_accuracy += (predict == targets).sum().item()
            # val_total += targets.size(0)

            val_metric += loss.item()
        train_loss = train_metric / train_step
        val_loss = val_metric / val_step
        # accuracy = val_accuracy / val_total
        print(
            f"[{epoch}]. Train loss: {train_loss:.4f}\
             Validation loss: {val_loss:.4f}"
        )
    save_model("transformer_model", detr)
