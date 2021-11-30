import argparse
import os

import PIL
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from pytorch_models_imp.visual_transformer import VisionTransformer

cudnn.benchmark = True
IMAGE_SIZE = 224
DEVICE = "cuda:0"
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)
TRAIN_INPUT_SIZE = (224, 224)
TEST_INPUT_SIZE = (224, 224)


# transformer
FORWARD_EXPANSION = 4
NUM_LAYERS = 6
HEADS = 3
PATCH_SIZE = 16
EMBEDDING_SIZE = 192
DROPOUT = 0.1

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


def create_datasetloader(is_train, shuffle, root, transform, batch_size, num_workers):
    imageset = torchvision.datasets.CIFAR100(
        root=root, train=is_train, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        imageset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    return loader


def save_model(name, model, accuracy):
    state = {"net": model.state_dict(), "acc": accuracy}
    output_folder = "model_outputs/cifar_transforer"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    torch.save(state, os.path.join(output_folder, name))


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    epochs = args.epochs
    total_steps = args.total_steps

    transform_train = transforms.Compose(
        [
            transforms.Resize(size=256, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(size=TRAIN_INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(size=TEST_INPUT_SIZE, interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )

    train_loader = create_datasetloader(
        is_train=True,
        shuffle=True,
        root=args.root,
        transform=transform_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = create_datasetloader(
        is_train=False,
        shuffle=False,
        root=args.root,
        transform=transform_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(train_loader.dataset.classes)

    if not args.model_name.startswith("my"):
        vision_transformer = timm.create_model(
            args.model_name, pretrained=True, num_classes=num_classes
        )
    else:
        vision_transformer = VisionTransformer(
            IMAGE_SIZE,
            num_classes,
            PATCH_SIZE,
            HEADS,
            EMBEDDING_SIZE,
            FORWARD_EXPANSION,
            NUM_LAYERS,
            DROPOUT,
        )

    loss_f = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(
    #     vision_transformer.parameters(), args.learning_rate, weight_decay=1e-5
    # )
    optimizer = torch.optim.SGD(
    vision_transformer.parameters(),
    lr=args.learning_rate,
    momentum=0.9,
    weight_decay=1e-5,
    )
    pct_start = 0.05
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        pct_start=pct_start,
        total_steps=total_steps,
        max_lr=args.learning_rate,
        # steps_per_epoch=len(train_loader),
        # epochs=args.epochs,
    )

    print(
        f"There is {total_steps} steps. For warmup there is {total_steps * pct_start} steps"
    )
    epochs = int(total_steps / len(train_loader))
    print(
        f"It will take {epochs} epochs for {total_steps} steps with {len(train_loader)} step"
    )

    vision_transformer.to(device)
    vision_transformer = nn.DataParallel(vision_transformer)
    for epoch in range(epochs):
        train_metric = 0
        val_metric = 0

        val_accuracy = 0
        val_total = 0

        for train_step, (images, targets) in enumerate(train_loader, 1):
            vision_transformer.train()
            images = images.to(device)
            targets = targets.to(device)

            out = vision_transformer(images)
            optimizer.zero_grad()
            loss = loss_f(out, targets)

            torch.nn.utils.clip_grad_norm_(vision_transformer.parameters(), max_norm=1)

            train_metric += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        for val_step, (images, targets) in enumerate(test_loader, 1):
            vision_transformer.eval()
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                out = vision_transformer(images)
                loss = loss_f(out, targets)

            predict = out.argmax(dim=1)
            val_accuracy += (predict == targets).sum().item()
            val_total += targets.size(0)

            val_metric += loss.item()
        train_loss = train_metric / train_step
        val_loss = val_metric / val_step
        accuracy = val_accuracy / val_total
        print(
            f"[{epoch}]. Train loss: {train_loss:.4f}\
             Validation loss: {val_loss:.4f}\
             Validation accuracy: {accuracy}"
        )
    save_name = f"{args.model_name}_{args.learning_rate}_{accuracy}.pthar"
    save_model(save_name, vision_transformer, accuracy)
