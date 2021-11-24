import timm
import torch
import torch.nn as nn
import torchvision
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchinfo import summary

from pytorch_models_imp.visual_transformer import VisionTransformer

BATCH_SIZE = 64
NUM_WORKERS = 5
LEARNING_RATE = 1e-4
EPOCHS = 20
IMAGE_SIZE = 224
DEVICE = "cuda:0"

# transformer
FORWARD_EXPANSION = 4
NUM_LAYERS = 6
HEADS = 3
PATCH_SIZE = 16
EMBEDDING_SIZE = 192
DROPOUT = 0.1

USE_TIMM = True
TIMM_MODEL = "vit_base_patch32_224_in21k"

config = resolve_data_config({}, model=TIMM_MODEL)
transform = create_transform(**config)



if __name__ == "__main__":
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    num_classes = len(classes)

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    if USE_TIMM:
        vision_transformer = timm.create_model(
            TIMM_MODEL, pretrained=True, num_classes=num_classes
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
    optimizer = torch.optim.Adam(
        vision_transformer.parameters(), LEARNING_RATE, weight_decay=1e-5
    )
    # optimizer = torch.optim.SGD(vision_transformer.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(trainloader), epochs=EPOCHS)

    print(f"There is {EPOCHS * len(trainloader)} steps. For warmup there is {EPOCHS * len(trainloader) * 0.3} steps")

    vision_transformer.to(DEVICE)
    for epoch in range(EPOCHS):
        train_metric = 0
        val_metric = 0
        train_step = 0
        val_step = 0
        for images, targets in trainloader:
            vision_transformer.train()
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            out = vision_transformer(images)
            optimizer.zero_grad()
            loss = loss_f(out, targets)

            torch.nn.utils.clip_grad_norm_(vision_transformer.parameters(), max_norm=1)

            train_metric += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_step += 1

        for images, targets in testloader:
            vision_transformer.eval()
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            with torch.no_grad():
                out = vision_transformer(images)
                loss = loss_f(out, targets)
            val_metric += loss.item()
            val_step += 1
        print(
            f"EPOCH {epoch}. Train loss: {train_metric / train_step}. Validation loss: {val_metric / val_step}"
        )

