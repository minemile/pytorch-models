import io
from functools import partial

import torch
import torch.nn as nn
from pytorch_models_imp.my_transformer import Transformer
# from pytorch_models_imp.transformer import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import build_vocab_from_iterator
from torch.optim.lr_scheduler import LambdaLR

BATCH_SIZE = 512
NUM_LAYERS = 3
HEADS = 8
EMBED_SIZE = 512
FORWARD_EXPANSION = 4
DROPOUT = 0.1
DEVICE = torch.device("cuda:1")
MAX_LENGTH = 100
# LEARNING_RATE = 3e-4
LEARNING_RATE = 3e-4
EPOCHS = 30
WARM_UP_STAGE = 4000


def yield_tokens(filepath, tokenizer):
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            yield tokenizer(string_)


def lr_lambda(step_num):
    step_num += 1
    step_num *= BATCH_SIZE
    mult = min(step_num**(-0.5), step_num * WARM_UP_STAGE**(-1.5))
    # print(LEARNING_RATE * mult)
    return mult


def train(model, loss_f, optimizer, train_loader, val_loader, warmup_steps=2000):
    # scheduler = LambdaLR(optimizer, lr_lambda)
    for epoch in range(EPOCHS):
        train_metric = 0
        val_metric = 0
        train_step = 0
        val_step = 0
        for de_batch, en_batch in train_loader:
            model.train()
            de_batch = de_batch.T.to(DEVICE)
            en_batch = en_batch.T.to(DEVICE)

            # print(en_batch.shape)

            out = model(de_batch, en_batch[:, :-1])

            out = out.reshape(-1, out.shape[2])
            target = en_batch[:, 1:].reshape(-1)

            optimizer.zero_grad()

            loss = loss_f(out, target)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            train_metric += loss.item()
            # print(f"Current loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            train_step += 1
            # scheduler.step()

        for de_batch, en_batch in val_loader:
            model.eval()
            de_batch = de_batch.T.to(DEVICE)
            en_batch = en_batch.T.to(DEVICE)
            with torch.no_grad():
                out = model(de_batch, en_batch[:, :-1])
                out = out.reshape(-1, out.shape[2])
                target = en_batch[:, 1:].reshape(-1)
                loss = loss_f(out, target)
            val_metric += loss.item()
            val_step += 1
            
        print(f"EPOCH {epoch}. Train loss: {train_metric / train_step}. Validation loss: {val_metric / val_step}")


def data_process(filepaths):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        de_tensor_ = torch.tensor(
            [de_vocab[token] for token in de_tokenizer(raw_de)], dtype=torch.long
        )
        en_tensor_ = torch.tensor(
            [en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long
        )
        data.append((de_tensor_, en_tensor_))
    return data


def generate_batch(data_batch, bos_indx, pad_indx, eos_indx):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(
            torch.cat(
                [torch.tensor([bos_indx]), de_item, torch.tensor([eos_indx])], dim=0
            )
        )
        en_batch.append(
            torch.cat(
                [torch.tensor([bos_indx]), en_item, torch.tensor([eos_indx])], dim=0
            )
        )
    de_batch = pad_sequence(de_batch, padding_value=pad_indx)
    en_batch = pad_sequence(en_batch, padding_value=pad_indx)
    return de_batch, en_batch


if __name__ == "__main__":
    url_base = (
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
    )
    train_urls = ("train.de.gz", "train.en.gz")
    val_urls = ("val.de.gz", "val.en.gz")
    test_urls = ("test_2016_flickr.de.gz", "test_2016_flickr.en.gz")

    train_filepaths = [
        extract_archive(download_from_url(url_base + url))[0] for url in train_urls
    ]
    val_filepaths = [
        extract_archive(download_from_url(url_base + url))[0] for url in val_urls
    ]
    test_filepaths = [
        extract_archive(download_from_url(url_base + url))[0] for url in test_urls
    ]

    de_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
    en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    de_vocab = build_vocab_from_iterator(
        yield_tokens(train_filepaths[0], de_tokenizer),
        specials=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    de_vocab.set_default_index(de_vocab["<unk>"])

    en_vocab = build_vocab_from_iterator(
        yield_tokens(train_filepaths[1], en_tokenizer),
        specials=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    en_vocab.set_default_index(de_vocab["<unk>"])

    train_data = data_process(train_filepaths)
    val_data = data_process(val_filepaths)
    test_data = data_process(test_filepaths)

    PAD_IDX = de_vocab["<pad>"]
    BOS_IDX = de_vocab["<bos>"]
    EOS_IDX = de_vocab["<eos>"]

    train_iter = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(
            generate_batch, bos_indx=BOS_IDX, pad_indx=PAD_IDX, eos_indx=EOS_IDX
        ),
    )
    valid_iter = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(
            generate_batch, bos_indx=BOS_IDX, pad_indx=PAD_IDX, eos_indx=EOS_IDX
        ),
    )
    test_iter = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(
            generate_batch, bos_indx=BOS_IDX, pad_indx=PAD_IDX, eos_indx=EOS_IDX
        ),
    )

    SRC_VOCAB_SIZE = len(de_vocab)
    TRG_VOCAB_SIZE = len(en_vocab)

    SRC_PAD_INDX = de_vocab["<pad>"]
    TRG_PAD_INDX = en_vocab["<pad>"]

    # train
    transformer = Transformer(
        SRC_VOCAB_SIZE,
        TRG_VOCAB_SIZE,
        SRC_PAD_INDX,
        TRG_PAD_INDX,
        EMBED_SIZE,
        NUM_LAYERS,
        FORWARD_EXPANSION,
        HEADS,
        DROPOUT,
        DEVICE,
        MAX_LENGTH,
    ).to(DEVICE)

    loss_f = nn.CrossEntropyLoss(ignore_index=TRG_PAD_INDX)
    optimizer = torch.optim.Adam(transformer.parameters(), LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

    train(transformer, loss_f, optimizer, train_iter, valid_iter)

    # save checkpoint
    checkpoint = {
        "state_dict": transformer.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, "checkpoint.pth")

