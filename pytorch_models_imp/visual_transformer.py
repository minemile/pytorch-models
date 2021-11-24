import torch
from torch._C import set_anomaly_enabled
import torch.nn as nn


class PatchEmbeddings(nn.Module):
    def __init__(self, image_size, patch_size=16, embed_size=768, dropout=0.0):
        super().__init__()
        self.num_of_patches = (image_size // patch_size)**2
        assert self.num_of_patches * patch_size**2 ==  image_size ** 2

        self.conv = nn.Conv2d(in_channels=3, out_channels=embed_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_of_patches + 1, embed_size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        # extract and embed patches
        x = self.conv(x).reshape(b, self.num_of_patches, -1)

        # add [class] token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add position embeddings
        position_embeddings = self.position_embeddings.expand(b, -1, -1)
        x = self.dropout(x + position_embeddings)
        return x

class SelfAttention(nn.Module):
    def __init__(self, heads, embed_size, dropout):
        super().__init__()
        self.embed_size = embed_size

        self.heads = heads
        self.heads_dim = embed_size // heads
        assert self.heads_dim * heads == embed_size, "Embed dim isn't divisible by heads"

        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        query_len = query.shape[1]
        key_len = key.shape[1]
        value_len = value.shape[1]

        query = self.queries(query) # N, query_len, d
        key = self.keys(key)
        value = self.values(value)

        query = query.reshape(-1, query_len, self.heads, self.heads_dim)
        key = key.reshape(-1, key_len, self.heads, self.heads_dim)
        value = value.reshape(-1, value_len, self.heads, self.heads_dim)



        raw_attention = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        raw_attention /= self.heads_dim**(0.5)
        # N, heads, query_len, keys_len

        attention = torch.softmax(raw_attention, dim=3)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, value]) # nqhd

        out = self.dropout(out)
        out = out.reshape(-1, query_len, self.embed_size)
        out = self.fc(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, heads, embed_size, forward_expansion, dropout):
        super().__init__()
        self.self_attention = SelfAttention(heads, embed_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * forward_expansion, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attention(normed, normed, normed))
        x = x + self.dropout(self.fc(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, num_classes, patch_size=16, heads=12, embed_size=768, forward_expansion=4, num_layers=12, dropout=0.0):
        super().__init__()
        self.patch_embedding = PatchEmbeddings(image_size, patch_size, embed_size, dropout)
        self.transformer_blocks = nn.ModuleList()

        for i in range(num_layers):
            self.transformer_blocks.add_module(f"module_{i}", TransformerBlock(heads, embed_size, forward_expansion, dropout))

        self.classifier = nn.Sequential(
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, x):
        out = self.patch_embedding(x) # N, num_of_patches + 1, embed_size
        for mod in self.transformer_blocks:
            out = mod(out)
        out = out[:, 0, :]
        out = self.classifier(out)
        return out


