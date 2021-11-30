import numpy as np
import torch
import torch.nn as nn


def partition_window(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return x  # nwindows, windows_size, windows_size, C


def reverse_partition(x, window_size, h, w):
    num_windows, windows_size, windows_size, C = x.shape
    batch_size = int(num_windows // (h // window_size) / (w // window_size))
    x = x.view(
        batch_size, h // window_size, w // window_size, window_size, window_size, C
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, h, w, C)
    return x  # N, H, W, C


class MPL(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, in_dim),
        )

    def forward(self, x):
        return self.ff(x)


class PatchEmbeddings(nn.Module):
    def __init__(self, image_size, patch_size=4, embed_size=96):
        super().__init__()
        self.num_of_patches = (image_size // patch_size) ** 2
        self.patch_resolution = image_size // patch_size
        assert self.num_of_patches * patch_size ** 2 == image_size ** 2

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=embed_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        b, c, h, w = x.shape
        # extract and embed patches
        x = self.conv(x).reshape(b, self.num_of_patches, -1)

        # normalize
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, patch_resolution, dim):
        super().__init__()
        self.patch_resolution = patch_resolution
        self.dim = dim

        self.norm = nn.LayerNorm(4 * self.dim)
        self.project = nn.Linear(4 * self.dim, 2 * self.dim, bias=False)

    def forward(self, x):
        H, W = self.patch_resolution
        assert H % 2 == 0 and W % 2 == 0
        N = x.size(0)
        x = x.reshape(N, H, W, self.dim)

        top_left = x[:, 0::2, 0::2, :]
        top_right = x[:, 0::2, 1::2, :]

        bottom_left = x[:, 1::2, 0::2, :]
        bottom_right = x[:, 1::2, 1::2, :]

        features = torch.cat((top_left, top_right, bottom_left, bottom_right), dim=3)
        features = features.view(N, H // 2 * W // 2, self.dim * 4)
        features = self.norm(features)
        features = self.project(features)
        return features


class WindowSelfAttention(nn.Module):
    def __init__(self, heads, window_size, embed_size, dropout):
        super().__init__()
        self.embed_size = embed_size

        self.heads = heads
        self.heads_dim = embed_size // heads
        self.window_size = window_size

        assert embed_size % self.heads == 0, "Embed dim isn't divisible by heads"

        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc = nn.Linear(embed_size, embed_size)

        self.relative_position = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1), (2 * window_size[1] - 1), heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        relative_position_index = self.get_relative_index(window_size)
        self.register_buffer("relative_position_index", relative_position_index)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B_, N, C = query.shape
        query = self.queries(query)  # N, query_len, d
        key = self.keys(key)
        value = self.values(value)

        query = query.reshape(-1, query.shape[1], self.heads, self.heads_dim)
        key = key.reshape(-1, key.shape[1], self.heads, self.heads_dim)
        value = value.reshape(-1, value.shape[1], self.heads, self.heads_dim)

        raw_attention = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        raw_attention /= self.heads_dim ** (0.5)
        # N, heads, query_len, keys_len

        # return self.relative_position_index
        relative_position_bias = (
            self.relative_position[
                self.relative_position_index[:, :, 0],
                self.relative_position_index[:, :, 1],
            ]
            .permute(2, 0, 1)
            .contiguous()
        )

        raw_attention += relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            raw_attention = raw_attention.view(
                B_ // nW, nW, self.heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            raw_attention = raw_attention.view(-1, self.heads, N, N)

        attention = torch.softmax(raw_attention, dim=3)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, value])  # nqhd

        out = out.reshape(-1, query.shape[1], self.embed_size)
        out = self.fc(out)
        out = self.dropout(out)
        return out

    def get_relative_index(self, window_size):
        indices = torch.tensor(
            np.array(
                [[x, y] for x in range(window_size[0]) for y in range(window_size[1])]
            )
        )
        relative_position_index = indices[None, :, :] - indices[:, None, :]
        return relative_position_index


class TransformerBlock(nn.Module):
    def __init__(
        self,
        heads,
        shift_size,
        embed_size,
        window_size,
        patches_resolution,
        forward_expansion,
        dropout,
    ):
        super().__init__()
        self.window_size = window_size
        self.patches_resolution = patches_resolution
        self.shift_size = shift_size

        self.self_attention = WindowSelfAttention(
            heads, (window_size, window_size), embed_size, dropout
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.fc = MPL(embed_size, embed_size * forward_expansion, dropout)
        self.dropout = nn.Dropout(dropout)

        if self.shift_size > 0:
            attn_mask = self.get_attn_mask(window_size, shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def get_attn_mask(self, window_size, shift_size):
        # calculate attention mask for SW-MSA
        H, W = self.patches_resolution
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = partition_window(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        return attn_mask

    def forward(self, x):
        B, T, C = x.shape
        H, W = self.patches_resolution
        assert T == H * W
        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition window
        x = partition_window(x, window_size=self.window_size)
        x = x.view(
            -1, self.window_size ** 2, C
        )  # nWindows, window_size * window_size, C

        # window attention
        attention_windows = self.self_attention(x, x, x, mask=self.attn_mask)
        attention_windows = attention_windows.view(
            -1, self.window_size, self.window_size, C
        )
        x = reverse_partition(attention_windows, self.window_size, H, W)  # B, H, W, C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = attention_windows.view(B, -1, C)

        x = x + shortcut
        x = x + self.fc(self.norm2(x))
        return x


class SwinBlock(nn.Module):
    def __init__(
        self,
        patches_resolution,
        heads,
        depth,
        embed_size,
        expansion,
        window_size,
        dropout,
        downsample=None,
    ):
        super().__init__()

        self.swin_modules = nn.ModuleList()

        for i in range(depth):
            self.swin_modules.add_module(
                f"trans_block_{i}",
                TransformerBlock(
                    heads,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    embed_size=embed_size,
                    window_size=window_size,
                    patches_resolution=patches_resolution,
                    forward_expansion=expansion,
                    dropout=dropout,
                ),
            )

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = self.downsample(patches_resolution, embed_size)

    def forward(self, x):
        for module in self.swin_modules:
            x = module(x)  # N, H * W, C

        if self.downsample is not None:
            x = self.downsample(x)  # N, H // 2 * W // 2, C*2

        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        num_classes,
        depths=[2, 2, 6, 2],
        patch_size=4,
        num_heads=[3, 6, 12, 24],
        embed_size=96,
        forward_expansion=4,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbeddings(image_size, patch_size, embed_size)
        self.pos_dropout = nn.Dropout(dropout)

        self.num_of_patches = self.patch_embedding.num_of_patches
        self.patch_resolution = self.patch_embedding.patch_resolution

        self.num_layers = len(depths)
        self.num_features = embed_size * 2 ** (self.num_layers - 1)

        self.transformer_blocks = nn.ModuleList()
        block_patch_resolution = self.patch_resolution
        for i in range(self.num_layers):
            self.transformer_blocks.add_module(
                f"swin_block_{i}",
                SwinBlock(
                    (block_patch_resolution, block_patch_resolution),
                    num_heads[i],
                    depths[i],
                    embed_size,
                    forward_expansion,
                    window_size=7,
                    dropout=dropout,
                    downsample=PatchMerging if i < (self.num_layers - 1) else None,
                ),
            )
            block_patch_resolution = block_patch_resolution // 2
            embed_size = embed_size * 2

        self.final_norm = nn.LayerNorm(self.num_features)

        self.classifier = nn.Sequential(nn.Linear(self.num_features, num_classes))

    def forward(self, x):
        out = self.patch_embedding(x)  # N, num_of_patches, embed_size
        out = self.pos_dropout(out)
        for mod in self.transformer_blocks:
            out = mod(out)

        out = self.final_norm(out)

        out = out.mean(axis=1)  # N, features
        out = self.classifier(out)
        return out


def swin_t(input_image=224, num_classes=1000):
    return SwinTransformer(
        input_image,
        num_classes,
        depths=[2, 2, 6, 2],
        patch_size=4,
        num_heads=[3, 6, 12, 24],
        embed_size=96,
        forward_expansion=4,
        dropout=0.0,
    )


def swin_s(input_image=224, num_classes=1000):
    return SwinTransformer(
        input_image,
        num_classes,
        depths=[2, 2, 18, 2],
        patch_size=4,
        num_heads=[3, 6, 12, 24],
        embed_size=96,
        forward_expansion=4,
        dropout=0.0,
    )


def swin_b(input_image=224, num_classes=1000):
    return SwinTransformer(
        input_image,
        num_classes,
        depths=[2, 2, 18, 2],
        patch_size=4,
        num_heads=[4, 8, 16, 32],
        embed_size=128,
        forward_expansion=4,
        dropout=0.0,
    )


def swin_l(input_image=224, num_classes=1000):
    return SwinTransformer(
        input_image,
        num_classes,
        depths=[2, 2, 18, 2],
        patch_size=4,
        num_heads=[6, 12, 24, 48],
        embed_size=192,
        forward_expansion=4,
        dropout=0.0,
    )
