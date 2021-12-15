from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchOverlapEmbeddings(nn.Module):
    def __init__(self, input_channels, image_sizes, stride, patch_size, embed_size):
        super().__init__()
        assert isinstance(
            image_sizes, tuple
        ), f"Image size is not a tuple. Got {type(image_sizes)}"
        self.patch_height_resolution = image_sizes[0] // patch_size
        self.patch_width_resoultion = image_sizes[1] // patch_size
        self.number_of_patches = (
            self.patch_height_resolution * self.patch_width_resoultion
        )
        self.image_sizes = image_sizes
        self.embed_size = embed_size
        # assert self.number_of_patches * patch_size ** 2 == image_sizes[0] * image_sizes[1]

        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_size,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size // 2, patch_size // 2),
        )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        # extract and embed patches
        x = self.conv(x)
        b, c, h_new, w_new = x.shape
        x = x.reshape(b, self.embed_size, -1).permute(0, 2, 1)
        # normalize
        x = self.norm(x)
        return x, h_new, w_new


class ReducedSelfAttention(nn.Module):
    def __init__(self, num_heads, embed_size, reduction, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.reduction = reduction
        assert (
            self.embed_size % self.num_heads == 0
        ), "Embed size is not divisible by num heads"
        self.head_embed_size = self.embed_size // self.num_heads

        self.query = nn.Linear(self.embed_size, self.embed_size)
        self.key = nn.Linear(self.embed_size, self.embed_size)
        self.value = nn.Linear(self.embed_size, self.embed_size)

        self.dropout = nn.Dropout(dropout)

        self.reduction = reduction
        if reduction > 1:
            self.reductor = nn.Conv2d(
                in_channels=embed_size,
                out_channels=embed_size,
                kernel_size=reduction,
                stride=reduction,
            )
            self.ln = nn.LayerNorm(embed_size)

        self.proj = nn.Linear(in_features=embed_size, out_features=embed_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_embed_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, h, w):
        batch_size, L, C = x.shape
        query = self.transpose_for_scores(self.query(x))  # nhqd
        kv = x
        if self.reduction > 1:
            kv = x.permute(0, 2, 1).reshape(batch_size, C, h, w)
            kv = self.reductor(kv)
            kv = kv.reshape(batch_size, C, -1).permute(0, 2, 1)
            kv = self.ln(kv)
        key = self.transpose_for_scores(self.key(kv))  # nhkd
        value = self.transpose_for_scores(self.value(kv))  # nhvd

        raw_attention = query @ key.transpose(2, 3)  # nhqk
        raw_attention = raw_attention / self.head_embed_size ** 0.5

        attention = nn.functional.softmax(raw_attention, dim=-1)
        attention = self.dropout(attention)

        attention = attention @ value

        attention = attention.permute(0, 2, 1, 3).contiguous()
        attention = attention.view(batch_size, L, C)
        attention = self.proj(attention)
        attention = self.dropout(attention)

        return attention


class PositionDWConv(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )

    def forward(self, x, h, w):
        batch, L, C = x.shape
        x = x.permute(0, 2, 1).view(batch, C, h, w)
        x = self.conv(x)
        return x.view(batch, C, L).permute(0, 2, 1)


class MixFFN(nn.Module):
    def __init__(self, embed_size, mlp_expansion, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, embed_size * mlp_expansion)
        self.fc2 = nn.Linear(embed_size * mlp_expansion, embed_size)
        self.position_conv = PositionDWConv(embed_size * mlp_expansion)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.activation(self.position_conv(x, h, w))
        x = self.fc2(self.dropout(x))
        x = self.dropout(x)
        return x


class SegformerBlock(nn.Module):
    def __init__(self, num_heads, embed_size, reduction, mlp_expansion, dropout):
        super().__init__()
        self.attention = ReducedSelfAttention(
            num_heads=num_heads,
            embed_size=embed_size,
            reduction=reduction,
            dropout=dropout,
        )
        self.mix_ffn = MixFFN(embed_size, mlp_expansion, dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x, h, w):
        x = x + self.attention(self.ln1(x), h, w)
        x = x + self.mix_ffn(self.ln2(x), h, w)
        return x


class SegformerLayer(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        embed_size,
        reduction,
        mlp_expansion,
        dropout,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = SegformerBlock(
                num_heads=num_heads,
                embed_size=embed_size,
                reduction=reduction,
                mlp_expansion=mlp_expansion,
                dropout=dropout,
            )
            self.layers.add_module(f"layer_{i}", layer)

    def forward(self, x, h, w):
        for layer in self.layers:
            x = layer(x, h, w)
        return x


PatchConfig = namedtuple(
    "Patch", ["input_channels", "embed_size", "patch_size", "stride", "padding"]
)


class SegformerLayersConfig:
    def __init__(
        self,
        patch_merge_config: PatchConfig,
        num_layers,
        num_heads,
        reduction,
        expansion,
        dropout,
    ) -> None:
        self.patch_merge_config = patch_merge_config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.reduction = reduction
        self.mlp_expansion = expansion
        self.dropout = dropout


class SegformerEncoder(nn.Module):
    def __init__(
        self,
        image_sizes,
        layer_configurations,
    ):
        super().__init__()
        assert isinstance(
            image_sizes, tuple
        ), f"Image size is not a tuple. Got {type(image_sizes)}"

        self.layer_configuration = layer_configurations
        self.image_sizes = image_sizes
        self.embed_sizes = []

        self.segformer_layers = nn.ModuleList()
        self.patch_merges = nn.ModuleList()
        self.stage_norms = nn.ModuleList()

        for indx, layer_configuration in enumerate(layer_configurations, 1):
            patch_config = layer_configuration.patch_merge_config
            H_reducted, W_reducted = (
                self.image_sizes[0] // 2 ** indx,
                self.image_sizes[1] // 2 ** indx,
            )
            embed_size = patch_config.embed_size
            patch_merger = PatchOverlapEmbeddings(
                input_channels=patch_config.input_channels,
                image_sizes=(H_reducted, W_reducted),
                stride=patch_config.stride,
                patch_size=patch_config.patch_size,
                embed_size=embed_size,
            )
            self.patch_merges.append(patch_merger)

            segformer_layer = SegformerLayer(
                num_layers=layer_configuration.num_layers,
                num_heads=layer_configuration.num_heads,
                embed_size=embed_size,
                reduction=layer_configuration.reduction,
                mlp_expansion=layer_configuration.mlp_expansion,
                dropout=layer_configuration.dropout,
            )
            self.segformer_layers.append(segformer_layer)
            self.stage_norms.append(nn.LayerNorm(embed_size))
            self.embed_sizes.append(embed_size)

    def forward(self, x):
        hidden_states = []
        b, c, h, w = x.shape
        for layer_num in range(len(self.segformer_layers)):
            patch_merger = self.patch_merges[layer_num]
            x, cur_h, cur_w = patch_merger(x)

            x = self.segformer_layers[layer_num](x, cur_h, cur_w)  # N, L, C
            x = self.stage_norms[layer_num](x)

            x = x.permute(0, 2, 1).reshape(b, -1, cur_h, cur_w)
            hidden_states.append(x)
        return hidden_states


class MLPDecoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ff = nn.Linear(in_features, out_features)

    def forward(self, x):
        batch, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.ff(x)
        x = (
            x.permute(0, 2, 1).contiguous().view(batch, -1, h, w)
        )  # batch, out_features, h, w
        return x


class SegformerDecoder(nn.Module):
    def __init__(self, embed_sizes, decoder_hidden_state, num_labels, dropout):
        super().__init__()
        self.mlps = nn.ModuleList()
        for layer_embed_dim in embed_sizes:
            self.mlps.append(MLPDecoder(layer_embed_dim, decoder_hidden_state))

        self.linear_fuse = nn.Conv2d(
            in_channels=decoder_hidden_state * len(embed_sizes),
            out_channels=decoder_hidden_state,
            kernel_size=1,
            bias=True,
        )
        self.batch_norm = nn.BatchNorm2d(decoder_hidden_state)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(decoder_hidden_state, num_labels, kernel_size=1)

    def forward(self, hidden_states):
        batch_size, c, h_4, w_4 = hidden_states[0].shape

        unified_features = []
        for indx, hidden_state in enumerate(hidden_states):
            unified_state = self.mlps[indx](hidden_state)
            unified_state = F.interpolate(
                unified_state, size=(h_4, w_4), mode="bilinear", align_corners=False
            )
            unified_features.append(unified_state)

        fused_features = torch.cat(unified_features, dim=1)
        fused_features = self.batch_norm(self.linear_fuse(fused_features))
        fused_features = self.dropout(self.activation(fused_features))
        fused_features = self.classifier(fused_features)
        return fused_features


class Segformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        hidden_states = self.encoder(x)
        output = self.decoder(hidden_states)
        return output


def segformer_b0(image_sizes, num_labels):
    encoder = mitb0(image_sizes)
    decoder = SegformerDecoder(encoder.embed_sizes, 256, num_labels, 0.0)
    return Segformer(encoder, decoder)


def segformer_b1(image_sizes, num_labels):
    encoder = mitb1(image_sizes)
    decoder = SegformerDecoder(encoder.embed_sizes, 256, num_labels, 0.0)
    return Segformer(encoder, decoder)


def segformer_b2(image_sizes, num_labels):
    encoder = mitb2(image_sizes)
    decoder = SegformerDecoder(encoder.embed_sizes, 768, num_labels, 0.0)
    return Segformer(encoder, decoder)


def mitb0(image_sizes):
    layer_configurations = [
        SegformerLayersConfig(PatchConfig(3, 32, 7, 4, 3), 2, 1, 8, 8, 0.0),
        SegformerLayersConfig(PatchConfig(32, 64, 3, 2, 1), 2, 2, 4, 8, 0.0),
        SegformerLayersConfig(PatchConfig(64, 160, 3, 2, 1), 2, 5, 2, 4, 0.0),
        SegformerLayersConfig(PatchConfig(160, 256, 3, 2, 1), 2, 8, 1, 4, 0.0),
    ]
    return SegformerEncoder(image_sizes, layer_configurations)


def mitb1(image_sizes):
    layer_configurations = [
        SegformerLayersConfig(
            PatchConfig(3, 64, 7, 4, 3),
            num_layers=2,
            num_heads=1,
            reduction=8,
            expansion=8,
            dropout=0.0,
        ),
        SegformerLayersConfig(
            PatchConfig(64, 128, 3, 2, 1),
            num_layers=2,
            num_heads=2,
            reduction=4,
            expansion=8,
            dropout=0.0,
        ),
        SegformerLayersConfig(
            PatchConfig(128, 320, 3, 2, 1),
            num_layers=2,
            num_heads=5,
            reduction=2,
            expansion=4,
            dropout=0.0,
        ),
        SegformerLayersConfig(
            PatchConfig(320, 512, 3, 2, 1),
            num_layers=2,
            num_heads=8,
            reduction=1,
            expansion=4,
            dropout=0.0,
        ),
    ]
    return SegformerEncoder(image_sizes, layer_configurations)


def mitb2(image_sizes):
    layer_configurations = [
        SegformerLayersConfig(
            PatchConfig(3, 64, 7, 4, 3),
            num_layers=3,
            num_heads=1,
            reduction=8,
            expansion=8,
            dropout=0.0,
        ),
        SegformerLayersConfig(
            PatchConfig(64, 128, 3, 2, 1),
            num_layers=3,
            num_heads=2,
            reduction=4,
            expansion=8,
            dropout=0.0,
        ),
        SegformerLayersConfig(
            PatchConfig(128, 320, 3, 2, 1),
            num_layers=6,
            num_heads=5,
            reduction=2,
            expansion=4,
            dropout=0.0,
        ),
        SegformerLayersConfig(
            PatchConfig(320, 512, 3, 2, 1),
            num_layers=3,
            num_heads=8,
            reduction=1,
            expansion=4,
            dropout=0.0,
        ),
    ]
    return SegformerEncoder(image_sizes, layer_configurations)
