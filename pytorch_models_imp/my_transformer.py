import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads=8):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.embed_head_size = self.embed_size // self.heads

        self.queries = nn.Linear(self.embed_head_size, self.embed_head_size, bias=True)
        self.keys = nn.Linear(self.embed_head_size, self.embed_head_size, bias=True)
        self.values = nn.Linear(self.embed_head_size, self.embed_head_size, bias=True)

        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, query, key, value, mask=None):
        # query: (N, t, embed_size) -> (N, t, heads, dims)

        N = query.shape[0]

        query_len = query.shape[1]
        keys_len = key.shape[1]
        values_len = value.shape[1]

        query = query.view(-1, query_len, self.heads, self.embed_head_size)
        key = key.view(-1, keys_len, self.heads, self.embed_head_size)
        value = value.view(-1, values_len, self.heads, self.embed_head_size)

        query = self.queries(query)
        key = self.keys(key)
        value = self.values(value)

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        # enery: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        energy = energy / (self.embed_size ** (0.5))
        attention = torch.softmax(energy, dim=3)

        out = torch.einsum("nhqk,nkhd->nqhd", [attention, value])

        out = out.reshape(N, query_len, self.embed_head_size * self.heads)
        out = self.fc_out(out)
        return out  # (N, query_len, embed_size)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

        self.fc_layers = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * forward_expansion, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)  # N, query_len, embed_size

        # x = self.dropout(self.layer_norm1(attention + query)) # origin. kind of word
        # x = query + (self.layer_norm1(self.dropout(attention))) # doesnt work at all
        x = self.layer_norm1(query + self.dropout(attention))

        out = self.fc_layers(x)
        # out = self.dropout(self.layer_norm2(x + out))
        out = self.layer_norm2(x + self.dropout(out))
        # out = x + self.layer_norm2(self.dropout(out)) # doesnt work at all
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        n_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device='cpu'
    ):
        super().__init__()

        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.src_position_embedding = nn.Embedding(max_length, embed_size)

        self.transformer_blocks = nn.ModuleList()
        for i in range(n_layers):
            self.transformer_blocks.add_module(
                f"trans_{i}",
                TransformerBlock(embed_size, heads, forward_expansion, dropout),
            )

        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.embed_size = embed_size

    def forward(self, x, mask):
        n, t = x.shape

        word_embeddings = self.src_word_embedding(x) * self.embed_size**0.5  # n, t, e

        positions = torch.arange(0, t).unsqueeze(0).expand(n, t).to(self.device)
        position_embeddings = self.src_position_embedding(positions) * self.embed_size**0.5  # n, t, e

        out = self.dropout(word_embeddings + position_embeddings)

        for mod in self.transformer_blocks:
            out = mod(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super().__init__()
        self.decoder_attention = SelfAttention(embed_size, heads)
        self.layer_norm = nn.LayerNorm(embed_size)

        self.transformer_block = TransformerBlock(
            embed_size, heads, forward_expansion, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, src_mask, trg_mask):
        decoder_attention = self.decoder_attention(query, query, query, trg_mask)
        x = self.layer_norm(query + self.dropout(decoder_attention))
        # x = self.dropout(self.layer_norm(decoder_attention +  query)) # origin doesnt word
        # x = query + (self.layer_norm(self.dropout(decoder_attention))) # doesnt work at all

        out = self.transformer_block(x, key, value, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        n_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device='cpu'
    ):
        super().__init__()
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.trg_position_embedding = nn.Embedding(max_length, embed_size)

        self.transformer_blocks = nn.ModuleList()
        for i in range(n_layers):
            self.transformer_blocks.add_module(
                f"trans_{i}",
                DecoderBlock(embed_size, heads, forward_expansion, dropout),
            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_size, trg_vocab_size)
        self.device = device
        self.embed_size = embed_size

    def forward(self, x, encoder_kv, scr_mask, trg_mask):
        n, t = x.shape

        word_embeddings = self.trg_word_embedding(x) * self.embed_size**0.5  # n, t, e

        positions = torch.arange(0, t).unsqueeze(0).expand(n, t).to(self.device)
        position_embeddings = self.trg_position_embedding(positions) * self.embed_size**0.5  # n, t, e

        out = self.dropout(word_embeddings + position_embeddings)

        for mod in self.transformer_blocks:
            out = mod(out, encoder_kv, encoder_kv, scr_mask, trg_mask)
        out = self.fc(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        # self.__init_params()

    def forward(self, src, trgt):
        src_mask = self.make_src_mask(src) # N, t, t
        trg_mask = self.make_trg_mask(trgt) # N, t, t

        encoder_kv = self.encoder(src, src_mask)
        out = self.decoder(trgt, encoder_kv, src_mask, trg_mask)
        return out


    def make_src_mask(self, x):
        mask = (x != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return mask.to(self.device)

    def make_trg_mask(self, x):
        N, seq_len = x.shape
        padded_mask = (x != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).expand(N, 1, seq_len, seq_len).to(self.device)
        return padded_mask & mask

    # def __init_params(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
