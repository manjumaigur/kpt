import torch
import torch.nn as nn
from torch.nn import functional as F


def initialize_weight_bias(layer: nn.Module, mean: float, std: float) -> None:
    layer.weight.data.normal_(mean, std)
    if hasattr(layer, "bias"):
        layer.bias.data.zero_()


class ReLUSquared(nn.Module):
    # https://arxiv.org/pdf/2109.08668
    def forward(self, x):
        F.relu_(x)
        return torch.square(x)


class FeedForward(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        embed_size = config["embed_size"]
        self.linear1 = nn.Linear(embed_size, embed_size * 4)
        self.squared_relu = ReLUSquared()
        self.linear2 = nn.Linear(embed_size * 4, embed_size)

        # initialization
        std = 0.02 * (2 * config["n_layer"]) ** -0.5
        initialize_weight_bias(self.linear1, 0.0, std)
        initialize_weight_bias(self.linear2, 0.0, std)

    def forward(self, x):
        x = self.linear1(x)
        # squared ReLU
        x = self.squared_relu(x)
        x = self.linear2(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embed_size = config["embed_size"]
        self.n_head = config["n_head"]

        # key, query, value projections
        self.attention_layer = nn.Linear(self.embed_size, 3 * self.embed_size)
        # output projection
        self.out_proj = nn.Linear(self.embed_size, self.embed_size)

        # initialization
        std = 0.02 * (2 * config["n_layer"]) ** -0.5
        initialize_weight_bias(self.attention_layer, 0.0, std)
        initialize_weight_bias(self.out_proj, 0.0, std)

    def forward(self, x):
        batch_size, context_length, embed_size = x.shape

        qkv = self.attention_layer(x)
        q, k, v = qkv.split(self.embed_size, dim=2)

        head_size = embed_size // self.n_head
        # B, nheads, context_length, head_size
        k = k.view(batch_size, context_length, self.n_head, head_size).transpose(1, 2)
        q = q.view(batch_size, context_length, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch_size, context_length, self.n_head, head_size).transpose(1, 2)

        # flash attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, context_length, embed_size)
        )
        return self.out_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config["embed_size"])
        self.attention_block = CausalSelfAttention(config)
        self.layer_norm2 = nn.LayerNorm(config["embed_size"])
        self.feed_forward_block = FeedForward(config)

    def forward(self, x):
        # pre-norm
        x = x + self.attention_block(self.layer_norm1(x))
        x = x + self.feed_forward_block(self.layer_norm2(x))
        return x


class KPT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        _validate_config(config)
        self.config = config
        embed_size = config["embed_size"]
        vocab_size = config["vocab_size"]
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(config["context_length"], embed_size)
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config["n_layer"])
        )
        self.layer_norm = nn.LayerNorm(embed_size)

        self.output_head = nn.Linear(embed_size, vocab_size, bias=False)

        # weight sharing
        self.token_embeddings.weight = self.output_head.weight

        # initialization
        std = 0.02
        initialize_weight_bias(self.token_embeddings, 0.0, std)
        initialize_weight_bias(self.position_embeddings, 0.0, std)
        initialize_weight_bias(self.output_head, 0.0, std)

    def forward(self, x):
        batch_size, context_length = x.size()
        assert context_length <= self.config["context_length"]

        # position embeddings
        pos = torch.arange(0, context_length, dtype=torch.long, device=x.device)

        # context_length, embed_size
        pos_emb = self.position_embeddings(pos)

        # batch_size, context_length, embed_size
        tok_emb = self.token_embeddings(x)

        x = tok_emb + pos_emb
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)

        x = self.layer_norm(x)

        # batch_size, context_length, vocab_size
        logits = self.output_head(x)

        return logits


def _validate_config(self, config: dict) -> bool:
    # very naive way of validating
    required_keys = [
        "embed_size",
        "context_length",
        "n_layer",
        "n_head",
        "vocab_size",
    ]
    for _key in required_keys:
        if _key not in config:
            raise KeyError(f"key {_key} not found in config")

        assert config[_key] > 0, f"{_key} should be positive"

        # TODO: Add more validations

    return True
