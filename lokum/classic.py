from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.hidden = nn.Linear(n_embed, 4 * n_embed)
        self.proj = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.hidden(x)))


class Head(nn.Module):
    def __init__(self, n_embed: int, head_size: int, context: int):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context, context)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        scores = scores.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        return F.softmax(scores, dim=-1) @ v


class MultiHead(nn.Module):
    def __init__(self, n_embed: int, n_head: int, context: int):
        super().__init__()
        size = n_embed // n_head
        self.heads = nn.ModuleList(Head(n_embed, size, context) for _ in range(n_head))
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class Block(nn.Module):
    def __init__(self, n_embed: int, n_head: int, context: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHead(n_embed, n_head, context)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ClassicTransformer(nn.Module):
    def __init__(self, vocab: int, n_embed: int, n_head: int, n_blocks: int, context: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embed)
        self.pos = nn.Embedding(context, n_embed)
        self.blocks = nn.ModuleList(Block(n_embed, n_head, context) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab)
        self.context = context

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


if __name__ == "__main__":
    model = ClassicTransformer(vocab=100, n_embed=32, n_head=4, n_blocks=2, context=16)
    logits = model(torch.randint(0, 100, (2, 16)))
    print("classic logits", tuple(logits.shape))