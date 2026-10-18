from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lokum.config import Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


def _rope_cache(head_dim: int, context: int, theta: float, device: torch.device, dtype: torch.dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(context, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x is (B, heads, T, head_dim). cos/sin are (T, head_dim).
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.q = nn.Linear(cfg.n_embed, cfg.n_head * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.n_embed, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.n_embed, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_head * cfg.head_dim, cfg.n_embed, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim)
        self.k_norm = RMSNorm(cfg.head_dim)
        self.rope_theta = cfg.rope_theta
        self.context_length = cfg.context_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = _rope_cache(self.head_dim, t, self.rope_theta, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        repeat = self.n_head // self.n_kv
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = cfg.hidden_dim
        self.gate = nn.Linear(cfg.n_embed, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embed, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embed, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embed)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embed)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.context_length = cfg.context_length
        self.n_embed = cfg.n_embed
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_blocks))
        self.norm = RMSNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.grad_checkpoint = cfg.grad_checkpoint
        self.apply(self._init)
        # tying shares one matrix; init it once, after apply() would have touched both names
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.lm_head(self.hidden(idx))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())