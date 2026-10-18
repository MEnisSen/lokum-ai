from __future__ import annotations
from dataclasses import asdict, dataclass

def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple

@dataclass
class Config:
    vocab_size: int = 16384
    context_length: int = 1024
    n_embed: int = 768
    n_head: int = 12
    n_kv_head: int = 4
    n_blocks: int = 24
    rope_theta: float = 10_000.0
    # training
    batch_size: int = 4
    grad_accum: int = 128
    train_steps: int = 6000
    warmup_steps: int = 100
    eval_steps: int = 200
    eval_iters: int = 20
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    grad_checkpoint: bool = False
    train_path: str = "data/lokum/train.h5"
    dev_path: str = "data/lokum/val.h5"
    out_path: str = "models/base.pt"
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.n_embed % self.n_head != 0:
            raise ValueError("n_embed must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if (self.n_embed // self.n_head) % 2 != 0:
            raise ValueError("head size must be even for rotary embeddings")
        if not 0 <= self.min_lr <= self.lr:
            raise ValueError("need 0 <= min_lr <= lr")
        if self.warmup_steps >= self.train_steps:
            raise ValueError("warmup_steps must be smaller than train_steps")

    @property
    def head_dim(self) -> int:
        return self.n_embed // self.n_head

    @property
    def hidden_dim(self) -> int:
        return round_up(int(8 * self.n_embed / 3), 64)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def smoke(self) -> "Config":
        """A tiny net on the sample file, same vocabulary."""
        return Config(
            vocab_size=self.vocab_size,
            context_length=128,
            n_embed=128,
            n_head=4,
            n_kv_head=2,
            n_blocks=2,
            batch_size=8,
            grad_accum=4,
            train_steps=200,
            warmup_steps=10,
            eval_steps=50,
            eval_iters=5,
            train_path="data/lokum_smoke/train.h5",
            dev_path="data/lokum_smoke/val.h5",
            out_path="models/smoke.pt",
            seed=self.seed,
        )