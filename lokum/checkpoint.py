from __future__ import annotations

from pathlib import Path

import torch

from lokum.config import Config
from lokum.model import Transformer


def save_checkpoint(path: str, model: Transformer, optimizer, step: int, cfg: Config) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "step": step,
            "config": cfg.to_dict(),
        },
        path,
    )
    print(f"saved {path} at step {step}")


def load_model(path: str, device: str) -> tuple[Transformer, Config, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt