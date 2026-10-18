from __future__ import annotations

import argparse
import math
import time

import torch

from lokum.checkpoint import save_checkpoint
from lokum.config import Config
from lokum.data import batch_iterator
from lokum.model import Transformer, count_parameters


def lr_at(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.train_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


def evaluate(model, cfg: Config, device: str) -> float:
    model.eval()
    total = 0.0
    iterator = batch_iterator(cfg.dev_path, cfg.batch_size, cfg.context_length, device)
    with torch.no_grad():
        for _ in range(cfg.eval_iters):
            x, y = next(iterator)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            total += float(loss)
    model.train()
    return total / cfg.eval_iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = Config().smoke() if args.smoke else Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    model = Transformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        start = int(ckpt["step"]) + 1
        print(f"resumed at step {start}")
    print(f"parameters {count_parameters(model):,}  device {device}  context {cfg.context_length}")
    print(f"tokens/step {cfg.batch_size * cfg.context_length * cfg.grad_accum:,}")
    iterator = batch_iterator(cfg.train_path, cfg.batch_size, cfg.context_length, device)
    model.train()
    started = time.perf_counter()
    for step in range(start, cfg.train_steps):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(cfg.grad_accum):
            x, y = next(iterator)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            loss.backward()
            running += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if step % 20 == 0:
            elapsed = time.perf_counter() - started
            rate = (20 * cfg.batch_size * cfg.context_length * cfg.grad_accum / elapsed) if step else 0
            started = time.perf_counter()
            print(f"step {step}  loss {running:.4f}  lr {lr:.2e}  {rate:,.0f} tok/s")
        if step > start and step % cfg.eval_steps == 0:
            print(f"  dev {evaluate(model, cfg, device):.4f}")
        if step > start and step % 1000 == 0:
            save_checkpoint(cfg.out_path, model, optimizer, step, cfg)
    save_checkpoint(cfg.out_path, model, optimizer, cfg.train_steps, cfg)


if __name__ == "__main__":
    main()