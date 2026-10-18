from __future__ import annotations

import argparse

import h5py
import torch

from lokum.checkpoint import save_checkpoint
from lokum.config import Config
from lokum.losses import sft_loss
from lokum.model import Transformer


def rows(path: str, batch: int, device: str):
    with h5py.File(path, "r") as handle:
        tokens = handle["tokens"]
        mask = handle["loss_mask"]
        n = tokens.shape[0]
        order = torch.randperm(n).tolist()
        for start in range(0, n, batch):
            idx = sorted(order[start : start + batch])
            if len(idx) < batch:
                continue
            x = torch.tensor(tokens[idx], dtype=torch.long, device=device)
            m = torch.tensor(mask[idx], dtype=torch.float, device=device)
            yield x, m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="models/base.pt")
    parser.add_argument("--train", default="data/lokum/sft_train.h5")
    parser.add_argument("--dev", default="data/lokum/sft_dev.h5")
    parser.add_argument("--out", default="models/sft.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    step = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        micro = 0
        for tokens, mask in rows(args.train, args.batch, device):
            warm = min(1.0, (step + 1) / max(1, args.warmup))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warm
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits, _ = model(tokens)
                loss = sft_loss(logits, tokens, mask) / args.grad_accum
            loss.backward()
            running += float(loss)
            micro += 1
            if micro == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step} loss {running:.4f}")
                step += 1
                running = 0.0
                micro = 0
        if args.warmup >= step and epoch == 0:
            print("warning: warmup is longer than the steps in one epoch; lower --warmup")
    cfg.out_path = args.out
    save_checkpoint(args.out, model, optimizer, step, cfg)


if __name__ == "__main__":
    main()