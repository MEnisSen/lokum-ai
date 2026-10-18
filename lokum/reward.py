from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

from lokum.chat_template import ASSISTANT, get_tokenizer
from lokum.checkpoint import load_model
from lokum.losses import bradley_terry


class RewardHead(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.head = nn.Linear(n_embed, 1)

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        last = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        return self.head(last).squeeze(-1)


def encode_pair(prompt: str, answer: str, context: int, device: str):
    tok = get_tokenizer()
    ids = tok.encode_ordinary(prompt + ASSISTANT + answer) + [tok.eot_token]
    ids = ids[:context]
    length = len(ids) - 1
    ids = ids + [tok.eot_token] * (context - len(ids))
    return torch.tensor(ids, dtype=torch.long, device=device), length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="models/sft.pt")
    parser.add_argument("--data", default="data/lokum/preferences.jsonl")
    parser.add_argument("--out", default="models/reward.pt")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-len", type=int, default=768)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_model(args.init, device)
    if args.max_len > cfg.context_length:
        raise SystemExit("max-len must be <= context_length")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    head = RewardHead(cfg.n_embed).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    right = 0
    for step in range(args.steps):
        batch = [rows[(step * args.batch + i) % len(rows)] for i in range(args.batch)]
        chosen, rejected, lc, lr = [], [], [], []
        for row in batch:
            c, clen = encode_pair(row["prompt"], row["chosen"], args.max_len, device)
            r, rlen = encode_pair(row["prompt"], row["rejected"], args.max_len, device)
            chosen.append(c)
            rejected.append(r)
            lc.append(clen)
            lr.append(rlen)
        with torch.no_grad():
            hc = model.hidden(torch.stack(chosen))
            hr = model.hidden(torch.stack(rejected))
        sc = head(hc, torch.tensor(lc, device=device))
        sr = head(hr, torch.tensor(lr, device=device))
        loss = bradley_terry(sc, sr)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        right += int((sc > sr).sum())
        if step % 20 == 0:
            print(f"step {step} loss {float(loss):.4f} acc {right / ((step + 1) * args.batch):.3f}")
    torch.save({"head": head.state_dict(), "init": args.init, "config": cfg.to_dict()}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()