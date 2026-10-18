from __future__ import annotations

import argparse
import copy
import json

import torch

from lokum.chat_template import encode_chat
from lokum.checkpoint import load_model, save_checkpoint
from lokum.losses import dpo_loss, sequence_logprobs


def encode(prompt: str, answer: str, context: int, device: str):
    ids, mask = encode_chat(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    )
    ids, mask = ids[:context], mask[:context]
    pad = context - len(ids)
    eot = ids[-1]
    ids = ids + [eot] * pad
    mask = mask + [0] * pad
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float, device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="models/sft.pt")
    parser.add_argument("--data", default="data/lokum/preferences.jsonl")
    parser.add_argument("--out", default="models/dpo.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=768)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, cfg, _ = load_model(args.init, device)
    policy.train()
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    if args.max_len > cfg.context_length:
        raise SystemExit("max-len must be <= context_length")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    for step in range(args.steps):
        batch = [rows[(step * args.batch + i) % len(rows)] for i in range(args.batch)]
        chosen_x, chosen_m, rejected_x, rejected_m = [], [], [], []
        for row in batch:
            x, m = encode(row["prompt"], row["chosen"], args.max_len, device)
            y, n = encode(row["prompt"], row["rejected"], args.max_len, device)
            chosen_x.append(x)
            chosen_m.append(m)
            rejected_x.append(y)
            rejected_m.append(n)
        cx, cm = torch.stack(chosen_x), torch.stack(chosen_m)
        rx, rm = torch.stack(rejected_x), torch.stack(rejected_m)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            pi_c, _ = policy(cx)
            pi_r, _ = policy(rx)
        with torch.no_grad():
            ref_c, _ = ref(cx)
            ref_r, _ = ref(rx)
        loss, acc = dpo_loss(
            sequence_logprobs(pi_c, cx, cm),
            sequence_logprobs(pi_r, rx, rm),
            sequence_logprobs(ref_c, cx, cm),
            sequence_logprobs(ref_r, rx, rm),
            beta=args.beta,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0:
            print(f"step {step} loss {float(loss):.4f} acc {float(acc):.3f}")
    save_checkpoint(args.out, policy, optimizer, args.steps, cfg)


if __name__ == "__main__":
    main()