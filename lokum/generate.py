from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from lokum.checkpoint import load_model
from lokum.tokenizer import BPETokenizer


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new: int,
    temperature: float,
    top_k: int,
    eot: int,
    top_p: float = 1.0,
) -> torch.Tensor:
    for _ in range(max_new):
        window = idx[:, -model.context_length :]
        logits, _ = model(window)
        logits = logits[:, -1, :].float()
        if temperature <= 0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1]
                logits = logits.masked_fill(logits < cutoff.unsqueeze(-1), float("-inf"))
            if top_p < 1.0:
                probs = F.softmax(logits, dim=-1)
                ordered, order = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(ordered, dim=-1)
                drop = cumulative - ordered > top_p
                ordered = ordered.masked_fill(drop, 0.0)
                nxt_i = torch.multinomial(ordered, num_samples=1)
                nxt = order.gather(-1, nxt_i)
            else:
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat((idx, nxt), dim=1)
        if int(nxt.item()) == eot:
            break
    return idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", default="Bir zamanlar")
    parser.add_argument("--max-new", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_model(args.ckpt, device)
    tok = BPETokenizer.load("data/lokum/tokenizer.json")
    if tok.n_vocab != cfg.vocab_size:
        raise SystemExit(f"tokenizer vocab {tok.n_vocab} != checkpoint vocab {cfg.vocab_size}")
    ids = torch.tensor([tok.encode_ordinary(args.prompt)], dtype=torch.long, device=device)
    out = generate(model, ids, args.max_new, args.temperature, args.top_k, tok.eot_token, args.top_p)
    ids_out = [i for i in out[0].tolist() if i != tok.eot_token]
    print(tok.decode(ids_out))


if __name__ == "__main__":
    main()