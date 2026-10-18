from __future__ import annotations

import argparse

import torch

from lokum.chat_template import ASSISTANT, USER, decode, get_tokenizer
from lokum.checkpoint import load_model
from lokum.generate import generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/sft.pt")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new", type=int, default=256)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = load_model(args.ckpt, device)
    tok = get_tokenizer()
    history: list[int] = []
    print("empty line to quit")
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            break
        if not user:
            break
        turn = tok.encode_ordinary(USER + user) + [tok.eot_token] + tok.encode_ordinary(ASSISTANT)
        history.extend(turn)
        idx = torch.tensor([history], dtype=torch.long, device=device)
        out = generate(model, idx, args.max_new, args.temperature, args.top_k, tok.eot_token, args.top_p)
        new_ids = out[0, idx.size(1) :].tolist()
        history.extend(new_ids)
        print("lokum>", decode(new_ids))


if __name__ == "__main__":
    main()