from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lokum.chat_template import decode, get_tokenizer
from lokum.checkpoint import load_model
from lokum.generate import generate
from lokum.rl import prompt_ids
from lokum.verifier import score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/lokum/rl_prompts_test.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new", type=int, default=420)
    parser.add_argument("--out", default="logs/stage_table.jsonl")
    parser.add_argument(
        "--ckpts",
        nargs="+",
        default=["models/base.pt", "models/sft.pt", "models/dpo.pt", "models/ppo.pt", "models/grpo.pt"],
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")][: args.limit]
    tok = get_tokenizer()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for path in args.ckpts:
            if not Path(path).exists():
                print("missing", path)
                continue
            model, _, _ = load_model(path, device)
            correct = 0
            for row in rows:
                prompt = prompt_ids(row["prompt"], device)
                out = generate(model, prompt, args.max_new, temperature=0.0, top_k=0, eot=tok.eot_token)
                correct += score(decode(out[0].tolist()), float(row["gold"])) >= 1.0
            acc = correct / max(1, len(rows))
            print(f"{path}  {acc:.3f}  n={len(rows)}")
            handle.write(json.dumps({"ckpt": path, "accuracy": acc, "n": len(rows)}) + "\n")


if __name__ == "__main__":
    main()