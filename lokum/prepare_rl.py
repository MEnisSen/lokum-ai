from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from lokum.verifier import extract_answer


def gold_of(row: dict) -> float | None:
    for key in ("answer", "cevap", "gold"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value)
        if "####" in text:
            found = extract_answer(text.split("####", 1)[-1])
            if found is not None:
                return found
        found = extract_answer(text)
        if found is not None:
            return found
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="ytu-ce-cosmos/gsm8k_tr")
    parser.add_argument("--out-dir", default="data/lokum")
    parser.add_argument("--dev-every", type=int, default=50)
    parser.add_argument("--arithmetic", type=int, default=2000)
    args = parser.parse_args()
    ds = load_dataset(args.repo, split="train")
    print("keys", ds.column_names, "splits-check: this call used split=train")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_train = n_dev = 0
    with (out / "rl_prompts_train.jsonl").open("w", encoding="utf-8") as train, (
        out / "rl_prompts_test.jsonl"
    ).open("w", encoding="utf-8") as dev:
        for i, row in enumerate(ds):
            question = row.get("question") or row.get("soru")
            gold = gold_of(row)
            if not question or gold is None:
                continue
            line = json.dumps({"prompt": str(question), "gold": gold}, ensure_ascii=False)
            if i % args.dev_every == 0:
                dev.write(line + "\n")
                n_dev += 1
            else:
                train.write(line + "\n")
                n_train += 1
    rng = random.Random(0)
    symbols = ["+", "-", "*"]
    with (out / "arithmetic_prompts.jsonl").open("w", encoding="utf-8") as handle:
        for _ in range(args.arithmetic):
            a, b = rng.randint(0, 20), rng.randint(1, 12)
            sym = rng.choice(symbols)
            gold = {"+": a + b, "-": a - b, "*": a * b}[sym]
            prompt = f"Kaç eder: {a} {sym} {b}?"
            handle.write(json.dumps({"prompt": prompt, "gold": float(gold)}, ensure_ascii=False) + "\n")
    print(f"gsm train {n_train} dev {n_dev} arithmetic {args.arithmetic}")


if __name__ == "__main__":
    main()