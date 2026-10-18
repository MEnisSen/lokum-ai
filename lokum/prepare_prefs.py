from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def text_of(value) -> str:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict) and not isinstance(value.get("content"), str):
        content = value.get("content")
        role = value.get("role")
        if content is not None and not isinstance(content, str):
            contents = list(content)
            roles = list(role) if role is not None and not isinstance(role, str) else [None] * len(contents)
            value = [{"content": c, "role": r} for c, r in zip(contents, roles)]
        else:
            return str(value.get("content") or value.get("value") or "").strip()
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict):
                text = str(item.get("content") or item.get("value") or "").strip()
                role = item.get("role")
            else:
                text, role = str(item).strip(), None
            if text and role in (None, "assistant"):
                return text
        return ""
    return ""


def row_to_pair(row: dict) -> dict:
    prompt = text_of(row.get("prompt") or row.get("question") or row.get("instruction"))
    chosen = text_of(row.get("chosen") or row.get("chosen_response") or row.get("output_a"))
    rejected = text_of(row.get("rejected") or row.get("rejected_response") or row.get("output_b"))
    if not prompt or not chosen or not rejected:
        raise KeyError(f"could not read a pair from keys {list(row)}")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="selimc/orpo-dpo-mix-TR-20k")
    parser.add_argument("--out-dir", default="data/lokum")
    parser.add_argument("--dev-every", type=int, default=50)
    args = parser.parse_args()
    ds = load_dataset(args.repo, split="train")
    print("keys", ds.column_names)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path, dev_path = out / "preferences.jsonl", out / "preferences_test.jsonl"
    n_train = n_dev = 0
    with train_path.open("w", encoding="utf-8") as train, dev_path.open("w", encoding="utf-8") as dev:
        for i, row in enumerate(ds):
            try:
                pair = row_to_pair(row)
            except KeyError:
                print("skip row", i, {k: type(row[k]).__name__ for k in ("prompt", "chosen", "rejected")})
                continue
            handle = dev if i % args.dev_every == 0 else train
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
            if handle is dev:
                n_dev += 1
            else:
                n_train += 1
    print(f"train {n_train}  dev {n_dev}")


if __name__ == "__main__":
    main()