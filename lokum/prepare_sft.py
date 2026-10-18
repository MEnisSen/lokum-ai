from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from datasets import load_dataset

from lokum.chat_template import ANSWER_CLOSE, ANSWER_OPEN, THINK_CLOSE, THINK_OPEN, encode_chat

# repo, config or None, optional language filter, gsm8k formatting
SOURCES = [
    ("AlicanKiraz0/Turkish-SFT-Dataset-v1.0", None, None, False),
    ("AlicanKiraz0/Turkce-Atlas-Instruct", None, None, False),
    ("CohereLabs/aya_dataset", None, "tr", False),
    ("SoAp9035/everyday-conversations-tur", None, None, False),
    ("alibayram/diyalog-dataset", None, None, False),
    ("Metin/WikiRAG-TR", None, None, False),
    ("selimc/InstructPapers-TR", None, None, False),
    ("ytu-ce-cosmos/gsm8k_tr", None, None, True),
]


def messages_of(row: dict) -> list[dict] | None:
    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"]
    pairs = (
        ("instruction", "output"),
        ("instruction", "response"),
        ("question", "answer"),
        ("prompt", "completion"),
        ("input", "output"),
    )
    for user_key, asst_key in pairs:
        if row.get(user_key) and row.get(asst_key):
            return [
                {"role": "user", "content": str(row[user_key])},
                {"role": "assistant", "content": str(row[asst_key])},
            ]
    return None


def gsm8k_messages(row: dict) -> list[dict] | None:
    question = row.get("question") or row.get("soru")
    answer = row.get("answer") or row.get("cevap")
    if not question or not answer:
        return None
    answer = str(answer)
    if "####" in answer:
        reasoning, final = answer.split("####", 1)
        content = f"{THINK_OPEN}{reasoning.strip()}{THINK_CLOSE}\n{ANSWER_OPEN}{final.strip()}{ANSWER_CLOSE}"
    else:
        content = f"{ANSWER_OPEN}{answer.strip()}{ANSWER_CLOSE}"
    return [{"role": "user", "content": str(question)}, {"role": "assistant", "content": content}]


def keep_language(row: dict, language: str | None) -> bool:
    if language is None:
        return True
    value = str(row.get("language") or row.get("lang") or "").lower()
    return value in {"tr", "tur", "turkish", "türkçe"}


def pack(rows: list[tuple[list[int], list[int]]], context: int, eot: int) -> tuple[np.ndarray, np.ndarray]:
    blocks_i: list[list[int]] = []
    blocks_m: list[list[int]] = []
    buf_i: list[int] = []
    buf_m: list[int] = []

    def flush() -> None:
        if not buf_i:
            return
        pad = context - len(buf_i)
        blocks_i.append(buf_i + [eot] * pad)
        blocks_m.append(buf_m + [0] * pad)
        buf_i.clear()
        buf_m.clear()

    for ids, mask in rows:
        if len(ids) > context:
            ids, mask = ids[:context], mask[:context]
        if buf_i and len(buf_i) + len(ids) > context:
            flush()
        buf_i.extend(ids)
        buf_m.extend(mask)
    flush()
    if not blocks_i:
        raise RuntimeError("no SFT rows were packed")
    return np.asarray(blocks_i, dtype=np.int32), np.asarray(blocks_m, dtype=np.int8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/lokum")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--dev-frac", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=0, help="0 means every row")
    args = parser.parse_args()
    train: list[tuple[list[int], list[int]]] = []
    dev: list[tuple[list[int], list[int]]] = []
    eot = None
    for repo, config, language, gsm in SOURCES:
        try:
            ds = load_dataset(repo, config, split="train") if config else load_dataset(repo, split="train")
        except Exception as exc:
            print("skip", repo, exc)
            continue
        print(repo, "keys", ds.column_names)
        for i, row in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            if not keep_language(row, language):
                continue
            messages = gsm8k_messages(row) if gsm else messages_of(row)
            if not messages:
                continue
            ids, mask = encode_chat(messages)
            if sum(mask) == 0 or len(ids) > args.context:
                continue
            eot = ids[-1]
            (dev if i % int(1 / args.dev_frac) == 0 else train).append((ids, mask))
    if eot is None:
        raise SystemExit("no SFT rows; print a dataset row and fix messages_of")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sft_train", train), ("sft_dev", dev)):
        tokens, loss_mask = pack(rows, args.context, eot)
        path = out / f"{name}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("tokens", data=tokens)
            handle.create_dataset("loss_mask", data=loss_mask)
        print(name, tokens.shape, "masked", int(loss_mask.sum()))


if __name__ == "__main__":
    main()