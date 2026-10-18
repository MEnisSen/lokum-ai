from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from lokum.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/turkish_sample.txt")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--train-mb", type=int, default=200)
    parser.add_argument("--tokenizer", default="data/lokum/tokenizer.json")
    parser.add_argument("--smoke-dir", default="data/lokum_smoke")
    parser.add_argument("--smoke-chars", type=int, default=2_000_000)
    args = parser.parse_args()

    raw = Path(args.input).read_text(encoding="utf-8", errors="replace")
    budget = args.train_mb * 1024 * 1024
    sample = raw[:budget]
    print(f"training on {len(sample):,} characters")
    tok = BPETokenizer.train([sample], vocab_size=args.vocab_size)
    tok.save(args.tokenizer)
    print(f"saved {args.tokenizer}  vocab {tok.n_vocab}  eot {tok.eot_token}")

    smoke = raw[: args.smoke_chars]
    docs = [d.strip() for d in smoke.split("\n\n") if d.strip()]
    if len(docs) < 4:
        docs = [smoke[i : i + 2000] for i in range(0, len(smoke), 2000) if smoke[i : i + 2000].strip()]
    train_ids: list[int] = []
    val_ids: list[int] = []
    for i, doc in enumerate(docs):
        ids = tok.encode_ordinary(doc[:100_000]) + [tok.eot_token]
        (val_ids if i % 200 == 0 else train_ids).extend(ids)
    out = Path(args.smoke_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", train_ids), ("val", val_ids or train_ids[:50_000])):
        path = out / f"{name}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("tokens", data=np.asarray(ids, dtype=np.int32))
        print(f"wrote {path}  {len(ids):,} tokens")


if __name__ == "__main__":
    main()