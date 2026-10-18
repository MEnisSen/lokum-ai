from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from lokum.tokenizer import BPETokenizer

WRITE_CHUNK = 8_000_000
VAL_EVERY = 200
VAL_CAP = 10_000_000
DOC_CHAR_CAP = 100_000

# name, repo, config or None, train-token cap
MIX = [
    ("web", "epfml/FineWeb2-HQ", "tur_Latn", 1_800_000_000),
    ("wiki", "HuggingFaceFW/finewiki", "tr", 375_000_000),
    ("pdfs", "HuggingFaceFW/finepdfs-edu", "tur_Latn", 225_000_000),
    ("cosmos", "Berkesule/COSMOS-Sentetic-Turkish-Corpus-2GB", None, 300_000_000),
    ("stories", "BILGEM-AI/BILGE-Synthetic-Stories", None, 150_000_000),
    ("math", "BILGEM-AI/BILGE-Synthetic-Math", None, 75_000_000),
    ("finemath", "HuggingFaceTB/finemath", "finemath-4plus", 75_000_000),
]


class TokenWriter:
    def __init__(self, path: Path, append: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        if append:
            self.handle = h5py.File(path, "a")
            self.data = self.handle["tokens"]
            self.total = int(self.data.shape[0])
        else:
            self.handle = h5py.File(path, "w")
            self.data = self.handle.create_dataset(
                "tokens",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=(WRITE_CHUNK,),
            )
            self.total = 0
        self.buf: list[int] = []

    def add(self, ids: list[int]) -> None:
        self.buf.extend(ids)
        if len(self.buf) >= WRITE_CHUNK:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        arr = np.asarray(self.buf, dtype=np.int32)
        start = self.data.shape[0]
        self.data.resize((start + arr.shape[0],))
        self.data[start:] = arr
        self.total += int(arr.shape[0])
        self.buf.clear()

    def close(self) -> None:
        self.flush()
        self.handle.close()


def text_of(row: dict) -> str:
    for key in ("text", "content", "raw_content", "story", "document", "markdown", "corpus_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(f"no text field in {list(row)}")


def stream(repo: str, config: str | None):
    if config is None:
        return load_dataset(repo, split="train", streaming=True)
    return load_dataset(repo, config, split="train", streaming=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--start-at", default="web")
    parser.add_argument("--tokenizer", default="data/lokum/tokenizer.json")
    parser.add_argument("--out-dir", default="data/lokum")
    parser.add_argument("--max-train-tokens", type=int, default=3_000_000_000)
    args = parser.parse_args()
    tok = BPETokenizer.load(args.tokenizer)
    out = Path(args.out_dir)
    train = TokenWriter(out / "train.h5", append=args.append)
    val = TokenWriter(out / "val.h5", append=args.append)
    names = [item[0] for item in MIX]
    if args.start_at not in names:
        raise SystemExit(f"unknown source {args.start_at}")
    mix = MIX[names.index(args.start_at):]
    seen = 0
    try:
        for name, repo, config, cap in mix:
            if train.total >= args.max_train_tokens:
                break
            local_cap = min(cap, args.max_train_tokens - train.total)
            got = 0
            bar = tqdm(desc=name, total=local_cap, unit="tok")
            for row in stream(repo, config):
                if got >= local_cap or train.total >= args.max_train_tokens:
                    break
                text = text_of(row)[:DOC_CHAR_CAP]
                digest = hashlib.sha1(text[:400].encode("utf-8", errors="replace")).digest()
                hold_out = digest[0] % VAL_EVERY == 0 and val.total < VAL_CAP
                ids = tok.encode_ordinary(text) + [tok.eot_token]
                if hold_out:
                    val.add(ids)
                else:
                    train.add(ids)
                    got += len(ids)
                    bar.update(len(ids))
                seen += 1
            bar.close()
            train.flush()
            val.flush()
            print(f"{name}: train {train.total:,}  val {val.total:,}  docs {seen:,}")
    finally:
        train.close()
        val.close()
    print(f"done  train {train.total:,}  val {val.total:,}")


if __name__ == "__main__":
    main()