from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import regex

EOT = "<|endoftext|>"
# No English contraction rule. Digits are one character each, so "42" is two tokens.
TURKISH_SPLIT = (
    r"[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]++[\r\n]*"
    r"|\s*[\r\n]|\s+(?!\S)|\s+"
)


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    a, b = pair
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]], pattern: str = TURKISH_SPLIT):
        self.pattern = pattern
        self._split = regex.compile(pattern)
        self.merges = [(int(a), int(b)) for a, b in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.bytes.append(self.bytes[a] + self.bytes[b])
        self.eot_token = 256 + len(self.merges)
        self._cache: dict[str, list[int]] = {}

    @property
    def n_vocab(self) -> int:
        return self.eot_token + 1

    @classmethod
    def train(cls, texts: list[str], vocab_size: int, pattern: str = TURKISH_SPLIT) -> "BPETokenizer":
        n_merges = vocab_size - 256 - 1
        if n_merges < 0:
            raise ValueError("vocab_size must be at least 257")
        split = regex.compile(pattern)
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(split.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in counts]
        freqs = list(counts.values())

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)
        heap = [(-n, pair) for pair, n in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if current <= 0:
                continue
            if -neg != current:
                heapq.heappush(heap, (-current, pair))
                continue
            new_id = 256 + len(merges)
            merges.append(pair)
            changed: set[tuple[int, int]] = set()
            for wi in where.pop(pair, ()):
                old = words[wi]
                new = _merge(old, pair, new_id)
                if len(new) == len(old):
                    continue
                freq = freqs[wi]
                for p in zip(old, old[1:]):
                    pair_counts[p] -= freq
                for p in zip(new, new[1:]):
                    pair_counts[p] += freq
                    where[p].add(wi)
                    changed.add(p)
                words[wi] = new
            pair_counts.pop(pair, None)
            for p in changed:
                if pair_counts.get(p, 0) > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))
            if len(merges) % 500 == 0 or len(merges) == n_merges:
                print(f"merge {len(merges)}/{n_merges}")
        return cls(merges, pattern)

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, 10**9))
            rank = self.ranks.get(pair)
            if rank is None:
                break
            ids = _merge(ids, pair, 256 + rank)
        if len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self._split.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            i = int(i)
            if i == self.eot_token:
                parts.append(EOT.encode())
            elif 0 <= i < len(self.bytes):
                parts.append(self.bytes[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"pattern": self.pattern, "merges": self.merges}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [(int(a), int(b)) for a, b in data["merges"]]
        return cls(merges, data["pattern"])