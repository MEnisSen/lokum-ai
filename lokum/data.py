from __future__ import annotations

from collections.abc import Iterator

import h5py
import numpy as np
import torch


def batch_iterator(
    path: str,
    batch_size: int,
    context_length: int,
    device: str,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    with h5py.File(path, "r") as handle:
        data = handle["tokens"]
        n = int(data.shape[0])
        if n <= context_length:
            raise RuntimeError(f"{path} has {n} tokens, which is shorter than context {context_length}")
        span = n - context_length
        while True:
            starts = np.random.randint(0, span, size=batch_size)
            rows = [data[int(s) : int(s) + context_length + 1] for s in starts]
            batch = torch.tensor(np.stack(rows), dtype=torch.long, device=device)
            yield batch[:, :-1], batch[:, 1:]