from __future__ import annotations

from functools import lru_cache

from lokum.tokenizer import BPETokenizer

USER = "<|user|>\n"
ASSISTANT = "<|assistant|>\n"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"


@lru_cache(maxsize=1)
def get_tokenizer() -> BPETokenizer:
    return BPETokenizer.load("data/lokum/tokenizer.json")


def eot_id() -> int:
    return get_tokenizer().eot_token


def encode_chat(messages: list[dict], add_generation_prompt: bool = False) -> tuple[list[int], list[int]]:
    tok = get_tokenizer()
    ids: list[int] = []
    mask: list[int] = []
    for message in messages:
        header = {"user": USER, "assistant": ASSISTANT, "system": "<|system|>\n"}[message["role"]]
        header_ids = tok.encode_ordinary(header)
        ids.extend(header_ids)
        mask.extend([0] * len(header_ids))
        content_ids = tok.encode_ordinary(message["content"])
        learn = message["role"] == "assistant"
        ids.extend(content_ids)
        mask.extend([1 if learn else 0] * len(content_ids))
        ids.append(tok.eot_token)
        mask.append(1 if learn else 0)
    if add_generation_prompt:
        header_ids = tok.encode_ordinary(ASSISTANT)
        ids.extend(header_ids)
        mask.extend([0] * len(header_ids))
    return ids, mask


def decode(ids: list[int]) -> str:
    tok = get_tokenizer()
    return tok.decode([i for i in ids if int(i) != tok.eot_token])