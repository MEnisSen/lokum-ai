from __future__ import annotations

import re

from lokum.chat_template import ANSWER_CLOSE, ANSWER_OPEN

_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _float(text: str) -> float | None:
    match = _NUM.search(text.replace(",", "."))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    tagged = re.findall(re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), text, flags=re.S)
    if tagged:
        return _float(tagged[-1])
    marked = re.findall(r"####\s*([-+]?\d+(?:[.,]\d+)?)", text)
    if marked:
        return _float(marked[-1])
    numbers = _NUM.findall(text)
    return _float(numbers[-1]) if numbers else None


def score(text: str, gold: float) -> float:
    pred = extract_answer(text)
    correct = pred is not None and abs(pred - gold) <= 1e-4
    one_block = text.count(ANSWER_OPEN) == 1 and text.count(ANSWER_CLOSE) == 1
    return min((1.0 if correct else 0.0) + (0.2 if one_block else 0.0), 1.2)