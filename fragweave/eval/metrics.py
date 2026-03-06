from __future__ import annotations

import string
from typing import Optional


_ARTICLES = {"a", "an", "the"}


def _remove_articles(text: str) -> str:
    return " ".join([w for w in text.split() if w not in _ARTICLES])


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


def _remove_punc(text: str) -> str:
    return "".join(ch for ch in text if ch not in set(string.punctuation))


def _lower(text: str) -> str:
    return text.lower()


def normalize(s: str) -> str:
    """SQuAD-style normalization (close to what many QA benchmarks use)."""
    s = s.strip()
    s = _lower(s)
    s = _remove_punc(s)
    s = _remove_articles(s)
    s = _white_space_fix(s)
    return s


def exact_match(pred: str, gold: Optional[str]) -> Optional[bool]:
    if gold is None:
        return None
    return normalize(pred) == normalize(gold)
