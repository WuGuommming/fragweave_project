from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fragweave.utils.io import read_jsonl


CAND_CONTEXT_KEYS = [
    "context",
    "email",
    "external_content",
    "document",
    "source",
    "text",
]
CAND_QUESTION_KEYS = [
    "question",
    "q",
    "query",
    "user_question",
    "task",
]
CAND_ANSWER_KEYS = [
    "answer",
    "answers",
    "ground_truth",
    "label",
    "expected",
]
CAND_ID_KEYS = [
    "id",
    "uid",
    "example_id",
    "idx",
]


@dataclass
class EmailQASample:
    uid: str
    context: str
    question: str
    answer: Optional[str]
    raw: Dict[str, Any]


def _auto_pick_key(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in row:
            return k
    return None


def _coerce_answer(ans: Any) -> Optional[str]:
    if ans is None:
        return None
    if isinstance(ans, str):
        return ans
    if isinstance(ans, list) and ans:
        # choose first stringy element
        for x in ans:
            if isinstance(x, str) and x.strip():
                return x
        return str(ans[0])
    return str(ans)


def find_emailqa_split_file(bipia_root: str | Path, split: str) -> Path:
    """Try to locate an EmailQA jsonl file under BIPIA's benchmark folder.

    We intentionally keep this conservative: if we find multiple plausible files,
    we pick the most obvious names (e.g., test.jsonl/train.jsonl) and otherwise fail.
    """

    bipia_root = Path(bipia_root)
    benchmark = bipia_root / "benchmark"
    if not benchmark.exists():
        raise FileNotFoundError(f"Missing benchmark/ under {bipia_root}")

    # Common layout per BIPIA docs: benchmark/email/{train.jsonl,test.jsonl}
    email_dir_candidates = [
        benchmark / "email",
        benchmark / "EmailQA",
        benchmark / "emailQA",
        benchmark / "email_qa",
    ]
    email_dir = None
    for d in email_dir_candidates:
        if d.exists() and d.is_dir():
            email_dir = d
            break
    if email_dir is None:
        # fallback: scan benchmark/** for directory containing 'email'
        for d in benchmark.rglob("*"):
            if d.is_dir() and "email" in d.name.lower():
                email_dir = d
                break
    if email_dir is None:
        raise FileNotFoundError(f"Could not find an EmailQA folder under {benchmark}")

    # pick split file
    split = split.lower()
    direct = email_dir / f"{split}.jsonl"
    if direct.exists():
        return direct

    # common naming
    name_map = {
        "test": ["test.jsonl", "test_set.jsonl", "eval.jsonl"],
        "train": ["train.jsonl", "train_set.jsonl"],
        "dev": ["dev.jsonl", "valid.jsonl", "val.jsonl"],
        "validation": ["validation.jsonl", "valid.jsonl", "val.jsonl"],
    }
    for fname in name_map.get(split, []):
        p = email_dir / fname
        if p.exists():
            return p

    # conservative fallback: if exactly one jsonl exists, use it.
    jsonls = sorted(email_dir.glob("*.jsonl"))
    if len(jsonls) == 1:
        return jsonls[0]

    raise FileNotFoundError(
        f"Could not find split='{split}' jsonl under {email_dir}. "
        f"Found: {[p.name for p in jsonls]}\n"
        "Set dataset.context_key/question_key/answer_key in config if the schema differs."
    )


def load_emailqa_samples(
    bipia_root: str | Path,
    split: str = "test",
    *,
    max_samples: Optional[int] = None,
    context_key: Optional[str] = None,
    question_key: Optional[str] = None,
    answer_key: Optional[str] = None,
    id_key: Optional[str] = None,
) -> Tuple[List[EmailQASample], Dict[str, str]]:
    path = find_emailqa_split_file(bipia_root, split)
    rows = read_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]

    if not rows:
        raise ValueError(f"No rows loaded from {path}")

    # auto-detect keys (can be overridden by config)
    probe = rows[0]
    ck = context_key or _auto_pick_key(probe, CAND_CONTEXT_KEYS)
    qk = question_key or _auto_pick_key(probe, CAND_QUESTION_KEYS)
    ak = answer_key or _auto_pick_key(probe, CAND_ANSWER_KEYS)
    ik = id_key or _auto_pick_key(probe, CAND_ID_KEYS)

    if ck is None or qk is None:
        raise KeyError(
            f"Could not auto-detect keys in {path}. "
            f"Available keys: {list(probe.keys())}. "
            "Please set dataset.context_key and dataset.question_key in the config."
        )

    samples: List[EmailQASample] = []
    for i, r in enumerate(rows):
        uid = str(r.get(ik, i)) if ik else str(i)
        ctx = str(r.get(ck, ""))
        q = str(r.get(qk, ""))
        ans = _coerce_answer(r.get(ak)) if ak else None
        samples.append(EmailQASample(uid=uid, context=ctx, question=q, answer=ans, raw=r))

    used = {"jsonl_path": str(path), "context_key": ck, "question_key": qk, "answer_key": ak or "", "id_key": ik or ""}
    return samples, used
