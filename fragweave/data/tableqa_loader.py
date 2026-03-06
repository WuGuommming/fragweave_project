from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fragweave.utils.io import read_jsonl


# BIPIA TableQA schema is close to EmailQA, but the answer key is often "ideal".
CAND_CONTEXT_KEYS = [
    "context",
    "table",
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
    "ideal",
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


def _auto_pick_key(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in row:
            return k
    return None


def _coerce_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts: List[str] = []
        for e in x:
            if e is None:
                continue
            parts.append(str(e))
        return "\n".join(parts)
    return str(x)


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
        return _coerce_text(ans)
    return str(ans)


def find_tableqa_split_file(bipia_root: str | Path, split: str) -> Path:
    bipia_root = Path(bipia_root)
    benchmark = bipia_root / "benchmark"
    if not benchmark.exists():
        raise FileNotFoundError(f"Missing benchmark/ under {bipia_root}")

    dir_candidates = [
        benchmark / "table",
        benchmark / "TableQA",
        benchmark / "tableQA",
        benchmark / "table_qa",
    ]
    table_dir = None
    for d in dir_candidates:
        if d.exists() and d.is_dir():
            table_dir = d
            break
    if table_dir is None:
        for d in benchmark.rglob("*"):
            if d.is_dir() and "table" in d.name.lower():
                table_dir = d
                break
    if table_dir is None:
        raise FileNotFoundError(f"Could not find a TableQA folder under {benchmark}")

    split = split.lower()
    direct = table_dir / f"{split}.jsonl"
    if direct.exists():
        return direct

    name_map = {
        "test": ["test.jsonl", "test_set.jsonl", "eval.jsonl"],
        "train": ["train.jsonl", "train_set.jsonl"],
        "dev": ["dev.jsonl", "valid.jsonl", "val.jsonl"],
        "validation": ["validation.jsonl", "valid.jsonl", "val.jsonl"],
    }
    for fname in name_map.get(split, []):
        p = table_dir / fname
        if p.exists():
            return p

    jsonls = sorted(table_dir.glob("*.jsonl"))
    if len(jsonls) == 1:
        return jsonls[0]

    raise FileNotFoundError(
        f"Could not find split='{split}' jsonl under {table_dir}. "
        f"Found: {[p.name for p in jsonls]}"
    )


@dataclass
class TableQASample:
    uid: str
    context: str
    question: str
    answer: Optional[str]
    raw: Dict[str, Any]


def load_tableqa_samples(
    bipia_root: str | Path,
    split: str = "test",
    *,
    max_samples: Optional[int] = None,
    context_key: Optional[str] = None,
    question_key: Optional[str] = None,
    answer_key: Optional[str] = None,
    id_key: Optional[str] = None,
) -> Tuple[List[TableQASample], Dict[str, str]]:
    path = find_tableqa_split_file(bipia_root, split)
    rows = read_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]

    if not rows:
        raise ValueError(f"No rows loaded from {path}")

    probe = rows[0]
    ck = context_key or _auto_pick_key(probe, CAND_CONTEXT_KEYS)
    qk = question_key or _auto_pick_key(probe, CAND_QUESTION_KEYS)
    ak = answer_key or _auto_pick_key(probe, CAND_ANSWER_KEYS)
    ik = id_key or _auto_pick_key(probe, CAND_ID_KEYS)

    if ck is None or qk is None:
        raise KeyError(
            f"Could not auto-detect keys in {path}. Available keys: {list(probe.keys())}. "
            "Please set dataset.context_key and dataset.question_key in the config."
        )

    samples: List[TableQASample] = []
    for i, r in enumerate(rows):
        uid = str(r.get(ik, i)) if ik else str(i)
        ctx = _coerce_text(r.get(ck, ""))
        q = _coerce_text(r.get(qk, ""))
        ans = _coerce_answer(r.get(ak)) if ak else None
        samples.append(TableQASample(uid=uid, context=ctx, question=q, answer=ans, raw=r))

    used = {
        "jsonl_path": str(path),
        "context_key": ck,
        "question_key": qk,
        "answer_key": ak or "",
        "id_key": ik or "",
    }
    return samples, used
