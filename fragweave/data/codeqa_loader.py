from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fragweave.utils.io import read_jsonl


# BIPIA CodeQA: fields are typically lists of lines (error/code/context/ideal).
CAND_ERROR_KEYS = [
    "error",
    "traceback",
    "stacktrace",
]
CAND_CODE_KEYS = [
    "code",
    "snippet",
    "program",
]
CAND_CONTEXT_KEYS = [
    "context",
    "post",
    "discussion",
    "external_content",
    "text",
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


DEFAULT_CODE_QUESTION = "Fix the error and provide the correct solution (code and/or explanation)."


def _auto_pick_key(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in row:
            return k
    return None


def _coerce_text(x: Any) -> str:
    """Turn list-of-lines or arbitrary objects into a readable text block."""
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
        return _coerce_text(ans)
    return str(ans)


def find_codeqa_split_file(bipia_root: str | Path, split: str) -> Path:
    bipia_root = Path(bipia_root)
    benchmark = bipia_root / "benchmark"
    if not benchmark.exists():
        raise FileNotFoundError(f"Missing benchmark/ under {bipia_root}")

    dir_candidates = [
        benchmark / "code",
        benchmark / "CodeQA",
        benchmark / "codeQA",
        benchmark / "code_qa",
    ]
    code_dir = None
    for d in dir_candidates:
        if d.exists() and d.is_dir():
            code_dir = d
            break
    if code_dir is None:
        for d in benchmark.rglob("*"):
            if d.is_dir() and "code" in d.name.lower():
                code_dir = d
                break
    if code_dir is None:
        raise FileNotFoundError(f"Could not find a CodeQA folder under {benchmark}")

    split = split.lower()
    direct = code_dir / f"{split}.jsonl"
    if direct.exists():
        return direct

    name_map = {
        "test": ["test.jsonl", "test_set.jsonl", "eval.jsonl"],
        "train": ["train.jsonl", "train_set.jsonl"],
        "dev": ["dev.jsonl", "valid.jsonl", "val.jsonl"],
        "validation": ["validation.jsonl", "valid.jsonl", "val.jsonl"],
    }
    for fname in name_map.get(split, []):
        p = code_dir / fname
        if p.exists():
            return p

    jsonls = sorted(code_dir.glob("*.jsonl"))
    if len(jsonls) == 1:
        return jsonls[0]

    raise FileNotFoundError(
        f"Could not find split='{split}' jsonl under {code_dir}. Found: {[p.name for p in jsonls]}"
    )


@dataclass
class CodeQASample:
    uid: str
    context: str
    question: str
    answer: Optional[str]
    raw: Dict[str, Any]


def load_codeqa_samples(
    bipia_root: str | Path,
    split: str = "test",
    *,
    max_samples: Optional[int] = None,
    context_key: Optional[str] = None,
    question_key: Optional[str] = None,
    answer_key: Optional[str] = None,
    id_key: Optional[str] = None,
    error_key: Optional[str] = None,
    code_key: Optional[str] = None,
    default_question: str = DEFAULT_CODE_QUESTION,
) -> Tuple[List[CodeQASample], Dict[str, str]]:
    path = find_codeqa_split_file(bipia_root, split)
    rows = read_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]

    if not rows:
        raise ValueError(f"No rows loaded from {path}")

    probe = rows[0]
    ek = error_key or _auto_pick_key(probe, CAND_ERROR_KEYS)
    pk = code_key or _auto_pick_key(probe, CAND_CODE_KEYS)
    ck = context_key or _auto_pick_key(probe, CAND_CONTEXT_KEYS)
    ak = answer_key or _auto_pick_key(probe, CAND_ANSWER_KEYS)
    ik = id_key or _auto_pick_key(probe, CAND_ID_KEYS)
    qk = question_key if question_key in (probe.keys() if isinstance(probe, dict) else []) else None

    if ek is None and pk is None and ck is None:
        raise KeyError(
            f"Could not auto-detect core fields in {path}. Available keys: {list(probe.keys())}. "
            "Please set dataset.context_key/error_key/code_key in the config."
        )

    samples: List[CodeQASample] = []
    for i, r in enumerate(rows):
        uid = str(r.get(ik, i)) if ik else str(i)

        error_text = _coerce_text(r.get(ek)) if ek else ""
        code_text = _coerce_text(r.get(pk)) if pk else ""
        post_text = _coerce_text(r.get(ck)) if ck else ""

        # Build a single context block that looks like a typical code-debugging post.
        ctx_parts: List[str] = []
        if error_text.strip():
            ctx_parts.append("ERROR:\n" + error_text.strip())
        if code_text.strip():
            ctx_parts.append("CODE:\n" + code_text.strip())
        if post_text.strip():
            ctx_parts.append("POST:\n" + post_text.strip())
        ctx = "\n\n".join(ctx_parts).strip()

        if qk and isinstance(r.get(qk), str) and str(r.get(qk)).strip():
            q = str(r.get(qk)).strip()
        else:
            q = default_question

        ans = _coerce_answer(r.get(ak)) if ak else None
        samples.append(CodeQASample(uid=uid, context=ctx, question=q, answer=ans, raw=r))

    used = {
        "jsonl_path": str(path),
        "context_key": ck or "",
        "error_key": ek or "",
        "code_key": pk or "",
        "question_key": qk or "",
        "answer_key": ak or "",
        "id_key": ik or "",
        "default_question": default_question,
    }
    return samples, used
