from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fragweave.utils.io import read_jsonl
from fragweave.data.bipia_fetch import ensure_bipia_benchmark_jsonl


@dataclass
class TaskSample:
    uid: str
    context: str
    question: str
    answer: Optional[str]
    raw: Dict[str, Any]


_CAND_CONTEXT_KEYS = [
    "context",
    "external_content",
    "document",
    "source",
    "text",
    "carrier",
    "table",
    "code",
]
_CAND_QUESTION_KEYS = ["question", "q", "query", "user_question", "task"]
_CAND_ANSWER_KEYS = ["answer", "answers", "ground_truth", "label", "expected"]
_CAND_ID_KEYS = ["id", "uid", "example_id", "idx"]


def _auto_pick_key(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in row:
            return k
    return None


def _coerce_text(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        # common for table/code carriers
        return "\n".join(str(x) for x in val)
    if isinstance(val, dict):
        # stable-ish rendering
        return "\n".join(f"{k}: {v}" for k, v in val.items())
    return str(val)


def _coerce_answer(ans: Any) -> Optional[str]:
    if ans is None:
        return None
    if isinstance(ans, str):
        return ans
    if isinstance(ans, list) and ans:
        for x in ans:
            if isinstance(x, str) and x.strip():
                return x
        return str(ans[0])
    return str(ans)


def find_bipia_task_split_file(bipia_root: str | Path, task: str, split: str) -> Path:
    """Locate a BIPIA task jsonl file under benchmark/.

    This loader is intentionally robust to slight naming differences.
    """
    bipia_root = Path(bipia_root)
    benchmark = bipia_root / "benchmark"
    if not benchmark.exists():
        raise FileNotFoundError(f"Missing benchmark/ under {bipia_root}")

    t = (task or "").lower()
    split = (split or "test").lower()

    # Candidate directories
    if "email" in t:
        needles = ["email", "emailqa", "email_qa"]
    elif "table" in t:
        needles = ["table", "tableqa", "table_qa"]
    elif "code" in t:
        needles = ["code", "codeqa", "code_qa"]
    elif t in {"web_qa", "webqa", "qa", "web"} or "web" in t:
        needles = ["qa", "webqa", "web_qa", "web"]
    elif t in {"summarization", "summary", "abstract"} or "abstract" in t:
        needles = ["abstract", "summarization", "summary"]
    else:
        needles = [t]

    cand_dirs: List[Path] = []
    for d in benchmark.rglob("*"):
        if d.is_dir() and any(n in d.name.lower() for n in needles):
            cand_dirs.append(d)

    # Prefer shallow dirs (closer to benchmark root)
    cand_dirs = sorted(cand_dirs, key=lambda p: len(p.parts))
    if not cand_dirs:
        raise FileNotFoundError(f"Could not find task folder for task='{task}' under {benchmark}")

    # Try direct match first
    name_map = {
        "test": ["test.jsonl", "test_set.jsonl", "eval.jsonl"],
        "train": ["train.jsonl", "train_set.jsonl"],
        "dev": ["dev.jsonl", "valid.jsonl", "val.jsonl"],
        "validation": ["validation.jsonl", "valid.jsonl", "val.jsonl"],
    }
    want_names = [f"{split}.jsonl"] + name_map.get(split, [])

    for d in cand_dirs:
        for fname in want_names:
            p = d / fname
            if p.exists():
                return p

    # Conservative fallback: if a directory has exactly one jsonl, use it.
    for d in cand_dirs:
        jsonls = sorted(d.glob("*.jsonl"))
        if len(jsonls) == 1:
            return jsonls[0]

    raise FileNotFoundError(
        f"Could not find split='{split}' jsonl for task='{task}' under candidates: {[str(d) for d in cand_dirs[:6]]}"
    )


def load_bipia_task_samples(
    bipia_root: str | Path,
    task: str,
    split: str = "test",
    *,
    max_samples: Optional[int] = None,
    context_key: Optional[str] = None,
    question_key: Optional[str] = None,
    answer_key: Optional[str] = None,
    id_key: Optional[str] = None,
) -> Tuple[List[TaskSample], Dict[str, str]]:
    # Generate license-gated benchmark jsonl if missing.
    ensure_bipia_benchmark_jsonl(bipia_root, task)

    path = find_bipia_task_split_file(bipia_root, task, split)
    rows = read_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"No rows loaded from {path}")

    probe = rows[0]
    ck = context_key or _auto_pick_key(probe, _CAND_CONTEXT_KEYS)
    qk = question_key or _auto_pick_key(probe, _CAND_QUESTION_KEYS)
    ak = answer_key or _auto_pick_key(probe, _CAND_ANSWER_KEYS)
    ik = id_key or _auto_pick_key(probe, _CAND_ID_KEYS)

    # Some BIPIA tasks (e.g., code / abstract) don't include an explicit question.
    # Provide a task-appropriate default instead of failing.
    t = (task or "").lower()
    if ck is None:
        raise KeyError(
            f"Could not auto-detect context key in {path}. Available keys: {list(probe.keys())}. "
            "Please set dataset.context_key in the config."
        )
    if qk is None:
        if t in {"summarization", "summary", "abstract"} or "abstract" in t:
            qk = None
        elif "code" in t:
            qk = None
        else:
            raise KeyError(
                f"Could not auto-detect question key in {path}. Available keys: {list(probe.keys())}. "
                "Please set dataset.question_key in the config."
            )

    samples: List[TaskSample] = []
    for i, r in enumerate(rows):
        uid = str(r.get(ik, i)) if ik else str(i)
        ctx = _coerce_text(r.get(ck))
        if qk is None:
            q = ""
            if t in {"summarization", "summary", "abstract"} or "abstract" in t:
                q = "Summarize the document in the CONTEXT."  # stable default
            elif "code" in t:
                err = _coerce_text(r.get("error"))
                code = _coerce_text(r.get("code"))
                q = (
                    "Fix the code to address the error shown below. Provide the corrected code only.\n\n"
                    f"ERROR:\n{err}\n\nCODE:\n{code}".strip()
                )
            else:
                q = "Answer the question using only the CONTEXT."
        else:
            q = _coerce_text(r.get(qk))
        ans = _coerce_answer(r.get(ak)) if ak else None
        samples.append(TaskSample(uid=uid, context=ctx, question=q, answer=ans, raw=r))

    used = {
        "jsonl_path": str(path),
        "task": (task or ""),
        "context_key": ck,
        "question_key": qk or "<default>",
        "answer_key": ak or "",
        "id_key": ik or "",
    }
    return samples, used
