from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fragweave.utils.io import read_jsonl


CAND_CONTEXT_KEYS = [
    "context",
    "carrier",
    "content",
    "document",
    "doc",
    "text",
    "page",
    "post",
    "html",
    "markdown",
    "md",
    "external_content",
]
CAND_QUESTION_KEYS = [
    "question",
    "query",
    "q",
    "user_question",
    "prompt",
    "task",
]
CAND_INSTRUCTION_KEYS = [
    "malicious_instruction",
    "instruction",
    "attack",
    "injection",
    "payload",
    "malicious",
]
CAND_ID_KEYS = [
    "id",
    "uid",
    "example_id",
    "idx",
    "sample_id",
]


@dataclass
class OpenRAGSocSample:
    uid: str
    context_clean: str
    context_poisoned: Optional[str]
    question: str
    instruction: Optional[str]
    raw: Dict[str, Any]


def _auto_pick_key(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in row:
            return k
    return None


def _to_text(v: Any) -> str:
    """
    Best-effort: coerce carrier/context to a single long string.
    Supports str, list[str], list[dict], dict.
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts: List[str] = []
        for x in v:
            if isinstance(x, str):
                if x.strip():
                    parts.append(x.strip())
            elif isinstance(x, dict):
                # common patterns: {"text": "..."} or {"content": "..."}
                for kk in ("text", "content", "body", "html", "markdown", "md"):
                    if kk in x and isinstance(x[kk], str) and x[kk].strip():
                        parts.append(x[kk].strip())
                        break
                else:
                    parts.append(json.dumps(x, ensure_ascii=False))
            else:
                parts.append(str(x))
        return "\n\n---\n\n".join([p for p in parts if p])
    if isinstance(v, dict):
        # try typical carrier fields
        for kk in ("text", "content", "body", "html", "markdown", "md"):
            if kk in v and isinstance(v[kk], str) and v[kk].strip():
                return v[kk].strip()
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            # if dict-of-lists, flatten if possible
            # otherwise wrap as single row
            return [obj]
    raise ValueError(f"Unsupported file type: {path}")


def _find_data_file(root: Path) -> Path:
    # Conservative: if exactly one jsonl/json exists under root, use it.
    cands = sorted(list(root.rglob("*.jsonl")) + list(root.rglob("*.json")))
    if len(cands) == 1:
        return cands[0]
    raise FileNotFoundError(
        f"OpenRAG-Soc data file ambiguous under {root}. "
        f"Found {len(cands)} candidates. Please set dataset.openrag_data_file explicitly."
    )


def load_openrag_soc_long_samples(
    openrag_root: Union[str, Path],
    *,
    openrag_data_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    # key overrides (if None -> auto detect)
    context_key: Optional[str] = None,
    clean_context_key: Optional[str] = None,
    poisoned_context_key: Optional[str] = None,
    question_key: Optional[str] = None,
    instruction_key: Optional[str] = None,
    id_key: Optional[str] = None,
    # optional carrier type filter
    carrier_type_key: Optional[str] = None,
    carrier_type_value: Optional[str] = None,
) -> Tuple[List[OpenRAGSocSample], Dict[str, str]]:
    root = Path(openrag_root)
    if not root.exists():
        raise FileNotFoundError(f"openrag_root does not exist: {root}")

    if openrag_data_file:
        path = (root / openrag_data_file).resolve()
        if not path.exists():
            raise FileNotFoundError(f"dataset.openrag_data_file not found: {path}")
    else:
        path = _find_data_file(root)

    rows = _read_json_or_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"No rows loaded from {path}")

    probe = rows[0]

    # carrier / context
    ck = context_key or _auto_pick_key(probe, CAND_CONTEXT_KEYS)
    cck = clean_context_key  # explicit overrides (preferred)
    pck = poisoned_context_key

    # question
    qk = question_key or _auto_pick_key(probe, CAND_QUESTION_KEYS)

    # instruction
    ik = instruction_key or _auto_pick_key(probe, CAND_INSTRUCTION_KEYS)

    # id
    idk = id_key or _auto_pick_key(probe, CAND_ID_KEYS)

    if qk is None:
        raise KeyError(
            f"Could not auto-detect question key in {path}. "
            f"Available keys: {list(probe.keys())}. "
            f"Set dataset.question_key in config."
        )

    # If neither clean_context_key nor context_key exists, fail.
    if cck is None and ck is None:
        raise KeyError(
            f"Could not auto-detect context key in {path}. "
            f"Available keys: {list(probe.keys())}. "
            f"Set dataset.context_key or dataset.clean_context_key in config."
        )

    out: List[OpenRAGSocSample] = []
    kept = 0
    for i, r in enumerate(rows):
        # carrier type filter
        if carrier_type_key and carrier_type_value is not None:
            v = r.get(carrier_type_key)
            if v is None:
                continue
            if str(v) != str(carrier_type_value):
                continue

        uid = str(r.get(idk, i)) if idk else str(i)

        # clean context preference order:
        # explicit clean_context_key -> context_key -> empty
        base_ctx = _to_text(r.get(cck)) if cck else ""
        if not base_ctx and ck:
            base_ctx = _to_text(r.get(ck))

        poisoned_ctx = _to_text(r.get(pck)) if pck else None
        if poisoned_ctx is not None and not poisoned_ctx.strip():
            poisoned_ctx = None

        question = str(r.get(qk, "")).strip()
        instruction = None
        if ik and isinstance(r.get(ik), str) and str(r.get(ik)).strip():
            instruction = str(r.get(ik)).strip()

        out.append(
            OpenRAGSocSample(
                uid=uid,
                context_clean=base_ctx,
                context_poisoned=poisoned_ctx,
                question=question,
                instruction=instruction,
                raw=r,
            )
        )
        kept += 1

    used_schema = {
        "data_path": str(path),
        "context_key": ck or "",
        "clean_context_key": cck or "",
        "poisoned_context_key": pck or "",
        "question_key": qk or "",
        "instruction_key": ik or "",
        "id_key": idk or "",
        "carrier_type_key": carrier_type_key or "",
        "carrier_type_value": str(carrier_type_value) if carrier_type_value is not None else "",
        "kept_rows": str(kept),
    }
    return out, used_schema
