from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .paths import INJSQUAD_BENCHMARK_NAME, assert_mandatory_benchmark_exists, get_default_paths
from .schema import InjSquadSample


def _first_text(record: Dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        elif isinstance(value, list):
            if value:
                return "\n".join(str(x) for x in value)
        elif isinstance(value, dict):
            if value:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            return str(value)
    return default


def _sample_id(record: Dict[str, Any], idx: int) -> str:
    for key in ("sample_id", "id", "uid", "example_id", "idx"):
        if key in record and record[key] is not None:
            return str(record[key])
    return str(idx)


def _normalize_record(record: Dict[str, Any], idx: int) -> InjSquadSample:
    question = _first_text(record, ["question", "query", "q"])
    clean_document = _first_text(
        record,
        ["clean_document", "context", "clean_context", "document", "paragraph", "article"],
    )
    injected_instruction = _first_text(
        record,
        ["injected_instruction", "instruction", "attack_instruction", "malicious_instruction"],
    )
    probe = _first_text(record, ["probe", "prompt", "task_prompt"], default=question)
    gold_answer = _first_text(record, ["gold_answer", "answer", "target", "ground_truth"])

    metadata = {
        "source_dataset": INJSQUAD_BENCHMARK_NAME,
        "has_injected_instruction": bool(injected_instruction.strip()),
    }

    return InjSquadSample(
        sample_id=_sample_id(record, idx),
        benchmark_name=INJSQUAD_BENCHMARK_NAME,
        question=question,
        clean_document=clean_document,
        injected_instruction=injected_instruction,
        probe=probe,
        gold_answer=gold_answer,
        metadata=metadata,
        raw_record=record,
    )


def load_injsquad_samples(
    *,
    repo_root: str | Path = ".",
    max_samples: Optional[int] = None,
) -> List[InjSquadSample]:
    """Load Inj-SQuAD samples deterministically from local benchmark JSON."""
    paths = get_default_paths(repo_root)
    benchmark_file = assert_mandatory_benchmark_exists(paths)

    rows = json.loads(benchmark_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {benchmark_file}")

    normalized = [_normalize_record(r, i) for i, r in enumerate(rows) if isinstance(r, dict)]
    normalized.sort(key=lambda s: s.sample_id)

    if max_samples is not None:
        if max_samples < 0:
            raise ValueError("max_samples must be >= 0")
        normalized = normalized[:max_samples]

    return normalized
