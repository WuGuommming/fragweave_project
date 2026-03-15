from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .paths import INJSQUAD_BENCHMARK_NAME, assert_mandatory_benchmark_exists, get_default_paths
from .schema import InjSquadSample


FieldValue = Tuple[str, str]


def _first_text_with_key(record: Dict[str, Any], keys: Iterable[str], default: str = "") -> FieldValue:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                return key, text
        elif isinstance(value, list):
            if value:
                return key, "\n".join(str(x) for x in value)
        elif isinstance(value, dict):
            if value:
                return key, json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            text = str(value).strip()
            if text:
                return key, text
    return "", default


def _sample_id(record: Dict[str, Any], idx: int) -> str:
    for key in ("sample_id", "id", "uid", "example_id", "idx"):
        if key in record and record[key] is not None:
            return str(record[key])
    return str(idx)


def _require_non_empty(field_name: str, field_value: str, sample_id: str, field_key: str) -> None:
    if field_value.strip():
        return
    key_msg = f" (resolved key: {field_key})" if field_key else ""
    raise ValueError(
        f"Inj-SQuAD sample is missing required field '{field_name}'{key_msg}; sample_id={sample_id}."
    )


def _normalize_record(record: Dict[str, Any], idx: int) -> InjSquadSample:
    sample_id = _sample_id(record, idx)

    question_key, question = _first_text_with_key(
        record,
        [
            "question",
            "query",
            "q",
            "instruction",  # official Inj-SQuAD task question field
        ],
    )
    clean_document_key, clean_document = _first_text_with_key(
        record,
        [
            "clean_document",
            "context",
            "clean_context",
            "document",
            "paragraph",
            "article",
            "input",  # official Inj-SQuAD clean context field
        ],
    )
    injected_instruction_key, injected_instruction = _first_text_with_key(
        record,
        [
            "injected_instruction",
            "injection",  # official Inj-SQuAD malicious instruction field
            "attack_instruction",
            "malicious_instruction",
            "attack",
        ],
    )
    probe_keys = [
        "injection_output",  # official Inj-SQuAD field
        "probe",  # internal migrated alias
        "expected_probe",  # explicit expected trigger string
        "success_signal",  # generic alias for attack-success probe
    ]
    probe_key, probe = _first_text_with_key(record, probe_keys)
    gold_answer_key, gold_answer = _first_text_with_key(
        record,
        [
            "gold_answer",
            "answer",
            "target",
            "ground_truth",
            "output",  # official Inj-SQuAD gold answer field
        ],
    )

    _require_non_empty("question", question, sample_id, question_key)
    _require_non_empty("clean_document", clean_document, sample_id, clean_document_key)
    _require_non_empty("injected_instruction", injected_instruction, sample_id, injected_instruction_key)
    _require_non_empty("probe", probe, sample_id, probe_key)

    metadata = {
        "source_dataset": INJSQUAD_BENCHMARK_NAME,
        "has_injected_instruction": bool(injected_instruction.strip()),
        "probe_field_candidates": probe_keys,
        "resolved_fields": {
            "question": question_key,
            "clean_document": clean_document_key,
            "injected_instruction": injected_instruction_key,
            "probe": probe_key,
            "gold_answer": gold_answer_key,
        },
    }

    return InjSquadSample(
        sample_id=sample_id,
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
