from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class InjSquadSample:
    """Normalized Inj-SQuAD sample used by local loading and conversion helpers."""

    sample_id: str
    benchmark_name: str
    question: str
    clean_document: str
    injected_instruction: str
    probe: str
    gold_answer: str
    metadata: Dict[str, Any]
    raw_record: Dict[str, Any]
