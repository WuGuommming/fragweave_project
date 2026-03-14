from __future__ import annotations

from typing import Any, Dict

from .schema import InjSquadSample


def to_attack_input(sample: InjSquadSample) -> Dict[str, Any]:
    """Convert an Inj-SQuAD sample into the internal attack-input shape.

    This helper is intentionally lightweight and does not run or wire attack logic.
    """

    return {
        "uid": sample.sample_id,
        "context": sample.clean_document,
        "question": sample.question,
        "answer": sample.gold_answer,
        "benchmark_name": sample.benchmark_name,
        "probe": sample.probe,
        "injected_instruction": sample.injected_instruction,
        "metadata": sample.metadata,
        "raw": sample.raw_record,
    }
