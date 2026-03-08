from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from fragweave.config import RunConfig, load_config
from fragweave.models.hf_chat import HFChat

DEFAULT_DEBUG_JSONL = "outputs/emailqa_fragweave_loc_san/debug_fragments_12.jsonl"
DEFAULT_CONFIG_PATH = "configs/emailqa_with_localization_and_sanitization.yaml"
DEFAULT_VARIANT_ID = "k3_refs1_guideA"
DEFAULT_MAX_PAIRED = 20
DEFAULT_SEED = 2026


@dataclass
class PairedSample:
    sample_id: str
    question: str
    malicious_instruction: str
    original_context: str
    baseline_context: str
    fragweave_context: str
    baseline_row: Dict[str, Any]
    fragweave_row: Dict[str, Any]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_run_config(out_dir: Path, cfg: Dict[str, Any]) -> None:
    (out_dir / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def log_line(out_dir: Path, text: str) -> None:
    with (out_dir / "run.log").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Debug jsonl file not found: {p}")
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {ln} in {p}: {exc}") from exc
    return rows


def _get_original_context(row: Dict[str, Any]) -> str:
    return str(row.get("original_context") or row.get("context") or "")


def _get_baseline_context(row: Dict[str, Any]) -> str:
    return str(row.get("baseline_context") or row.get("baseline_shadow") or row.get("context") or "")


def _get_fragweave_context(row: Dict[str, Any]) -> str:
    return str(row.get("context") or row.get("cleaned_context") or "")


def pair_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    variant_id: str,
    max_pairs: int,
) -> List[PairedSample]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for row in rows:
        if str(row.get("variant_id", "")) != variant_id:
            continue
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            continue
        slot = "baseline" if bool(row.get("is_direct_baseline")) else "fragweave"
        grouped.setdefault(sample_id, {})[slot] = row

    pairs: List[PairedSample] = []
    for sample_id in sorted(grouped.keys()):
        pair = grouped[sample_id]
        if "baseline" not in pair or "fragweave" not in pair:
            continue
        base = pair["baseline"]
        fw = pair["fragweave"]
        question = str(fw.get("question") or base.get("question") or "")
        malicious_instruction = str(
            fw.get("malicious_instruction") or base.get("malicious_instruction") or ""
        )
        original_context = _get_original_context(fw) or _get_original_context(base)
        baseline_context = _get_baseline_context(base)
        fragweave_context = _get_fragweave_context(fw)
        if not (question and original_context and baseline_context and fragweave_context):
            continue
        pairs.append(
            PairedSample(
                sample_id=sample_id,
                question=question,
                malicious_instruction=malicious_instruction,
                original_context=original_context,
                baseline_context=baseline_context,
                fragweave_context=fragweave_context,
                baseline_row=base,
                fragweave_row=fw,
            )
        )
        if len(pairs) >= max_pairs:
            break

    if not pairs:
        raise RuntimeError(
            f"No valid paired samples found for variant_id={variant_id}."
        )
    return pairs


def build_prompt(template: str, context: str, question: str) -> str:
    return template.format(context=context, question=question)


def load_analysis_stack(config_path: str | Path) -> Tuple[RunConfig, HFChat]:
    cfg = load_config(config_path)
    chat = HFChat.from_config(cfg.target_model)
    return cfg, chat


def to_numpy(v: torch.Tensor) -> np.ndarray:
    return v.detach().cpu().float().numpy()


def token_char_spans(tokenizer: Any, input_ids: List[int]) -> Tuple[str, List[Tuple[int, int]]]:
    pieces = [
        tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for tid in input_ids
    ]
    merged = "".join(pieces)
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        nxt = cursor + len(piece)
        spans.append((cursor, nxt))
        cursor = nxt
    return merged, spans


def find_sections(prompt_text: str) -> Dict[str, Tuple[int, int]]:
    sections: Dict[str, Tuple[int, int]] = {}
    q_tag = "QUESTION:"
    e_tag = "EMAIL:"
    answer_tag = "Answer:"

    e_idx = prompt_text.find(e_tag)
    q_idx = prompt_text.find(q_tag)
    a_idx = prompt_text.find(answer_tag)

    if e_idx >= 0 and q_idx > e_idx:
        sections["main_context"] = (e_idx + len(e_tag), q_idx)
    if q_idx >= 0 and a_idx > q_idx:
        sections["question"] = (q_idx + len(q_tag), a_idx)
    if a_idx >= 0:
        sections["answer_prefix"] = (a_idx, len(prompt_text))
    return sections

