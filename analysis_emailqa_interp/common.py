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


@dataclass
class PairingDiagnostics:
    variant_counts: Dict[str, int]
    rows_seen_for_variant: int
    grouped_sample_ids: int
    complete_pairs: int
    used_pairs: int
    skipped_missing_fields: int

    def short_variant_hint(self, top_k: int = 8) -> str:
        if not self.variant_counts:
            return "no variants found in debug file"
        top = sorted(self.variant_counts.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        return ", ".join(f"{k}({v})" for k, v in top)


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
    return str(
        row.get("poisoned_context")
        or row.get("shadow_context")
        or row.get("context")
        or row.get("cleaned_context")
        or ""
    )


def pair_rows_with_diagnostics(
    rows: Iterable[Dict[str, Any]],
    *,
    variant_id: str,
    max_pairs: int,
) -> Tuple[List[PairedSample], PairingDiagnostics]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    variant_counts: Dict[str, int] = {}
    rows_seen_for_variant = 0

    for row in rows:
        vid = str(row.get("variant_id", ""))
        if vid:
            variant_counts[vid] = variant_counts.get(vid, 0) + 1
        if vid != variant_id:
            continue

        rows_seen_for_variant += 1
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            continue
        slot = "baseline" if bool(row.get("is_direct_baseline")) else "fragweave"
        grouped.setdefault(sample_id, {})[slot] = row

    grouped_sample_ids = len(grouped)
    complete_pairs = sum(1 for pair in grouped.values() if "baseline" in pair and "fragweave" in pair)
    skipped_missing_fields = 0

    pairs: List[PairedSample] = []
    for sample_id in sorted(grouped.keys()):
        pair = grouped[sample_id]
        if "baseline" not in pair or "fragweave" not in pair:
            continue
        base = pair["baseline"]
        fw = pair["fragweave"]
        question = str(fw.get("question") or base.get("question") or "")
        malicious_instruction = str(fw.get("malicious_instruction") or base.get("malicious_instruction") or "")
        original_context = _get_original_context(fw) or _get_original_context(base)
        baseline_context = _get_baseline_context(base)
        fragweave_context = _get_fragweave_context(fw)
        if not (question and original_context and baseline_context and fragweave_context):
            skipped_missing_fields += 1
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

    diag = PairingDiagnostics(
        variant_counts=variant_counts,
        rows_seen_for_variant=rows_seen_for_variant,
        grouped_sample_ids=grouped_sample_ids,
        complete_pairs=complete_pairs,
        used_pairs=len(pairs),
        skipped_missing_fields=skipped_missing_fields,
    )
    return pairs, diag


def pair_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    variant_id: str,
    max_pairs: int,
) -> List[PairedSample]:
    pairs, diag = pair_rows_with_diagnostics(rows, variant_id=variant_id, max_pairs=max_pairs)
    if not pairs:
        raise RuntimeError(
            "No valid paired samples found. "
            f"variant_id={variant_id}, rows_seen_for_variant={diag.rows_seen_for_variant}, "
            f"grouped_sample_ids={diag.grouped_sample_ids}, complete_pairs={diag.complete_pairs}, "
            f"skipped_missing_fields={diag.skipped_missing_fields}. "
            f"Available variants (top): {diag.short_variant_hint()}"
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


def token_char_spans(tokenizer: Any, input_ids: List[int]) -> Tuple[str, List[Tuple[int, int]], str]:
    """Return decoded prompt text and per-token character spans.

    Alignment mode is "offset_mapping" when exact offsets are available from a fast tokenizer,
    otherwise "approximate" using conservative token-string matching against decoded text.
    """
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    if getattr(tokenizer, "is_fast", False):
        try:
            enc = tokenizer(
                decoded_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = enc.get("offset_mapping")
            if offsets and isinstance(offsets, list):
                spans = [(int(s), int(e)) for s, e in offsets]
                if len(spans) == len(input_ids):
                    return decoded_text, spans, "offset_mapping"
        except Exception:
            pass

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    spans: List[Tuple[int, int]] = []
    cursor = 0
    text_low = decoded_text.lower()
    for tok in tokens:
        norm = str(tok).replace("▁", " ").replace("Ġ", " ").replace("##", "")
        norm = norm.replace("Ċ", "\n")
        if not norm:
            spans.append((cursor, cursor))
            continue

        idx = text_low.find(norm.lower(), cursor)
        if idx < 0:
            nxt = min(len(decoded_text), cursor + max(1, len(norm)))
            spans.append((cursor, nxt))
            cursor = nxt
        else:
            end = min(len(decoded_text), idx + len(norm))
            spans.append((idx, end))
            cursor = end

    if len(spans) != len(input_ids):
        spans = spans[: len(input_ids)] + [(cursor, cursor)] * max(0, len(input_ids) - len(spans))
    return decoded_text, spans, "approximate"


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
