from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from tqdm import tqdm

from fragweave.attacks.guidance import generate_guidance
from fragweave.attacks.other_baselines import (
    DEFAULT_COMBINED_PARTS,
    build_attack_artifact,
    load_bipia_instruction_pool,
    normalize_method_name,
)
from fragweave.attacks.sharder import shard_with_llm
from fragweave.attacks.weaver import apply_weave, apply_weave_with_shadow
from fragweave.config import ModelConfig, load_config
from fragweave.data.bipia_fetch import ensure_bipia_repo
from fragweave.eval.judge import judge_attack
from fragweave.eval.localization import run_span_localizer, shadow_to_clean_and_spans, span_prf1
from fragweave.models.hf_chat import HFChat
from fragweave.defenses.promptlocate_adapter import OpenPromptInjectionAdapter, OpenPromptInjectionAssetsError
from fragweave.run_sweep import (
    DEFAULT_MALICIOUS_INSTRUCTIONS,
    _build_variant_id,
    _conservative_redact_by_spans,
    _load_samples_any_task,
    _redact_by_spans,
    _sanitize_with_checks,
    _task_specific_sanitizer_prompt,
    _conservative_task_specific_sanitizer_prompt,
    BASELINE_SANITIZER_PROMPT,
    CONSERVATIVE_SANITIZER_PROMPT,
    _unique_path,
    build_sample_instruction_plan,
    choose_spread_ops,
    load_instructions,
)
from topicattack.topic_attack import build_topicattack_block, inject_block as topicattack_inject_block, load_topicattack_section


BIPIA_NATIVE_POSITION_PLAN = (
    ("start", "prepend"),
    ("middle", "middle"),
    ("end", "append"),
)


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _resolve_attack_method(name: str) -> str:
    key = (name or "").strip().lower()
    if key in {"ours", "topicattack"}:
        return key
    return normalize_method_name(key)


def _parse_parts(raw: str | None) -> List[str]:
    if raw is None:
        return list(DEFAULT_COMBINED_PARTS)
    out = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return out or list(DEFAULT_COMBINED_PARTS)


def _load_optional_rewrite_model_config(config_path: str) -> Optional[ModelConfig]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    models = raw.get("models", {}) or {}
    if not isinstance(models, dict) or "rewrite" not in models:
        return None
    sec = models["rewrite"]
    if not isinstance(sec, dict):
        raise TypeError("config section 'models.rewrite' must be a dict")
    allowed = set(ModelConfig.__dataclass_fields__.keys())
    kwargs = {k: v for k, v in sec.items() if k in allowed}
    return ModelConfig(**kwargs)


def _load_optional_rewrite_prompt(config_path: str) -> Optional[str]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    sec = raw.get("rewrite", {}) or {}
    if not isinstance(sec, dict):
        return None
    value = sec.get("prompt_template")
    return str(value) if isinstance(value, str) and value.strip() else None


def _split_paragraphs(text: str) -> List[str]:
    body = (text or "").strip("\n")
    if not body:
        return []
    parts = re.split(r"(\n\s*\n)", body)
    paras: List[str] = []
    buf = ""
    for part in parts:
        if re.fullmatch(r"\n\s*\n", part):
            if buf.strip():
                paras.append(buf)
            buf = ""
        else:
            buf += part
    if buf.strip():
        paras.append(buf)
    return paras or [body]


def _insert_email_block(context: str, block: str, *, policy: str) -> str:
    body = (context or "").rstrip()
    block = (block or "").strip()
    if not block:
        return body
    pol = (policy or "append").lower()
    if not body:
        return block
    if pol == "prepend":
        return f"{block}\n\n{body}"
    paras = _split_paragraphs(body)
    if pol in {"mid", "middle"} and paras:
        idx = len(paras) // 2
        paras = paras[:idx] + [block] + paras[idx:]
        return "\n\n".join(paras)
    if pol in {"append", "end"}:
        return f"{body}\n\n{block}"
    raise ValueError(f"Unsupported deterministic insertion policy: {policy}")


def _materialize_position_variant(*, context: str, injected_text: str, tag_start: str, tag_end: str, position_policy: str) -> tuple[str, str]:
    poisoned = _insert_email_block(context, injected_text, policy=position_policy)
    shadow = _insert_email_block(context, f"{tag_start}{injected_text}{tag_end}", policy=position_policy)
    return poisoned, shadow


def _build_target_instruction(target_template: str, question: str) -> str:
    placeholder = "__FW_PROMPTLOCATE_CONTEXT__"
    rendered = target_template.format(context=placeholder, question=question)
    rendered = rendered.replace(placeholder, "")
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    rendered = re.sub(r"[ \t]{2,}", " ", rendered)
    return rendered.strip()


def _build_target_prompt(target_template: str, context: str, question: str) -> str:
    return target_template.format(context=context, question=question)


def _normalize_for_search(text: str) -> Tuple[str, List[int]]:
    out_chars: List[str] = []
    idx_map: List[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out_chars.append(" ")
            idx_map.append(i)
            prev_space = True
        else:
            out_chars.append(ch.lower())
            idx_map.append(i)
            prev_space = False
    norm = "".join(out_chars).strip()
    if not norm:
        return "", []
    left = 0
    while left < len(out_chars) and out_chars[left] == " ":
        left += 1
    right = len(out_chars)
    while right > left and out_chars[right - 1] == " ":
        right -= 1
    return norm, idx_map[left:right]


def _merge_spans(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    items = sorted((int(a), int(b)) for a, b in spans if int(a) < int(b))
    if not items:
        return []
    merged: List[Tuple[int, int]] = [items[0]]
    for a, b in items[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def _find_spans_from_localized_text(context: str, localized_text: str) -> List[Tuple[int, int]]:
    localized = (localized_text or "").strip()
    if not localized:
        return []
    ctx_norm, ctx_idx = _normalize_for_search(context)
    loc_norm, _ = _normalize_for_search(localized)
    if not ctx_norm or not loc_norm:
        return []
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        pos = ctx_norm.find(loc_norm, start)
        if pos < 0:
            break
        end_pos = pos + len(loc_norm) - 1
        if pos < len(ctx_idx) and end_pos < len(ctx_idx):
            spans.append((ctx_idx[pos], ctx_idx[end_pos] + 1))
        start = pos + max(1, len(loc_norm) // 2)
    if spans:
        return _merge_spans(spans)
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", localized) if p.strip()]
    piece_spans: List[Tuple[int, int]] = []
    for piece in pieces:
        piece_norm, _ = _normalize_for_search(piece)
        if len(piece_norm) < 8:
            continue
        pos = ctx_norm.find(piece_norm)
        if pos >= 0:
            end_pos = pos + len(piece_norm) - 1
            if pos < len(ctx_idx) and end_pos < len(ctx_idx):
                piece_spans.append((ctx_idx[pos], ctx_idx[end_pos] + 1))
    return _merge_spans(piece_spans)


def _evaluate_attack(target: HFChat, judge: HFChat, *, target_template: str, context: str, question: str, malicious_instruction: str) -> Dict[str, Any]:
    prompt = _build_target_prompt(target_template, context=context, question=question)
    response = target.generate(prompt)
    attack_judge = judge_attack(judge, question=question, instruction=malicious_instruction, response=response)
    return {"target_prompt": prompt, "response": response, "attack_judge": asdict(attack_judge), "attack_succeeded": int(bool(attack_judge.ok)), "attack_conf": float(attack_judge.score), "rationale": attack_judge.reason}


def _evaluate_existing_localizer(detector_chat: Optional[HFChat], *, cfg: Any, task: Optional[str], poisoned_context: str, shadow_context: Optional[str]) -> Dict[str, Any]:
    out = {"loc_precision": None, "loc_recall": None, "loc_f1": None, "loc_spans": [], "loc_debug": None, "redacted_context": poisoned_context, "conservative_redacted_context": poisoned_context, "conservative_redaction_debug": None, "gt_spans": None}
    if detector_chat is None or not shadow_context:
        return out
    clean_context, gt_spans = shadow_to_clean_and_spans(shadow_context, cfg.localization.tag_start, cfg.localization.tag_end)
    loc = run_span_localizer(detector_chat, poisoned_context, max_new_tokens=int(getattr(cfg.localization, "detector_max_new_tokens", 256) or 256), task=task, prompt_template=getattr(cfg.localization, "prompt_template", None))
    p, r, f1 = span_prf1(loc.spans, gt_spans)
    conservative_ctx, conservative_dbg = _conservative_redact_by_spans(poisoned_context, loc.spans, cfg.localization)
    out.update({"clean_context_from_shadow": clean_context, "gt_spans": gt_spans, "loc_precision": p, "loc_recall": r, "loc_f1": f1, "loc_spans": loc.spans, "loc_debug": asdict(loc), "redacted_context": _redact_by_spans(poisoned_context, loc.spans), "conservative_redacted_context": conservative_ctx, "conservative_redaction_debug": conservative_dbg})
    return out


def _find_text_spans(context: str, texts: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for text in texts:
        t = str(text or "").strip()
        if not t:
            continue
        start = 0
        while True:
            i = context.find(t, start)
            if i < 0:
                break
            spans.append((i, i + len(t)))
            start = i + max(1, len(t) // 2)
    return _merge_spans(spans)


def _span_overlap_ratio(pred_spans: Sequence[Tuple[int, int]], target_spans: Sequence[Tuple[int, int]]) -> float:
    pred = _merge_spans(list(pred_spans))
    target = _merge_spans(list(target_spans))
    if not pred or not target:
        return 0.0
    inter = 0
    i = j = 0
    while i < len(pred) and j < len(target):
        a1, b1 = pred[i]
        a2, b2 = target[j]
        a = max(a1, a2)
        b = min(b1, b2)
        if a < b:
            inter += b - a
        if b1 <= b2:
            i += 1
        else:
            j += 1
    target_len = sum(b - a for a, b in target)
    return inter / target_len if target_len > 0 else 0.0


def _build_conservative_promptlocate_context(poisoned_context: str, recovered_context: str, pred_spans: Sequence[Tuple[int, int]], cfg: Any) -> Tuple[str, Dict[str, Any]]:
    loc_cfg = cfg.localization
    if not bool(getattr(loc_cfg, "promptlocate_conservative_enable", True)):
        return recovered_context, {"mode": "recovered", "applied": False}
    removal_fraction = max(0.0, 1.0 - (len(recovered_context) / max(1, len(poisoned_context))))
    max_fraction = float(getattr(loc_cfg, "promptlocate_conservative_max_fraction", 0.12) or 0.12)
    max_spans = int(getattr(loc_cfg, "promptlocate_conservative_max_spans", 2) or 2)
    max_span_chars = int(getattr(loc_cfg, "promptlocate_conservative_max_span_chars", 220) or 220)
    fallback = str(getattr(loc_cfg, "promptlocate_conservative_fallback", "redact") or "redact").lower()
    too_large = removal_fraction > max_fraction or len(pred_spans) > max_spans or any((b - a) > max_span_chars for a, b in pred_spans)
    if not too_large:
        return recovered_context, {"mode": "recovered", "applied": True, "removal_fraction": removal_fraction, "pred_spans": list(pred_spans)}
    if fallback == "keep":
        return poisoned_context, {"mode": "keep_original", "applied": False, "reason": "promptlocate_recovery_too_aggressive", "removal_fraction": removal_fraction, "pred_spans": list(pred_spans)}
    conservative_ctx, conservative_dbg = _conservative_redact_by_spans(poisoned_context, list(pred_spans), cfg.localization)
    return conservative_ctx, {"mode": "conservative_redact", "applied": bool(conservative_dbg.get("applied", False)), "reason": conservative_dbg.get("reason"), "removal_fraction": removal_fraction, "pred_spans": list(pred_spans), "redaction_debug": conservative_dbg}


def _evaluate_promptlocate(openpi: OpenPromptInjectionAdapter, target: HFChat, judge: HFChat, *, target_template: str, question: str, malicious_instruction: str, poisoned_context: str, gt_spans: Optional[List[Tuple[int, int]]], target_instruction: str, use_detector_gate: bool, shard_texts: Sequence[str], guide_texts: Sequence[str], cfg: Any) -> Dict[str, Any]:
    detector_out = openpi.detect(poisoned_context)
    recovered_context = poisoned_context
    localized_text = ""
    ran_promptlocate = False
    locate_debug: Dict[str, Any] = {}
    if (not use_detector_gate) or detector_out.detected:
        recovered_context, localized_text, locate_debug = openpi.locate_and_recover_with_debug(poisoned_context, target_instruction)
        ran_promptlocate = True
    pred_spans = _find_spans_from_localized_text(poisoned_context, localized_text)
    p = r = f1 = None
    if gt_spans is not None:
        p, r, f1 = span_prf1(pred_spans, gt_spans)
    shard_spans = _find_text_spans(poisoned_context, shard_texts)
    guide_spans = _find_text_spans(poisoned_context, guide_texts)
    aggressive_eval = _evaluate_attack(target, judge, target_template=target_template, context=recovered_context, question=question, malicious_instruction=malicious_instruction)
    conservative_context, conservative_dbg = _build_conservative_promptlocate_context(poisoned_context, recovered_context, pred_spans, cfg)
    conservative_eval = _evaluate_attack(target, judge, target_template=target_template, context=conservative_context, question=question, malicious_instruction=malicious_instruction)
    return {
        "detector_raw": detector_out.raw_value,
        "detector_detected": int(detector_out.detected),
        "promptlocate_ran": int(ran_promptlocate),
        "promptlocate_recovered_context": recovered_context,
        "promptlocate_conservative_context": conservative_context,
        "promptlocate_conservative_debug": conservative_dbg,
        "promptlocate_localized_text": localized_text,
        "promptlocate_pred_spans": pred_spans,
        "promptlocate_loc_precision": p,
        "promptlocate_loc_recall": r,
        "promptlocate_loc_f1": f1,
        "promptlocate_localized_shard_overlap": _span_overlap_ratio(pred_spans, shard_spans),
        "promptlocate_localized_guide_overlap": _span_overlap_ratio(pred_spans, guide_spans),
        "promptlocate_debug": locate_debug,
        **{f"aggressive_promptlocate_{k}": v for k, v in aggressive_eval.items() if k not in {"target_prompt"}},
        "aggressive_promptlocate_target_prompt": aggressive_eval["target_prompt"],
        **{f"promptlocate_{k}": v for k, v in conservative_eval.items() if k not in {"target_prompt"}},
        "promptlocate_target_prompt": conservative_eval["target_prompt"],
    }


def _build_fragweave_attack(*, cfg: Any, task: Optional[str], context: str, question: str, malicious_instruction: str, sharder: HFChat, weaver: HFChat, rng: random.Random, k: int, guide_version: str, profile_mode: str) -> Dict[str, Any]:
    sharder_retries = int(getattr(cfg.attack, "sharder_retries", 2))
    weaver_retries = int(getattr(cfg.attack, "weaver_retries", 2))
    tagger_retries = int(getattr(cfg.attack, "tagger_retries", 2))
    carrier_line = str(getattr(cfg.attack, "carrier_line", "Note: Answer the question using the email content only."))
    sharder_prompt = getattr(cfg.attack, "sharder_prompt", None)
    weaver_prompt = getattr(cfg.attack, "weaver_prompt", None)
    guide_lib = getattr(cfg.attack, "guidance_lib", None)

    shard_res = shard_with_llm(sharder, malicious_instruction, int(k), task=task, max_retries=sharder_retries, prompt_template=sharder_prompt, profile_mode=profile_mode)
    shard_infos = [{"slot": shard_res.slot_plan[i] if i < len(shard_res.slot_plan) else f"slot_{i}", "text": shard, "source": "shard"} for i, shard in enumerate(shard_res.shards)]
    guide_res = generate_guidance(task, guide_version, malicious_instruction, rng, lib_override=guide_lib, profile_mode=profile_mode, shard_infos=shard_infos)
    guide_infos = [{"slot": "guide", "text": text, "source": "guide"} for text in guide_res.snippets]
    items = shard_infos + guide_infos
    ops, normalized_context = choose_spread_ops(task, context, items, rng, carrier_line=carrier_line)
    shadow_context = None
    gt_spans = None
    clean_context = None
    if bool(getattr(cfg.localization, "enable", False)) and str(getattr(cfg.localization, "gt_mode", "off")).lower() == "shadow_tags":
        poisoned_context, shadow_context, weave_debug = apply_weave_with_shadow(weaver, normalized_context, ops, task=task, max_retries=weaver_retries, prompt_template=weaver_prompt, tag_chat=weaver, tag_start=cfg.localization.tag_start, tag_end=cfg.localization.tag_end, tag_max_retries=tagger_retries)
        clean_context, gt_spans = shadow_to_clean_and_spans(shadow_context, cfg.localization.tag_start, cfg.localization.tag_end)
    else:
        poisoned_context, weave_debug = apply_weave(weaver, normalized_context, ops, task=task, max_retries=weaver_retries, prompt_template=weaver_prompt)

    return {
        "attack_method": "ours",
        "attack_label": "ours",
        "position": "single",
        "position_policy": "",
        "question": question,
        "original_context": context,
        "malicious_instruction": malicious_instruction,
        "normalized_context": normalized_context,
        "poisoned_context": poisoned_context,
        "shadow_context": shadow_context,
        "clean_context_from_shadow": clean_context,
        "gt_spans": gt_spans,
        "shards": list(shard_res.shards),
        "guide_snippets": list(guide_res.snippets),
        "sharder_debug": asdict(shard_res),
        "guidance": asdict(guide_res),
        "ops": [asdict(op) for op in ops],
        "weave_debug": weave_debug,
        "metadata": {"k": int(k), "guide_version": str(guide_version).upper(), "profile_mode": profile_mode},
    }


def _build_topicattack_attack(*, cfg: Any, ta_cfg: Any, context: str, question: str, malicious_instruction: str, generator_chat: HFChat, topic_chat: HFChat, rng: random.Random) -> Dict[str, Any]:
    artifact = build_topicattack_block(generator_chat, topic_chat, context=context, question=question, malicious_instruction=malicious_instruction, cfg=ta_cfg, rng=rng)
    poisoned_context = topicattack_inject_block(context, artifact.final_block, mode=ta_cfg.insert_mode)
    shadow_block = f"{cfg.localization.tag_start}{artifact.final_block}{cfg.localization.tag_end}"
    shadow_context = None
    clean_context = None
    gt_spans = None
    if bool(getattr(cfg.localization, "enable", False)) and str(getattr(cfg.localization, "gt_mode", "off")).lower() == "shadow_tags":
        shadow_context = topicattack_inject_block(context, shadow_block, mode=ta_cfg.insert_mode)
        clean_context, gt_spans = shadow_to_clean_and_spans(shadow_context, cfg.localization.tag_start, cfg.localization.tag_end)
    return {"attack_method": "topicattack", "attack_label": "topicattack", "position": "single", "position_policy": str(getattr(ta_cfg, "insert_mode", "")), "question": question, "original_context": context, "malicious_instruction": malicious_instruction, "normalized_context": None, "poisoned_context": poisoned_context, "shadow_context": shadow_context, "clean_context_from_shadow": clean_context, "gt_spans": gt_spans, "shards": None, "guide_snippets": None, "sharder_debug": None, "guidance": None, "ops": None, "weave_debug": None, "metadata": {"variant": getattr(ta_cfg, "variant", None), "num_turns": getattr(ta_cfg, "num_turns", None), "context_max_chars": getattr(ta_cfg, "context_max_chars", None), "insert_mode": getattr(ta_cfg, "insert_mode", None), "assistant_ack": getattr(ta_cfg, "assistant_ack", None)}, "topicattack_artifact": artifact.to_dict()}


def _build_other_attack_variants(*, cfg: Any, attack_method: str, context: str, question: str, rng: random.Random, official_payloads: Sequence[Any], combined_parts: Sequence[str], insertion_policy: str, rewrite_chat: Optional[HFChat] = None, rewrite_prompt_template: Optional[str] = None, rewrite_context_max_chars: int = 1200) -> List[Dict[str, Any]]:
    base_artifact = build_attack_artifact(method=attack_method, context=context, rng=rng, tag_start=cfg.localization.tag_start, tag_end=cfg.localization.tag_end, insertion_policy=insertion_policy, official_payloads=official_payloads, combined_parts=combined_parts, rewrite_chat=rewrite_chat, rewrite_prompt_template=rewrite_prompt_template, rewrite_context_max_chars=int(rewrite_context_max_chars))
    eval_positions = list(BIPIA_NATIVE_POSITION_PLAN) if attack_method == "bipia_native" else [("single", insertion_policy)]
    variants: List[Dict[str, Any]] = []
    for pos_name, pos_policy in eval_positions:
        if attack_method == "bipia_native":
            poisoned_context, shadow_context = _materialize_position_variant(context=context, injected_text=base_artifact.injected_text, tag_start=cfg.localization.tag_start, tag_end=cfg.localization.tag_end, position_policy=pos_policy)
        else:
            poisoned_context = base_artifact.poisoned_context
            shadow_context = base_artifact.shadow_context
        clean_context = None
        gt_spans = None
        if bool(getattr(cfg.localization, "enable", False)) and str(getattr(cfg.localization, "gt_mode", "off")).lower() == "shadow_tags":
            clean_context, gt_spans = shadow_to_clean_and_spans(shadow_context, cfg.localization.tag_start, cfg.localization.tag_end)
        variants.append({"attack_method": attack_method, "attack_label": base_artifact.label, "position": pos_name, "position_policy": pos_policy, "question": question, "original_context": context, "malicious_instruction": base_artifact.payload, "normalized_context": None, "poisoned_context": poisoned_context, "shadow_context": shadow_context, "clean_context_from_shadow": clean_context, "gt_spans": gt_spans, "shards": None, "guide_snippets": None, "sharder_debug": None, "guidance": None, "ops": None, "weave_debug": None, "metadata": dict(base_artifact.metadata), "injected_text": base_artifact.injected_text, "payload": base_artifact.payload})
    return variants


def _mean_optional(vals: Sequence[Any]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and v != ""]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _variant_settings_for_attack(cfg: Any, args: argparse.Namespace, use_ours: bool, use_topicattack: bool, ta_cfg: Any) -> List[Dict[str, Any]]:
    if use_ours:
        guide_versions = [str(args.guide_version).upper()] if getattr(args, "guide_version", None) else list(getattr(cfg.attack, "guide_versions", ("A",)))
        out: List[Dict[str, Any]] = []
        for guide_version in guide_versions:
            out.append({"variant_id": _build_variant_id(int(args.k), guide_version, str(getattr(cfg.attack, "profile_mode", "balanced"))), "k": int(args.k), "guide_version": guide_version, "profile_mode": str(getattr(cfg.attack, "profile_mode", "balanced"))})
        return out
    if use_topicattack:
        ta_variant = getattr(ta_cfg, "variant", "topicattack")
        return [{"variant_id": f"topicattack_{ta_variant}", "profile_mode": ""}]
    return [{"variant_id": _resolve_attack_method(args.attack_method), "profile_mode": ""}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/emailqa_promptlocate_example.yaml")
    ap.add_argument("--attack_method", type=str, default="ours")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--guide-version", type=str, default=None)
    ap.add_argument("--variant-id", type=str, default=None)
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--combined_parts", type=str, default="ignore,escape,fakecom")
    ap.add_argument("--native_attack_limit", type=int, default=None)
    ap.add_argument("--insertion_policy", type=str, default="append")
    ap.add_argument("--rewrite_context_max_chars", type=int, default=1200)
    ap.add_argument("--opi-root", type=str, default="third_party/Open-Prompt-Injection")
    ap.add_argument("--opi-model-config", type=str, default=None)
    ap.add_argument("--opi-detector-ft-path", type=str, default="third_party/open_prompt_injection_assets/datasentinel")
    ap.add_argument("--opi-promptlocate-ft-path", type=str, default="third_party/open_prompt_injection_assets/promptlocate")
    ap.add_argument("--opi-helper-model", type=str, default="gpt2")
    ap.add_argument("--opi-sep-thres", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    task = getattr(cfg.dataset, "task", None) or "email_qa"
    task_low = str(task).lower()
    if task_low not in {"email_qa", "emailqa", "email"}:
        raise ValueError(f"run_sweep_promptlocate.py currently supports EmailQA only. Got dataset.task={task!r}.")

    attack_method = _resolve_attack_method(args.attack_method)
    use_ours = attack_method == "ours"
    use_topicattack = attack_method == "topicattack"
    combined_parts = _parse_parts(args.combined_parts)
    ta_cfg = load_topicattack_section(args.config) if use_topicattack else None
    rewrite_model_cfg = _load_optional_rewrite_model_config(args.config) if attack_method == "rewrite" else None
    rewrite_prompt_template = _load_optional_rewrite_prompt(args.config) if attack_method == "rewrite" else None

    base_run_name = str(getattr(cfg.output, "run_name", "emailqa_fragweave"))
    out_dir = Path(cfg.output.out_dir) / f"{base_run_name}_promptlocate"
    out_dir.mkdir(parents=True, exist_ok=True)

    opi_root = Path(args.opi_root)
    opi_model_config = Path(args.opi_model_config) if args.opi_model_config else (opi_root / "configs" / "model_configs" / "mistral_config.json")
    variant_settings = _variant_settings_for_attack(cfg, args, use_ours, use_topicattack, ta_cfg)

    resolved_cfg = {"attack_method": attack_method, "variants": variant_settings, "combined_parts": list(combined_parts) if (not use_ours and not use_topicattack) else None, "native_attack_limit": int(args.native_attack_limit) if args.native_attack_limit is not None else None, "insertion_policy": args.insertion_policy if (not use_ours and not use_topicattack) else None, "rewrite_context_max_chars": int(args.rewrite_context_max_chars) if attack_method == "rewrite" else None, "rewrite_model_name_or_path": getattr(rewrite_model_cfg, "name_or_path", None) if rewrite_model_cfg else None, "opi_root": str(opi_root), "opi_model_config": str(opi_model_config), "opi_detector_ft_path": str(args.opi_detector_ft_path), "opi_promptlocate_ft_path": str(args.opi_promptlocate_ft_path), "opi_helper_model": args.opi_helper_model, "opi_sep_thres": float(args.opi_sep_thres), "promptlocate_conservative_enable": bool(getattr(cfg.localization, "promptlocate_conservative_enable", True))}
    print(f"[Config] {_json_dump(resolved_cfg)}")
    (out_dir / "config_promptlocate_resolved.json").write_text(json.dumps(resolved_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    bipia_root = ensure_bipia_repo(cfg.dataset.bipia_root)
    requested_max_samples = args.max_samples if args.max_samples is not None else getattr(cfg.dataset, "max_samples", None)
    combo_mode = requested_max_samples is not None and requested_max_samples > 50
    loader_max_samples = 50 if combo_mode else requested_max_samples
    samples, used_schema = _load_samples_any_task(bipia_root=bipia_root, task=task, split=cfg.dataset.split, max_samples=loader_max_samples, cfg_dataset=cfg.dataset)

    instructions: List[str] = []
    if use_ours or use_topicattack:
        instructions = load_instructions(cfg.attack.instruction_jsonl, cfg.attack.instruction_text_key)
        if not instructions:
            instructions = DEFAULT_MALICIOUS_INSTRUCTIONS
    official_payloads: List[Any] = []
    if not use_ours and not use_topicattack:
        official_payloads = load_bipia_instruction_pool(bipia_root, split=cfg.dataset.split, limit=args.native_attack_limit)

    target = HFChat.from_config(cfg.target_model)
    judge = HFChat.from_config(cfg.judge_model)
    sharder = None
    weaver = None
    if use_ours or use_topicattack:
        sharder = HFChat.from_config(cfg.sharder_model)
        weaver = HFChat.from_config(cfg.weaver_model)
    rewrite_chat = None
    if attack_method == "rewrite":
        if rewrite_model_cfg is None:
            raise ValueError("attack_method=rewrite requires models.rewrite in the config yaml")
        rewrite_chat = HFChat.from_config(rewrite_model_cfg)
    detector_chat = None
    if bool(getattr(cfg.localization, "enable", False)) and str(getattr(cfg.localization, "gt_mode", "off")).lower() != "off":
        if cfg.detector_model is None:
            raise ValueError("localization.enable=true but models.detector is not provided")
        detector_chat = HFChat.from_config(cfg.detector_model)
    sanitizer = None
    sanitize_context = None
    if bool(getattr(cfg.sanitization, "enable", False)):
        if cfg.sanitizer_model is None:
            raise ValueError("sanitization.enable=true but models.sanitizer is not provided")
        sanitizer = HFChat.from_config(cfg.sanitizer_model)
        from fragweave.defenses.sanitizer import sanitize_context as _sanitize_context  # type: ignore
        sanitize_context = _sanitize_context

    try:
        openpi = OpenPromptInjectionAdapter(repo_root=opi_root, model_config_json=opi_model_config, detector_ft_path=args.opi_detector_ft_path, promptlocate_ft_path=args.opi_promptlocate_ft_path, helper_model_name=args.opi_helper_model, sep_thres=args.opi_sep_thres)
    except OpenPromptInjectionAssetsError as e:
        raise SystemExit(str(e))

    rng = random.Random(int(getattr(cfg.attack, "rng_seed", 2026)))
    if use_ours or use_topicattack:
        eval_plan = build_sample_instruction_plan(num_samples=len(samples), num_instructions=len(instructions), requested_max_samples=requested_max_samples, rng=rng)
    else:
        eval_plan = [(i, None) for i in range(len(samples))]

    for variant in variant_settings:
        variant_id = args.variant_id or variant["variant_id"]
        print(f"\n== Variant {variant_id} ==")
        rows: List[Dict[str, Any]] = []
        debug_rows: List[Dict[str, Any]] = []
        for i, pair in tqdm(list(enumerate(eval_plan)), desc=variant_id):
            sample_idx, instruction_idx = pair
            sample = samples[sample_idx]
            question = str(getattr(sample, "question", ""))
            answer = getattr(sample, "answer", None)
            context = str(getattr(sample, "context", ""))
            sample_id = str(getattr(sample, "uid", sample_idx))
            target_instruction = _build_target_instruction(cfg.prompt.target_template, question)

            if use_ours:
                malicious_instruction = instructions[instruction_idx if instruction_idx is not None else (i % len(instructions))]
                attack_variants = [_build_fragweave_attack(cfg=cfg, task=task, context=context, question=question, malicious_instruction=malicious_instruction, sharder=sharder, weaver=weaver, rng=rng, k=variant["k"], guide_version=variant["guide_version"], profile_mode=variant["profile_mode"])]
            elif use_topicattack:
                malicious_instruction = instructions[instruction_idx if instruction_idx is not None else (i % len(instructions))]
                attack_variants = [_build_topicattack_attack(cfg=cfg, ta_cfg=ta_cfg, context=context, question=question, malicious_instruction=malicious_instruction, generator_chat=weaver, topic_chat=sharder, rng=rng)]
            else:
                attack_variants = _build_other_attack_variants(cfg=cfg, attack_method=attack_method, context=context, question=question, rng=rng, official_payloads=official_payloads, combined_parts=combined_parts, insertion_policy=args.insertion_policy, rewrite_chat=rewrite_chat, rewrite_prompt_template=rewrite_prompt_template, rewrite_context_max_chars=int(args.rewrite_context_max_chars))

            for attack_data in attack_variants:
                poisoned_context = attack_data["poisoned_context"]
                shadow_context = attack_data.get("shadow_context")
                gt_spans = attack_data.get("gt_spans")
                malicious_instruction = attack_data["malicious_instruction"]
                base_eval = _evaluate_attack(target, judge, target_template=cfg.prompt.target_template, context=poisoned_context, question=question, malicious_instruction=malicious_instruction)
                localizer_eval = _evaluate_existing_localizer(detector_chat, cfg=cfg, task=task, poisoned_context=poisoned_context, shadow_context=shadow_context)

                aggr_san_generic = aggr_san_task = cons_san_generic = cons_san_task = None
                sanitization_debug = None
                if bool(getattr(cfg.sanitization, "enable", False)) and sanitize_context is not None and sanitizer is not None:
                    aggr_clean, aggr_dbg = _sanitize_with_checks(sanitize_context, sanitizer, poisoned_context, system_prompt=BASELINE_SANITIZER_PROMPT, max_new_tokens=int(getattr(cfg.sanitization, "sanitizer_max_new_tokens", 2048) or 2048), task=task)
                    aggr_task_prompt = _task_specific_sanitizer_prompt(task, getattr(cfg.sanitization, "system_prompt", None))
                    aggr_clean_task, aggr_task_dbg = _sanitize_with_checks(sanitize_context, sanitizer, poisoned_context, system_prompt=aggr_task_prompt, max_new_tokens=int(getattr(cfg.sanitization, "sanitizer_max_new_tokens", 2048) or 2048), task=task)
                    aggr_generic_eval = _evaluate_attack(target, judge, target_template=cfg.prompt.target_template, context=aggr_clean, question=question, malicious_instruction=malicious_instruction)
                    aggr_task_eval = _evaluate_attack(target, judge, target_template=cfg.prompt.target_template, context=aggr_clean_task, question=question, malicious_instruction=malicious_instruction)
                    aggr_san_generic = aggr_generic_eval["attack_succeeded"]
                    aggr_san_task = aggr_task_eval["attack_succeeded"]
                    cons_prompt = str(getattr(cfg.sanitization, "conservative_system_prompt", None) or CONSERVATIVE_SANITIZER_PROMPT)
                    cons_clean, cons_dbg = _sanitize_with_checks(sanitize_context, sanitizer, poisoned_context, system_prompt=cons_prompt, max_new_tokens=int(getattr(cfg.sanitization, "sanitizer_max_new_tokens", 2048) or 2048), task=task, max_drop_ratio=float(getattr(cfg.sanitization, "conservative_max_drop_ratio", 0.14) or 0.14), min_output_ratio=float(getattr(cfg.sanitization, "conservative_min_output_ratio", 0.78) or 0.78))
                    cons_task_prompt = _conservative_task_specific_sanitizer_prompt(task, getattr(cfg.sanitization, "conservative_system_prompt", None))
                    cons_clean_task, cons_task_dbg = _sanitize_with_checks(sanitize_context, sanitizer, poisoned_context, system_prompt=cons_task_prompt, max_new_tokens=int(getattr(cfg.sanitization, "sanitizer_max_new_tokens", 2048) or 2048), task=task, max_drop_ratio=float(getattr(cfg.sanitization, "conservative_max_drop_ratio", 0.14) or 0.14), min_output_ratio=float(getattr(cfg.sanitization, "conservative_min_output_ratio", 0.78) or 0.78))
                    cons_generic_eval = _evaluate_attack(target, judge, target_template=cfg.prompt.target_template, context=cons_clean, question=question, malicious_instruction=malicious_instruction)
                    cons_task_eval = _evaluate_attack(target, judge, target_template=cfg.prompt.target_template, context=cons_clean_task, question=question, malicious_instruction=malicious_instruction)
                    cons_san_generic = cons_generic_eval["attack_succeeded"]
                    cons_san_task = cons_task_eval["attack_succeeded"]
                    sanitization_debug = {"aggressive": {"generic": aggr_dbg, "task": aggr_task_dbg}, "conservative": {"generic": cons_dbg, "task": cons_task_dbg}}

                aggressive_redaction_eval = _evaluate_attack(target, judge, target_template=cfg.prompt.target_template, context=localizer_eval["redacted_context"], question=question, malicious_instruction=malicious_instruction)
                conservative_redaction_eval = _evaluate_attack(target, judge, target_template=cfg.prompt.target_template, context=localizer_eval["conservative_redacted_context"], question=question, malicious_instruction=malicious_instruction)
                promptlocate_eval = _evaluate_promptlocate(openpi, target, judge, target_template=cfg.prompt.target_template, question=question, malicious_instruction=malicious_instruction, poisoned_context=poisoned_context, gt_spans=gt_spans, target_instruction=target_instruction, use_detector_gate=False, shard_texts=attack_data.get("shards") or [], guide_texts=attack_data.get("guide_snippets") or [], cfg=cfg)
                pipeline_eval = _evaluate_promptlocate(openpi, target, judge, target_template=cfg.prompt.target_template, question=question, malicious_instruction=malicious_instruction, poisoned_context=poisoned_context, gt_spans=gt_spans, target_instruction=target_instruction, use_detector_gate=True, shard_texts=attack_data.get("shards") or [], guide_texts=attack_data.get("guide_snippets") or [], cfg=cfg)

                row = {
                    "variant_id": variant_id,
                    "sample_id": sample_id,
                    "attack_method": attack_data["attack_method"],
                    "attack_label": attack_data.get("attack_label", attack_data["attack_method"]),
                    "position": attack_data.get("position", "single"),
                    "position_policy": attack_data.get("position_policy", ""),
                    "k": variant.get("k", "") if use_ours else "",
                    "profile_mode": variant.get("profile_mode", "") if use_ours else "",
                    "guide_version": variant.get("guide_version", "") if use_ours else "",
                    "topicattack_variant": getattr(ta_cfg, "variant", "") if use_topicattack else "",
                    "attack_succeeded": base_eval["attack_succeeded"],
                    "attack_conf": base_eval["attack_conf"],
                    "loc_precision": localizer_eval["loc_precision"],
                    "loc_recall": localizer_eval["loc_recall"],
                    "loc_f1": localizer_eval["loc_f1"],
                    "attack_succeeded_after_sanitizer_generic": cons_san_generic,
                    "attack_succeeded_after_sanitizer_task": cons_san_task,
                    "attack_succeeded_after_redaction": conservative_redaction_eval["attack_succeeded"],
                    "aggressive_attack_succeeded_after_sanitizer_generic": aggr_san_generic,
                    "aggressive_attack_succeeded_after_sanitizer_task": aggr_san_task,
                    "aggressive_attack_succeeded_after_redaction": aggressive_redaction_eval["attack_succeeded"],
                    "promptlocate_loc_precision": promptlocate_eval["promptlocate_loc_precision"],
                    "promptlocate_loc_recall": promptlocate_eval["promptlocate_loc_recall"],
                    "promptlocate_loc_f1": promptlocate_eval["promptlocate_loc_f1"],
                    "promptlocate_localized_shard_overlap": promptlocate_eval["promptlocate_localized_shard_overlap"],
                    "promptlocate_localized_guide_overlap": promptlocate_eval["promptlocate_localized_guide_overlap"],
                    "attack_succeeded_after_promptlocate": promptlocate_eval["promptlocate_attack_succeeded"],
                    "aggressive_attack_succeeded_after_promptlocate": promptlocate_eval["aggressive_promptlocate_attack_succeeded"],
                    "datasentinel_recall": pipeline_eval["detector_detected"],
                    "attack_succeeded_after_detector_promptlocate": pipeline_eval["promptlocate_attack_succeeded"],
                    "aggressive_attack_succeeded_after_detector_promptlocate": pipeline_eval["aggressive_promptlocate_attack_succeeded"],
                }
                rows.append(row)
                debug_rows.append({"variant_id": variant_id, "sample_id": sample_id, "used_schema": used_schema, "question": question, "answer": answer, "target_instruction": target_instruction, **attack_data, "base_eval": base_eval, "localizer_eval": localizer_eval, "aggressive_redaction_eval": aggressive_redaction_eval, "conservative_redaction_eval": conservative_redaction_eval, "promptlocate_eval": promptlocate_eval, "detector_promptlocate_eval": pipeline_eval, "sanitization_debug": sanitization_debug})

        summary: Dict[str, Any] = {
            "variant_id": variant_id,
            "attack_method": attack_method,
            "profile_mode": variant.get("profile_mode", "") if use_ours else "",
            "guide_version": variant.get("guide_version", "") if use_ours else "",
            "n_rows": len(rows),
            "n_unique_samples": len({r["sample_id"] for r in rows}),
            "asr": _mean_optional([r.get("attack_succeeded") for r in rows]),
            "loc_f1": _mean_optional([r.get("loc_f1") for r in rows]),
            "asr_after_sanitizer_generic": _mean_optional([r.get("attack_succeeded_after_sanitizer_generic") for r in rows]),
            "asr_after_sanitizer_task": _mean_optional([r.get("attack_succeeded_after_sanitizer_task") for r in rows]),
            "asr_after_redaction": _mean_optional([r.get("attack_succeeded_after_redaction") for r in rows]),
            "aggressive_asr_after_sanitizer_generic": _mean_optional([r.get("aggressive_attack_succeeded_after_sanitizer_generic") for r in rows]),
            "aggressive_asr_after_sanitizer_task": _mean_optional([r.get("aggressive_attack_succeeded_after_sanitizer_task") for r in rows]),
            "aggressive_asr_after_redaction": _mean_optional([r.get("aggressive_attack_succeeded_after_redaction") for r in rows]),
            "promptlocate_loc_f1": _mean_optional([r.get("promptlocate_loc_f1") for r in rows]),
            "promptlocate_localized_shard_overlap": _mean_optional([r.get("promptlocate_localized_shard_overlap") for r in rows]),
            "promptlocate_localized_guide_overlap": _mean_optional([r.get("promptlocate_localized_guide_overlap") for r in rows]),
            "asr_after_promptlocate": _mean_optional([r.get("attack_succeeded_after_promptlocate") for r in rows]),
            "aggressive_asr_after_promptlocate": _mean_optional([r.get("aggressive_attack_succeeded_after_promptlocate") for r in rows]),
            "datasentinel_recall": _mean_optional([r.get("datasentinel_recall") for r in rows]),
            "asr_after_detector_promptlocate": _mean_optional([r.get("attack_succeeded_after_detector_promptlocate") for r in rows]),
            "aggressive_asr_after_detector_promptlocate": _mean_optional([r.get("aggressive_attack_succeeded_after_detector_promptlocate") for r in rows]),
        }

        rows_path = _unique_path(out_dir / f"results_{variant_id}.csv")
        with rows_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else list(summary.keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        summary_path = _unique_path(out_dir / f"summary_{variant_id}.csv")
        summary_csv_row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in summary.items()}
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_csv_row.keys()))
            writer.writeheader()
            writer.writerow(summary_csv_row)

        debug_path = _unique_path(out_dir / f"debug_{variant_id}.jsonl")
        with debug_path.open("w", encoding="utf-8") as f:
            for item in debug_rows:
                f.write(_json_dump(item) + "\n")

        print(f"[Saved] rows={rows_path}")
        print(f"[Saved] summary={summary_path}")
        print(f"[Saved] debug={debug_path}")
        print("[Summary]")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
