from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from fragweave.attacks.other_baselines import (
    DEFAULT_COMBINED_PARTS,
    build_attack_artifact,
    load_bipia_instruction_pool,
    normalize_method_name,
)
from fragweave.config import load_config
from fragweave.data.bipia_fetch import ensure_bipia_repo
from fragweave.data.emailqa_loader import load_emailqa_samples
from fragweave.eval.judge import judge_attack
from fragweave.eval.localization import shadow_to_clean_and_spans, run_span_localizer, span_prf1
from fragweave.models.hf_chat import HFChat
from fragweave.run_sweep import (
    GENERIC_SANITIZER_PROMPT,
    _normalize_sanitized_text,
    _redact_by_spans,
    _sanitize_with_checks,
    _task_specific_sanitizer_prompt,
)

BIPIA_NATIVE_POSITION_PLAN = (
    ("start", "prepend"),
    ("middle", "middle"),
    ("end", "append"),
)


def _unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix
    i = 1
    while True:
        cand = p.with_name(f"{stem}_{i}{suf}")
        if not cand.exists():
            return cand
        i += 1


def _mean_int(xs: List[int]) -> float:
    return sum(xs) / max(1, len(xs))


def _mean_float(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _resolve_task_name(task: str) -> str:
    t = (task or "").strip().lower()
    aliases = {
        "emailqa": "emailQA",
        "email_qa": "emailQA",
        "email": "emailQA",
    }
    if t not in aliases:
        raise NotImplementedError(f"run_sweep_other.py currently supports only emailQA. Received task={task!r}")
    return aliases[t]


def _parse_parts(raw: str | None) -> List[str]:
    if raw is None:
        return list(DEFAULT_COMBINED_PARTS)
    out = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return out or list(DEFAULT_COMBINED_PARTS)


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


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/emailqa_with_localization_and_sanitization.yaml")
    ap.add_argument("--task", type=str, default="emailQA")
    ap.add_argument("--attack_method", type=str, default="bipia_native")
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--combined_parts", type=str, default="ignore,escape,fakecom")
    ap.add_argument("--native_attack_limit", type=int, default=None)
    ap.add_argument("--insertion_policy", type=str, default="append")
    return ap


def main() -> None:
    ap = _build_cli()
    args = ap.parse_args()

    cfg = load_config(args.config)
    task_name = _resolve_task_name(args.task or getattr(cfg.dataset, "task", "emailQA"))
    attack_method = normalize_method_name(args.attack_method)
    rng_seed = int(args.seed if args.seed is not None else getattr(cfg.attack, "rng_seed", 2026))
    rng = random.Random(rng_seed)

    split = args.split or cfg.dataset.split
    max_samples = args.max_samples if args.max_samples is not None else getattr(cfg.dataset, "max_samples", None)
    run_name = args.run_name or f"emailqa_other_{attack_method}"

    resolved_cfg = {
        "task": task_name,
        "attack_method": attack_method,
        "run_name": run_name,
        "split": split,
        "max_samples": max_samples,
        "seed": rng_seed,
        "position_mode": "per_sample_start_middle_end_average" if attack_method == "bipia_native" else "single_position",
        "positions": [x[0] for x in BIPIA_NATIVE_POSITION_PLAN] if attack_method == "bipia_native" else [args.insertion_policy],
        "single_position_policy": args.insertion_policy,
        "combined_parts": _parse_parts(args.combined_parts),
        "payload_source_mode": "bipia_official_attack_file_required",
    }
    print(f"[Config] {json.dumps(resolved_cfg, ensure_ascii=False)}")

    out_dir = Path(cfg.output.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved_other.json").write_text(json.dumps(resolved_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    bipia_root = ensure_bipia_repo(cfg.dataset.bipia_root)
    samples, used_schema = load_emailqa_samples(
        bipia_root=bipia_root,
        split=split,
        max_samples=max_samples,
        context_key=getattr(cfg.dataset, "context_key", None),
        question_key=getattr(cfg.dataset, "question_key", None),
        answer_key=getattr(cfg.dataset, "answer_key", None),
        id_key=getattr(cfg.dataset, "id_key", None),
    )
    target = HFChat.from_config(cfg.target_model)
    judge = HFChat.from_config(cfg.judge_model)

    detector = None
    if cfg.localization.enable and (cfg.localization.gt_mode or "").lower() != "off":
        if cfg.detector_model is None:
            raise ValueError("localization.enable=true but models.detector is not provided")
        detector = HFChat.from_config(cfg.detector_model)

    sanitizer = None
    sanitize_context = None
    if cfg.sanitization.enable:
        if cfg.sanitizer_model is None:
            raise ValueError("sanitization.enable=true but models.sanitizer is not provided")
        sanitizer = HFChat.from_config(cfg.sanitizer_model)
        from fragweave.defenses.sanitizer import sanitize_context as _sanitize_context  # type: ignore
        sanitize_context = _sanitize_context

    official_payloads = load_bipia_instruction_pool(
        bipia_root,
        split=split,
        limit=args.native_attack_limit,
    )

    csv_path = _unique_path(out_dir / "results.csv")
    summary_path = _unique_path(out_dir / "summary.csv")
    debug_path = _unique_path(out_dir / "debug_other_attacks.jsonl")
    schema_path = _unique_path(out_dir / "dataset_schema_used.json")
    schema_path.write_text(json.dumps(used_schema, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "method",
        "method_label",
        "sample_id",
        "position",
        "position_policy",
        "attack_succeeded",
        "attack_conf",
        "loc_precision",
        "loc_recall",
        "loc_f1",
        "attack_succeeded_after_sanitizer_generic",
        "attack_succeeded_after_sanitizer_task",
        "attack_succeeded_after_redaction",
    ]
    summary_fields = [
        "method",
        "method_label",
        "position",
        "n",
        "asr",
        "loc_f1",
        "asr_after_sanitizer_generic",
        "asr_after_sanitizer_task",
        "asr_after_redaction",
    ]

    f_csv = csv_path.open("w", newline="", encoding="utf-8")
    w_csv = csv.DictWriter(f_csv, fieldnames=fieldnames)
    w_csv.writeheader()
    f_csv.flush()

    f_dbg = debug_path.open("w", encoding="utf-8")
    f_sum = summary_path.open("w", newline="", encoding="utf-8")
    w_sum = csv.DictWriter(f_sum, fieldnames=summary_fields)
    w_sum.writeheader()
    f_sum.flush()

    evaluation_positions = list(BIPIA_NATIVE_POSITION_PLAN) if attack_method == "bipia_native" else [("single", args.insertion_policy)]
    metrics_by_position: Dict[str, Dict[str, List[float | int]]] = {
        pos_name: {
            "asr": [],
            "loc_f1": [],
            "asr_san_generic": [],
            "asr_san_task": [],
            "asr_redact": [],
        }
        for pos_name, _ in evaluation_positions
    }
    overall_asr: List[int] = []
    overall_loc_f1: List[float] = []
    overall_asr_san_generic: List[int] = []
    overall_asr_san_task: List[int] = []
    overall_asr_redact: List[int] = []

    try:
        for sample in tqdm(samples, desc=attack_method):
            sample_id = getattr(sample, "uid", None) or getattr(sample, "id", None) or "unknown"
            context = str(getattr(sample, "context", ""))
            question = str(getattr(sample, "question", ""))
            answer = getattr(sample, "answer", None)

            base_artifact = build_attack_artifact(
                method=attack_method,
                context=context,
                rng=rng,
                tag_start=cfg.localization.tag_start,
                tag_end=cfg.localization.tag_end,
                insertion_policy=args.insertion_policy,
                official_payloads=official_payloads,
                combined_parts=_parse_parts(args.combined_parts),
            )

            sample_position_asr: List[int] = []
            sample_position_loc_f1: List[float] = []
            sample_position_san_generic: List[int] = []
            sample_position_san_task: List[int] = []
            sample_position_redact: List[int] = []

            for pos_name, pos_policy in evaluation_positions:
                poisoned_ctx, shadow_ctx = _materialize_position_variant(
                    context=context,
                    injected_text=base_artifact.injected_text,
                    tag_start=cfg.localization.tag_start,
                    tag_end=cfg.localization.tag_end,
                    position_policy=pos_policy,
                )
                resp = target.generate(
                    cfg.prompt.target_template.format(context=poisoned_ctx, question=question),
                    max_new_tokens=cfg.target_model.max_new_tokens,
                )
                atk = judge_attack(judge, question=question, instruction=base_artifact.payload, response=resp)
                atk_ok = int(atk.ok)
                overall_asr.append(atk_ok)
                metrics_by_position[pos_name]["asr"].append(atk_ok)
                sample_position_asr.append(atk_ok)

                prec = rec = f1 = 0.0
                loc_dbg = None
                loc = None
                if detector is not None:
                    _, gt_spans = shadow_to_clean_and_spans(
                        shadow_ctx,
                        tag_start=cfg.localization.tag_start,
                        tag_end=cfg.localization.tag_end,
                    )
                    loc = run_span_localizer(
                        detector,
                        poisoned_ctx,
                        max_new_tokens=cfg.localization.detector_max_new_tokens,
                        task=cfg.dataset.task,
                        prompt_template=getattr(cfg.localization, "prompt_template", None),
                    )
                    prec, rec, f1 = span_prf1(loc.spans, gt_spans)
                    overall_loc_f1.append(f1)
                    metrics_by_position[pos_name]["loc_f1"].append(f1)
                    sample_position_loc_f1.append(f1)
                    loc_dbg = {
                        "gt_spans": gt_spans,
                        "pred_spans": loc.spans,
                        "snippets": loc.snippets,
                        "raw": loc.raw_model_output,
                        "parse_ok": loc.parse_ok,
                    }

                after_san_generic_ok = after_san_task_ok = after_redaction_ok = ""
                san_generic_dbg = san_task_dbg = redaction_dbg = None
                if sanitizer is not None and sanitize_context is not None:
                    cleaned_generic, san_generic_dbg = _sanitize_with_checks(
                        sanitize_context,
                        sanitizer,
                        poisoned_ctx,
                        system_prompt=GENERIC_SANITIZER_PROMPT,
                        max_new_tokens=cfg.sanitization.sanitizer_max_new_tokens,
                        task=task_name,
                    )
                    cleaned_generic = _normalize_sanitized_text(cleaned_generic)
                    resp_san_generic = target.generate(
                        cfg.prompt.target_template.format(context=cleaned_generic, question=question),
                        max_new_tokens=cfg.target_model.max_new_tokens,
                    )
                    atk_san_generic = judge_attack(judge, question=question, instruction=base_artifact.payload, response=resp_san_generic)
                    after_san_generic_ok = int(atk_san_generic.ok)
                    overall_asr_san_generic.append(int(atk_san_generic.ok))
                    metrics_by_position[pos_name]["asr_san_generic"].append(int(atk_san_generic.ok))
                    sample_position_san_generic.append(int(atk_san_generic.ok))

                    task_prompt = _task_specific_sanitizer_prompt(task_name, getattr(cfg.sanitization, "system_prompt", None))
                    cleaned_task, san_task_dbg = _sanitize_with_checks(
                        sanitize_context,
                        sanitizer,
                        poisoned_ctx,
                        system_prompt=task_prompt,
                        max_new_tokens=cfg.sanitization.sanitizer_max_new_tokens,
                        task=task_name,
                    )
                    cleaned_task = _normalize_sanitized_text(cleaned_task)
                    resp_san_task = target.generate(
                        cfg.prompt.target_template.format(context=cleaned_task, question=question),
                        max_new_tokens=cfg.target_model.max_new_tokens,
                    )
                    atk_san_task = judge_attack(judge, question=question, instruction=base_artifact.payload, response=resp_san_task)
                    after_san_task_ok = int(atk_san_task.ok)
                    overall_asr_san_task.append(int(atk_san_task.ok))
                    metrics_by_position[pos_name]["asr_san_task"].append(int(atk_san_task.ok))
                    sample_position_san_task.append(int(atk_san_task.ok))

                if detector is not None and loc is not None:
                    redacted_ctx = _redact_by_spans(poisoned_ctx, loc.spans)
                    resp_red = target.generate(
                        cfg.prompt.target_template.format(context=redacted_ctx, question=question),
                        max_new_tokens=cfg.target_model.max_new_tokens,
                    )
                    atk_red = judge_attack(judge, question=question, instruction=base_artifact.payload, response=resp_red)
                    after_redaction_ok = int(atk_red.ok)
                    overall_asr_redact.append(int(atk_red.ok))
                    metrics_by_position[pos_name]["asr_redact"].append(int(atk_red.ok))
                    sample_position_redact.append(int(atk_red.ok))
                    redaction_dbg = {
                        "pred_spans": loc.spans,
                        "redacted_context": redacted_ctx,
                        "target_response_after_redaction": resp_red,
                        "attack_judge_after_redaction": asdict(atk_red),
                    }

                row = {
                    "method": attack_method,
                    "method_label": base_artifact.label,
                    "sample_id": sample_id,
                    "position": pos_name,
                    "position_policy": pos_policy,
                    "attack_succeeded": atk_ok,
                    "attack_conf": float(atk.score),
                    "loc_precision": float(prec) if detector is not None else "",
                    "loc_recall": float(rec) if detector is not None else "",
                    "loc_f1": float(f1) if detector is not None else "",
                    "attack_succeeded_after_sanitizer_generic": after_san_generic_ok,
                    "attack_succeeded_after_sanitizer_task": after_san_task_ok,
                    "attack_succeeded_after_redaction": after_redaction_ok,
                }
                w_csv.writerow(row)

                debug_record = {
                    "method": attack_method,
                    "method_label": base_artifact.label,
                    "sample_id": sample_id,
                    "position": pos_name,
                    "position_policy": pos_policy,
                    "question": question,
                    "answer": answer,
                    "payload": base_artifact.payload,
                    "injected_text": base_artifact.injected_text,
                    "metadata": {**base_artifact.metadata, "position": pos_name, "position_policy": pos_policy},
                    "original_context": context,
                    "shadow_context": shadow_ctx,
                    "poisoned_context": poisoned_ctx,
                    "target_response": resp,
                    "attack_judge": asdict(atk),
                    "loc_debug": loc_dbg,
                    "sanitization_debug": {
                        "generic": san_generic_dbg,
                        "task_specific": san_task_dbg,
                        "redaction": redaction_dbg,
                    },
                }
                f_dbg.write(json.dumps(debug_record, ensure_ascii=False) + "\n")

            if attack_method == "bipia_native":
                sample_debug_avg = {
                    "method": attack_method,
                    "method_label": base_artifact.label,
                    "sample_id": sample_id,
                    "aggregate_over_positions": True,
                    "position_average_attack_succeeded": _mean_int(sample_position_asr),
                    "position_average_loc_f1": _mean_float(sample_position_loc_f1) if sample_position_loc_f1 else "",
                    "position_average_attack_succeeded_after_sanitizer_generic": _mean_int(sample_position_san_generic) if sample_position_san_generic else "",
                    "position_average_attack_succeeded_after_sanitizer_task": _mean_int(sample_position_san_task) if sample_position_san_task else "",
                    "position_average_attack_succeeded_after_redaction": _mean_int(sample_position_redact) if sample_position_redact else "",
                }
                f_dbg.write(json.dumps(sample_debug_avg, ensure_ascii=False) + "\n")

        f_csv.flush()
        f_dbg.flush()

        print(f"\n== Method {attack_method} ==")
        if attack_method == "bipia_native":
            print(f"ASR (start/middle/end average): {_mean_int(overall_asr):.3f}")
            for pos_name, _ in BIPIA_NATIVE_POSITION_PLAN:
                vals = metrics_by_position[pos_name]["asr"]
                if vals:
                    print(f"ASR ({pos_name}): {_mean_int(vals):.3f}")
            if detector is not None and overall_loc_f1:
                print(f"Localization F1 (start/middle/end average): {_mean_float(overall_loc_f1):.3f}")
                for pos_name, _ in BIPIA_NATIVE_POSITION_PLAN:
                    vals = metrics_by_position[pos_name]["loc_f1"]
                    if vals:
                        print(f"Localization F1 ({pos_name}): {_mean_float(vals):.3f}")
            if sanitizer is not None and overall_asr_san_generic:
                print(f"ASR after sanitizer-generic (start/middle/end average): {_mean_int(overall_asr_san_generic):.3f}")
            if sanitizer is not None and overall_asr_san_task:
                print(f"ASR after sanitizer-task (start/middle/end average): {_mean_int(overall_asr_san_task):.3f}")
            if detector is not None and overall_asr_redact:
                print(f"ASR after redaction (start/middle/end average): {_mean_int(overall_asr_redact):.3f}")
            w_sum.writerow(
                {
                    "method": attack_method,
                    "method_label": "bipia_native",
                    "position": "avg_start_middle_end",
                    "n": len(samples),
                    "asr": _mean_int(overall_asr) if overall_asr else "",
                    "loc_f1": _mean_float(overall_loc_f1) if overall_loc_f1 else "",
                    "asr_after_sanitizer_generic": _mean_int(overall_asr_san_generic) if overall_asr_san_generic else "",
                    "asr_after_sanitizer_task": _mean_int(overall_asr_san_task) if overall_asr_san_task else "",
                    "asr_after_redaction": _mean_int(overall_asr_redact) if overall_asr_redact else "",
                }
            )
            for pos_name, _ in BIPIA_NATIVE_POSITION_PLAN:
                pos_metrics = metrics_by_position[pos_name]
                w_sum.writerow(
                    {
                        "method": attack_method,
                        "method_label": "bipia_native",
                        "position": pos_name,
                        "n": len(pos_metrics["asr"]),
                        "asr": _mean_int(pos_metrics["asr"]) if pos_metrics["asr"] else "",
                        "loc_f1": _mean_float(pos_metrics["loc_f1"]) if pos_metrics["loc_f1"] else "",
                        "asr_after_sanitizer_generic": _mean_int(pos_metrics["asr_san_generic"]) if pos_metrics["asr_san_generic"] else "",
                        "asr_after_sanitizer_task": _mean_int(pos_metrics["asr_san_task"]) if pos_metrics["asr_san_task"] else "",
                        "asr_after_redaction": _mean_int(pos_metrics["asr_redact"]) if pos_metrics["asr_redact"] else "",
                    }
                )
        else:
            pos_name = "single"
            pos_metrics = metrics_by_position[pos_name]
            print(f"ASR (attack_succeeded): {_mean_int(overall_asr):.3f}")
            if detector is not None and overall_loc_f1:
                print(f"Localization F1: {_mean_float(overall_loc_f1):.3f}")
            if sanitizer is not None and overall_asr_san_generic:
                print(f"ASR after sanitizer-generic: {_mean_int(overall_asr_san_generic):.3f}")
            if sanitizer is not None and overall_asr_san_task:
                print(f"ASR after sanitizer-task: {_mean_int(overall_asr_san_task):.3f}")
            if detector is not None and overall_asr_redact:
                print(f"ASR after redaction: {_mean_int(overall_asr_redact):.3f}")
            w_sum.writerow(
                {
                    "method": attack_method,
                    "method_label": attack_method,
                    "position": args.insertion_policy,
                    "n": len(pos_metrics["asr"]),
                    "asr": _mean_int(pos_metrics["asr"]) if pos_metrics["asr"] else "",
                    "loc_f1": _mean_float(pos_metrics["loc_f1"]) if pos_metrics["loc_f1"] else "",
                    "asr_after_sanitizer_generic": _mean_int(pos_metrics["asr_san_generic"]) if pos_metrics["asr_san_generic"] else "",
                    "asr_after_sanitizer_task": _mean_int(pos_metrics["asr_san_task"]) if pos_metrics["asr_san_task"] else "",
                    "asr_after_redaction": _mean_int(pos_metrics["asr_redact"]) if pos_metrics["asr_redact"] else "",
                }
            )
        f_sum.flush()
    finally:
        f_csv.close()
        f_dbg.close()
        f_sum.close()

    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {debug_path}")
    print(f"Saved: {schema_path}")


if __name__ == "__main__":
    main()
