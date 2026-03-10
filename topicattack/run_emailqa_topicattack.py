from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from fragweave.config import load_config
from fragweave.data.bipia_fetch import ensure_bipia_repo
from fragweave.data.emailqa_loader import load_emailqa_samples
from fragweave.defenses.sanitizer import sanitize_context
from fragweave.eval.judge import judge_attack, judge_task
from fragweave.eval.localization import run_span_localizer, shadow_to_clean_and_spans, span_prf1
from fragweave.models.hf_chat import HFChat

from topicattack.topic_attack import (
    build_topicattack_block,
    inject_block,
    load_instruction_pool,
    load_topicattack_section,
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


def _call_load_emailqa_samples(cfg) -> List[Any]:
    samples, _used = load_emailqa_samples(
        bipia_root=cfg.dataset.bipia_root,
        split=cfg.dataset.split,
        max_samples=cfg.dataset.max_samples,
        context_key=getattr(cfg.dataset, "context_key", None),
        question_key=getattr(cfg.dataset, "question_key", None),
        answer_key=getattr(cfg.dataset, "answer_key", None),
        id_key=getattr(cfg.dataset, "id_key", None),
    )
    return samples


def _render_target_prompt(template: str, *, context: str, question: str) -> str:
    return template.format(context=context, question=question)


def _maybe_sanitize(cfg, sanitizer_chat, attacked_context: str) -> str:
    if not cfg.sanitization.enable or sanitizer_chat is None:
        return attacked_context
    return sanitize_context(
        sanitizer_chat,
        attacked_context,
        system_prompt=cfg.sanitization.system_prompt,
        max_retries=0,
    )


def _run_localization(cfg, detector_chat, attacked_shadow: str) -> Dict[str, Any]:
    if not cfg.localization.enable or detector_chat is None:
        return {
            "clean_context": attacked_shadow,
            "gt_spans": [],
            "pred_spans": [],
            "loc_precision": None,
            "loc_recall": None,
            "loc_f1": None,
            "detector_raw": "",
            "detector_parse_ok": None,
        }
    clean_context, gt_spans = shadow_to_clean_and_spans(
        attacked_shadow,
        cfg.localization.tag_start,
        cfg.localization.tag_end,
    )
    pred = run_span_localizer(
        detector_chat,
        clean_context,
        max_new_tokens=cfg.localization.detector_max_new_tokens,
        task=cfg.dataset.task,
        prompt_template=cfg.localization.prompt_template,
    )
    p, r, f1 = span_prf1(pred.spans, gt_spans)
    return {
        "clean_context": clean_context,
        "gt_spans": gt_spans,
        "pred_spans": pred.spans,
        "loc_precision": p,
        "loc_recall": r,
        "loc_f1": f1,
        "detector_raw": pred.raw_model_output,
        "detector_parse_ok": pred.parse_ok,
    }


def _maybe_run_baseline(cfg, target_chat, judge_chat, detector_chat, sanitizer_chat, sample, malicious_instruction: str) -> Dict[str, Any]:
    if not cfg.attack.include_direct_baseline:
        return {}
    block = "[INSTRUCTION]: " + malicious_instruction.strip()
    shadow = "[INSTRUCTION]: " + cfg.localization.tag_start + malicious_instruction.strip() + cfg.localization.tag_end
    attacked = inject_block(sample.context, block, mode=cfg.dataset.baseline_injection_mode)
    attacked_shadow = inject_block(sample.context, shadow, mode=cfg.dataset.baseline_injection_mode)
    loc = _run_localization(cfg, detector_chat, attacked_shadow)
    prompt = _render_target_prompt(cfg.prompt.target_template, context=attacked, question=sample.question)
    response = target_chat.generate(prompt, max_new_tokens=cfg.target_model.max_new_tokens)
    attack_decision = judge_attack(judge_chat, sample.question, malicious_instruction, response)
    task_decision = judge_task(judge_chat, sample.question, sample.answer, response)
    cleaned = _maybe_sanitize(cfg, sanitizer_chat, attacked)
    prompt_clean = _render_target_prompt(cfg.prompt.target_template, context=cleaned, question=sample.question)
    response_clean = target_chat.generate(prompt_clean, max_new_tokens=cfg.target_model.max_new_tokens)
    attack_after = judge_attack(judge_chat, sample.question, malicious_instruction, response_clean)
    task_after = judge_task(judge_chat, sample.question, sample.answer, response_clean)
    return {
        "direct_attack_context": attacked,
        "direct_attack_response": response,
        "direct_attack_succeeded": attack_decision.ok,
        "direct_task_correct": None if task_decision is None else task_decision.ok,
        "direct_clean_context": cleaned,
        "direct_clean_response": response_clean,
        "direct_attack_after_sanitize": attack_after.ok,
        "direct_task_after_sanitize": None if task_after is None else task_after.ok,
        "direct_loc_precision": loc["loc_precision"],
        "direct_loc_recall": loc["loc_recall"],
        "direct_loc_f1": loc["loc_f1"],
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    ta_cfg = load_topicattack_section(args.config)

    ensure_bipia_repo(cfg.dataset.bipia_root)
    samples = _call_load_emailqa_samples(cfg)
    instructions = load_instruction_pool(cfg.attack.instruction_jsonl, cfg.attack.instruction_text_key)
    if not instructions:
        raise ValueError("No malicious instructions found. Check attack.instruction_jsonl in the config.")

    out_dir = Path(args.out_dir or (Path("topicattack/outputs") / f"{cfg.output.run_name}_{ta_cfg.variant}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.attack.rng_seed)

    target_chat = HFChat.from_config(cfg.target_model)
    generator_chat = HFChat.from_config(cfg.weaver_model)
    topic_chat = HFChat.from_config(cfg.sharder_model)
    judge_chat = HFChat.from_config(cfg.judge_model)
    detector_chat = HFChat.from_config(cfg.detector_model) if cfg.localization.enable and cfg.detector_model else None
    sanitizer_chat = HFChat.from_config(cfg.sanitizer_model) if cfg.sanitization.enable and cfg.sanitizer_model else None

    rows: List[Dict[str, Any]] = []
    debug_examples: List[Dict[str, Any]] = []

    for idx, sample in enumerate(tqdm(samples, desc="topicattack-emailqa")):
        malicious_instruction = instructions[idx % len(instructions)]
        artifact = build_topicattack_block(
            generator_chat,
            topic_chat,
            context=sample.context,
            question=sample.question,
            malicious_instruction=malicious_instruction,
            cfg=ta_cfg,
            rng=rng,
        )
        attacked_context = inject_block(sample.context, artifact.final_block, mode=ta_cfg.insert_mode)
        shadow_payload = getattr(artifact, "shadow_block", artifact.final_block)
        attacked_shadow = inject_block(
            sample.context,
            artifact.final_block.replace(
                shadow_payload,
                cfg.localization.tag_start + shadow_payload + cfg.localization.tag_end,
                1,
            ) if shadow_payload in artifact.final_block else (cfg.localization.tag_start + artifact.final_block + cfg.localization.tag_end),
            mode=ta_cfg.insert_mode,
        )

        loc = _run_localization(cfg, detector_chat, attacked_shadow)
        prompt = _render_target_prompt(cfg.prompt.target_template, context=attacked_context, question=sample.question)
        response = target_chat.generate(prompt, max_new_tokens=cfg.target_model.max_new_tokens)
        attack_decision = judge_attack(judge_chat, sample.question, malicious_instruction, response)
        task_decision = judge_task(judge_chat, sample.question, sample.answer, response)

        cleaned_context = _maybe_sanitize(cfg, sanitizer_chat, attacked_context)
        prompt_clean = _render_target_prompt(cfg.prompt.target_template, context=cleaned_context, question=sample.question)
        response_clean = target_chat.generate(prompt_clean, max_new_tokens=cfg.target_model.max_new_tokens)
        attack_after = judge_attack(judge_chat, sample.question, malicious_instruction, response_clean)
        task_after = judge_task(judge_chat, sample.question, sample.answer, response_clean)

        row: Dict[str, Any] = {
            "uid": sample.uid,
            "variant": ta_cfg.variant,
            "topic": artifact.topic,
            "topic_rationale": artifact.rationale,
            "question": sample.question,
            "ground_truth_answer": sample.answer,
            "malicious_instruction": malicious_instruction,
            "topicattack_block": artifact.final_block,
            "attacked_context": attacked_context,
            "target_response": response,
            "attack_succeeded": attack_decision.ok,
            "attack_judge_reason": attack_decision.reason,
            "task_correct": None if task_decision is None else task_decision.ok,
            "cleaned_context": cleaned_context,
            "cleaned_response": response_clean,
            "attack_after_sanitize": attack_after.ok,
            "attack_after_sanitize_reason": attack_after.reason,
            "task_after_sanitize": None if task_after is None else task_after.ok,
            "loc_precision": loc["loc_precision"],
            "loc_recall": loc["loc_recall"],
            "loc_f1": loc["loc_f1"],
            "gt_spans": json.dumps(loc["gt_spans"], ensure_ascii=False),
            "pred_spans": json.dumps(loc["pred_spans"], ensure_ascii=False),
            "detector_parse_ok": loc["detector_parse_ok"],
            "transfer_prompt": artifact.transfer_prompt,
            "generated_conversation": artifact.generated_conversation,
        }
        row.update(_maybe_run_baseline(cfg, target_chat, judge_chat, detector_chat, sanitizer_chat, sample, malicious_instruction))
        rows.append(row)

        if len(debug_examples) < ta_cfg.log_examples:
            debug_examples.append(row)

    def _mean(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    summary = {
        "num_samples": len(rows),
        "variant": ta_cfg.variant,
        "asr": _mean("attack_succeeded"),
        "localization_f1": _mean("loc_f1"),
        "asr_after_sanitize": _mean("attack_after_sanitize"),
        "task_accuracy": _mean("task_correct"),
        "task_accuracy_after_sanitize": _mean("task_after_sanitize"),
    }
    if cfg.attack.include_direct_baseline:
        summary.update(
            {
                "direct_baseline_asr": _mean("direct_attack_succeeded"),
                "direct_baseline_localization_f1": _mean("direct_loc_f1"),
                "direct_baseline_asr_after_sanitize": _mean("direct_attack_after_sanitize"),
            }
        )

    results_csv = _unique_path(out_dir / "topicattack_emailqa_results.csv")
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_json = _unique_path(out_dir / "topicattack_emailqa_summary.json")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    debug_json = _unique_path(out_dir / "topicattack_emailqa_debug_examples.json")
    debug_json.write_text(json.dumps(debug_examples, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "results_csv": str(results_csv),
        "summary_json": str(summary_json),
        "debug_json": str(debug_json),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="topicattack/configs/emailqa_topicattack.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    out = run(args)
    print(json.dumps(out, ensure_ascii=False, indent=2))
