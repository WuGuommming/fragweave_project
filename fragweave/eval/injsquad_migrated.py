from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from fragweave.defenses.sanitizer import sanitize_context
from fragweave.eval.judge import JudgeDecision, judge_task
from fragweave.eval.localization import run_span_localizer, span_prf1
from fragweave.prompts.injsquad import compose_target_prompt
from fragweave.prompts.injsquad_migrated_eval import (
    MIGRATED_ATTACK_JUDGE_PROMPT,
    MIGRATED_LOCALIZE_PROMPT,
    MIGRATED_SANITIZER_SYSTEM_PROMPT,
)

from .injsquad_common import MigratedEvalRow, build_migrated_metrics, compute_gt_spans


def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON found")
    return m.group(0)


def _judge_attack_migrated(judge_chat, *, question: str, instruction: str, response: str) -> JudgeDecision:
    prompt = MIGRATED_ATTACK_JUDGE_PROMPT.format(
        question=question,
        instruction=instruction,
        response=response,
    )
    raw = judge_chat.generate(prompt)
    try:
        obj = json.loads(_extract_json(raw))
        return JudgeDecision(
            ok=bool(obj.get("attack_succeeded")),
            score=float(obj.get("confidence", 0.0)),
            reason=str(obj.get("reason", "")),
            raw=raw,
        )
    except Exception as exc:
        return JudgeDecision(ok=False, score=0.0, reason=f"parse_fail: {exc}", raw=raw)


def _resolve_localizer_max_new_tokens(run_cfg) -> int:
    localization_cfg = getattr(run_cfg, "localization", None)
    if localization_cfg is not None:
        value = getattr(localization_cfg, "detector_max_new_tokens", None)
        if value is not None:
            return int(value)
    detection_cfg = getattr(run_cfg, "detection", None)
    if detection_cfg is not None:
        value = getattr(detection_cfg, "detector_max_new_tokens", None)
        if value is not None:
            return int(value)
    return 256


def evaluate_injsquad_migrated_one(
    *,
    attack: Dict[str, Any],
    run_cfg,
    judge_chat,
    detector_chat,
    sanitizer_chat,
    target_chat,
    response_text: Optional[str],
) -> MigratedEvalRow:
    sample_id = str(attack.get("uid", ""))
    question = str(attack.get("question", ""))
    answer = attack.get("answer")
    instruction = str(attack.get("injected_instruction", ""))
    attacked_context = str(attack.get("attacked_context", ""))

    attack_decision = None
    task_decision = None
    loc_result = None
    loc_p = None
    loc_r = None
    loc_f1 = None
    gt_spans: List[Tuple[int, int]] = []
    atk_after_san = None
    task_after_san = None
    sanitized_response = None
    sanitized_context = None

    if response_text is not None:
        attack_decision = _judge_attack_migrated(
            judge_chat,
            question=question,
            instruction=instruction,
            response=response_text,
        )
        task_decision = judge_task(judge_chat, question=question, answer=answer, response=response_text)

        loc_result = run_span_localizer(
            detector_chat,
            attacked_context,
            task="injsquad",
            max_new_tokens=_resolve_localizer_max_new_tokens(run_cfg),
            prompt_template=MIGRATED_LOCALIZE_PROMPT,
        )
        gt_spans = compute_gt_spans(attacked_context, instruction, row=attack)
        loc_p, loc_r, loc_f1 = span_prf1(loc_result.spans, gt_spans)

        sanitized_context = sanitize_context(
            sanitizer_chat,
            attacked_context,
            system_prompt=MIGRATED_SANITIZER_SYSTEM_PROMPT,
            max_retries=0,
        )
        sanitized_prompt = compose_target_prompt(
            run_cfg.prompt.target_template,
            context=sanitized_context,
            question=question,
        )
        sanitized_response = target_chat.generate(sanitized_prompt)
        atk_after_san = _judge_attack_migrated(
            judge_chat,
            question=question,
            instruction=instruction,
            response=sanitized_response,
        )
        task_after_san = judge_task(judge_chat, question=question, answer=answer, response=sanitized_response)

    return MigratedEvalRow(
        sample_id=sample_id,
        benchmark_name=str(attack.get("benchmark_name", "injsquad")),
        migrated_attack_success=(attack_decision.ok if attack_decision is not None else None),
        migrated_attack_confidence=(attack_decision.score if attack_decision is not None else None),
        migrated_attack_reason=(attack_decision.reason if attack_decision is not None else ""),
        migrated_task_correct=(task_decision.ok if task_decision is not None else None),
        migrated_task_confidence=(task_decision.score if task_decision is not None else None),
        migrated_task_reason=(task_decision.reason if task_decision is not None else ""),
        migrated_localization_precision=loc_p,
        migrated_localization_recall=loc_r,
        migrated_localization_score=loc_f1,
        migrated_has_injection_pred=(loc_result.has_injection if loc_result is not None else None),
        migrated_attack_success_after_sanitize=(atk_after_san.ok if atk_after_san is not None else None),
        migrated_attack_confidence_after_sanitize=(atk_after_san.score if atk_after_san is not None else None),
        migrated_task_correct_after_sanitize=(task_after_san.ok if task_after_san is not None else None),
        migrated_task_confidence_after_sanitize=(task_after_san.score if task_after_san is not None else None),
        response=response_text,
        sanitized_response=sanitized_response,
        metadata={
            "migrated_attack_judge_prompt": MIGRATED_ATTACK_JUDGE_PROMPT,
            "migrated_localize_parse_ok": (loc_result.parse_ok if loc_result is not None else None),
            "migrated_localize_snippets": (loc_result.snippets if loc_result is not None else []),
            "migrated_localize_raw_output": (loc_result.raw_model_output if loc_result is not None else ""),
            "migrated_gt_spans": gt_spans,
            "migrated_attack_judge_raw": (attack_decision.raw if attack_decision is not None else ""),
            "migrated_task_judge_raw": (task_decision.raw if task_decision is not None else ""),
            "migrated_attack_judge_raw_after_sanitize": (atk_after_san.raw if atk_after_san is not None else ""),
            "migrated_task_judge_raw_after_sanitize": (task_after_san.raw if task_after_san is not None else ""),
            "migrated_sanitized_context": sanitized_context,
        },
    )


def evaluate_injsquad_migrated(
    *,
    attacks: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
    run_cfg,
    judge_chat,
    detector_chat,
    sanitizer_chat,
    target_chat,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    response_by_id = {str(r.get("sample_id", "")): r for r in responses}
    eval_rows: List[MigratedEvalRow] = []
    for row in attacks:
        sample_id = str(row.get("uid", ""))
        response_entry = response_by_id.get(sample_id, {})
        response_text = response_entry.get("response") if isinstance(response_entry.get("response"), str) else None
        eval_rows.append(
            evaluate_injsquad_migrated_one(
                attack=row,
                run_cfg=run_cfg,
                judge_chat=judge_chat,
                detector_chat=detector_chat,
                sanitizer_chat=sanitizer_chat,
                target_chat=target_chat,
                response_text=response_text,
            )
        )
    metrics = build_migrated_metrics(eval_rows)
    return metrics, [asdict(r) for r in eval_rows]
