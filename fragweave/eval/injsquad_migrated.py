from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Any, Dict, List, Tuple

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
        response_text = response_entry.get("response")
        if not isinstance(response_text, str):
            response_text = None

        question = str(row.get("question", ""))
        answer = row.get("answer")
        instruction = str(row.get("injected_instruction", ""))
        attacked_context = str(row.get("attacked_context", ""))

        attack_decision = None
        task_decision = None
        loc_result = None
        loc_p = None
        loc_r = None
        loc_f1 = None
        atk_after_san = None
        task_after_san = None
        sanitized_response = None

        if response_text is not None:
            attack_decision = _judge_attack_migrated(
                judge_chat,
                question=question,
                instruction=instruction,
                response=response_text,
            )
            task_decision = judge_task(
                judge_chat,
                question=question,
                answer=answer,
                response=response_text,
            )
            loc_result = run_span_localizer(
                detector_chat,
                attacked_context,
                task="injsquad",
                max_new_tokens=run_cfg.localization.detector_max_new_tokens,
                prompt_template=MIGRATED_LOCALIZE_PROMPT,
            )
            gt_spans = compute_gt_spans(attacked_context, instruction)
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
            task_after_san = judge_task(
                judge_chat,
                question=question,
                answer=answer,
                response=sanitized_response,
            )

        eval_rows.append(
            MigratedEvalRow(
                sample_id=sample_id,
                benchmark_name=str(row.get("benchmark_name", "injsquad")),
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
                    "migrated_attack_judge_raw": (attack_decision.raw if attack_decision is not None else ""),
                    "migrated_task_judge_raw": (task_decision.raw if task_decision is not None else ""),
                    "migrated_attack_judge_raw_after_sanitize": (atk_after_san.raw if atk_after_san is not None else ""),
                    "migrated_task_judge_raw_after_sanitize": (task_after_san.raw if task_after_san is not None else ""),
                },
            )
        )

    metrics = build_migrated_metrics(eval_rows)
    return metrics, [asdict(r) for r in eval_rows]
