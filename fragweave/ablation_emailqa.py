from __future__ import annotations

"""
EmailQA ablation runner (FragWeave) — with an extra "no-shard" variant.

This script is placed next to `run_sweep.py` (same package level) and reuses
project implementations to avoid guessing behavior.

Ablation dimensions:
  - shard mode:
      * k in {3,4,5} with use_cross_refs in {False, True}
      * plus an extra mode: no-shard
          - if weave is enabled: give the original injection to weaver to blend
          - if weave is disabled: same as direct injection baseline
  - weave mode: {without, with, with+decoy}
  - guidance mode: {without, A, B}

Default: EmailQA, 50 samples.

Outputs (incremental writes + flush):
  - results.csv (per-sample)
  - summary_by_variant.csv (per-variant)
  - debug_fragments.jsonl (per-sample debug)
  - dataset_schema_used.json
"""

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from fragweave.config import load_config
from fragweave.data.bipia_fetch import ensure_bipia_repo
from fragweave.attacks.sharder import shard_with_llm
from fragweave.attacks.weaver import apply_weave, apply_weave_with_shadow
from fragweave.attacks.decoy import generate_decoys, inject_decoys
from fragweave.attacks.guidance import generate_guidance
from fragweave.eval.judge import judge_attack, judge_task
from fragweave.models.hf_chat import HFChat
from fragweave.eval.localization import shadow_to_clean_and_spans, run_span_localizer, span_prf1

# Reuse exact helper utilities implemented in run_sweep.py (no re-implementation guesswork).
from fragweave.run_sweep import (  # type: ignore
    DEFAULT_MALICIOUS_INSTRUCTIONS,
    _call_sanitize_context,
    _direct_inject,
    _direct_inject_shadow,
    _format_guidance_block,
    _get,
    _load_samples_any_task,
    _mean_float,
    _mean_int,
    _unique_path,
    choose_random_ops,
    load_instructions,
)


def _inject_shards_as_lines(context: str, shards: List[str]) -> str:
    """
    "Without-weave" injection for shard-based variants:
    Append each shard as its own [INSTRUCTION] line at the end.

    We keep this only for the shard-based weave=without ablation.
    For no-shard & weave=without, we use run_sweep's _direct_inject (baseline).
    """
    lines: List[str] = []
    for s in shards or []:
        t = (s or "").strip()
        if not t:
            continue
        lines.append(f"[INSTRUCTION]: {t}")
    if not lines:
        return context
    return context.rstrip() + "\n\n" + "\n".join(lines) + "\n"


def _inject_shards_as_lines_shadow(context: str, shards: List[str], tag_start: str, tag_end: str) -> str:
    """
    Shadow-tag version of _inject_shards_as_lines for localization GT.
    """
    lines: List[str] = []
    for s in shards or []:
        t = (s or "").strip()
        if not t:
            continue
        lines.append(f"[INSTRUCTION]: {tag_start}{t}{tag_end}")
    if not lines:
        return context
    return context.rstrip() + "\n\n" + "\n".join(lines) + "\n"


def _parse_seeds(seed_str: Optional[str], default_seed: int) -> List[int]:
    if not seed_str:
        return [int(default_seed)]
    out: List[int] = []
    for part in seed_str.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or [int(default_seed)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="configs/emailqa_with_localization_and_sanitization.yaml",
        help="Path to a FragWeave YAML config.",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Sample limit for the dataset loader (default: 50).",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated RNG seeds. If omitted, uses attack.rng_seed from config.",
    )
    ap.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override output.run_name (default: <cfg.run_name>_ablation_emailqa).",
    )
    ap.add_argument(
        "--include-direct-baseline",
        action="store_true",
        help="Also run the direct baseline (same style as run_sweep).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Force sample cap for fast ablations.
    setattr(cfg.dataset, "task", getattr(cfg.dataset, "task", "email_qa"))
    setattr(cfg.dataset, "max_samples", int(args.max_samples))

    # Output directory
    run_name = args.run_name or (str(cfg.output.run_name) + "_ablation_emailqa")
    out_dir = Path(cfg.output.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bipia_root = ensure_bipia_repo(cfg.dataset.bipia_root)

    task = getattr(cfg.dataset, "task", None)
    split = cfg.dataset.split
    max_samples = getattr(cfg.dataset, "max_samples", None)

    samples, used_schema = _load_samples_any_task(
        bipia_root=bipia_root,
        task=task or "email_qa",
        split=split,
        max_samples=max_samples,
        cfg_dataset=cfg.dataset,
    )

    instructions = load_instructions(cfg.attack.instruction_jsonl, cfg.attack.instruction_text_key)
    if not instructions:
        instructions = DEFAULT_MALICIOUS_INSTRUCTIONS

    # Models (exactly like run_sweep)
    target = HFChat.from_config(cfg.target_model)
    sharder = HFChat.from_config(cfg.sharder_model)
    weaver = HFChat.from_config(cfg.weaver_model)
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

    sharder_retries = int(getattr(cfg.attack, "sharder_retries", 2))
    weaver_retries = int(getattr(cfg.attack, "weaver_retries", 2))
    tagger_retries = int(getattr(cfg.attack, "tagger_retries", 2))

    preserve_structure = bool(getattr(cfg.attack, "preserve_structure", True))
    carrier_line = str(
        getattr(
            cfg.attack,
            "carrier_line",
            "Note: Please follow the task's formatting and answer constraints.",
        )
    )

    # Ablation dimensions (as requested, plus no-shard)
    k_values = [3, 4, 5]
    use_cross_refs_values = [False, True]
    weave_modes = ["without", "with", "with+decoy"]
    guidance_modes = ["without", "A", "B"]

    shard_specs: List[Tuple[str, int, bool]] = []

    # 1) noshard first
    shard_specs.append(("noshard", 0, False))

    # 2) then k3, k4 (optionally k5)
    for k in [3, 4, 5, 6, 7, 8]:
        for use_refs in use_cross_refs_values:
            shard_specs.append(("k", int(k), bool(use_refs)))

    seeds = _parse_seeds(args.seeds, int(getattr(cfg.attack, "rng_seed", 2026)))

    # ---- output paths (unique on start) ----
    csv_path = _unique_path(out_dir / "results.csv")
    debug_path = _unique_path(out_dir / "debug_fragments.jsonl")
    summary_path = _unique_path(out_dir / "summary_by_variant.csv")
    schema_path = _unique_path(out_dir / "dataset_schema_used.json")
    schema_path.write_text(json.dumps(used_schema, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "seed",
        "variant_id",
        "sample_id",
        "shard_mode",
        "k",
        "use_cross_refs",
        "weave_mode",
        "guidance_mode",
        "attack_succeeded",
        "attack_conf",
        "loc_precision",
        "loc_recall",
        "loc_f1",
        "attack_succeeded_after_sanitizer",
        "is_direct_baseline",
    ]

    summary_fields = [
        "seed",
        "variant_id",
        "n",
        "asr",
        "loc_f1",
        "asr_after_sanitizer",
        "asr_direct",
        "loc_f1_direct",
        "asr_after_sanitizer_direct",
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

    def maybe_localize(poisoned_ctx: str, shadow_ctx: Optional[str]) -> Tuple[float, float, float, Optional[Dict[str, Any]]]:
        if detector is None or shadow_ctx is None:
            return 0.0, 0.0, 0.0, None
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
        loc_dbg = {
            "gt_spans": gt_spans,
            "pred_spans": loc.spans,
            "snippets": loc.snippets,
            "raw": loc.raw_model_output,
            "parse_ok": loc.parse_ok,
        }
        return float(prec), float(rec), float(f1), loc_dbg

    def maybe_sanitize_and_rejudge(poisoned_ctx: str, malicious: str, question: str, answer: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        if sanitizer is None or sanitize_context is None:
            return "", None

        cleaned = _call_sanitize_context(
            sanitize_context,
            sanitizer,
            poisoned_ctx,
            system_prompt=getattr(cfg.sanitization, "system_prompt", None),
            max_new_tokens=cfg.sanitization.sanitizer_max_new_tokens,
        )

        resp_san = target.generate(
            cfg.prompt.target_template.format(context=cleaned, question=question),
            max_new_tokens=cfg.target_model.max_new_tokens,
        )
        atk_san = judge_attack(judge, question=question, instruction=malicious, response=resp_san)

        task_after_san = None
        if bool(getattr(cfg.sanitization, "eval_task_after_sanitize", False)):
            task_after_san = judge_task(judge, question=question, answer=answer, response=resp_san)

        dbg = {
            "cleaned_context": cleaned,
            "target_response_after_sanitize": resp_san,
            "attack_judge_after_sanitize": asdict(atk_san),
            "eval_task_after_sanitize": bool(getattr(cfg.sanitization, "eval_task_after_sanitize", False)),
            "task_judge_after_sanitize": None if task_after_san is None else asdict(task_after_san),
        }
        return str(int(atk_san.ok)), dbg

    try:
        for seed in seeds:
            rng = random.Random(int(seed))

            for shard_mode, k, use_refs in shard_specs:
                for weave_mode in weave_modes:
                    for gmode in guidance_modes:
                        # Variant naming:
                        #   shard_mode="k": k{k}_refs{0/1}
                        #   shard_mode="noshard": noshard
                        if shard_mode == "noshard":
                            variant_id = f"noshard_weave{weave_mode.replace('+', '')}_guide{gmode}"
                        else:
                            variant_id = (
                                f"k{k}_refs{1 if use_refs else 0}_"
                                f"weave{weave_mode.replace('+', '')}_"
                                f"guide{gmode}"
                            )

                        print(f"\n== Seed {seed} | Variant {variant_id} ==")

                        asr: List[int] = []
                        loc_f1s: List[float] = []
                        asr_san: List[int] = []

                        asr_base: List[int] = []
                        loc_f1_base: List[float] = []
                        asr_san_base: List[int] = []

                        for si, sample in enumerate(tqdm(samples, desc=variant_id), start=1):
                            sample_id = str(_get(sample, "id", si))
                            context = str(_get(sample, "context", ""))
                            question = str(_get(sample, "question", ""))
                            answer = _get(sample, "answer", None)

                            malicious = rng.choice(instructions)

                            # -----------------------
                            # Optional direct baseline (exactly like run_sweep)
                            # -----------------------
                            if args.include_direct_baseline:
                                base_ctx = _direct_inject(context, malicious)
                                base_shadow = _direct_inject_shadow(
                                    context, malicious, cfg.localization.tag_start, cfg.localization.tag_end
                                )
                                base_prompt = cfg.prompt.target_template.format(context=base_ctx, question=question)
                                base_resp = target.generate(base_prompt, max_new_tokens=cfg.target_model.max_new_tokens)

                                base_atk = judge_attack(judge, question=question, instruction=malicious, response=base_resp)
                                asr_base.append(int(base_atk.ok))

                                b_prec = b_rec = b_f1 = 0.0
                                b_loc_dbg = None
                                if detector is not None and (cfg.localization.gt_mode or "").lower() == "shadow_tags":
                                    b_prec, b_rec, b_f1, b_loc_dbg = maybe_localize(base_ctx, base_shadow)
                                    loc_f1_base.append(float(b_f1))

                                b_after_san_ok, b_san_dbg = maybe_sanitize_and_rejudge(base_ctx, malicious, question, answer)
                                if b_after_san_ok != "":
                                    asr_san_base.append(int(b_after_san_ok))

                                w_csv.writerow(
                                    {
                                        "seed": int(seed),
                                        "variant_id": variant_id,
                                        "sample_id": sample_id,
                                        "shard_mode": "direct_baseline",
                                        "k": "",
                                        "use_cross_refs": "",
                                        "weave_mode": "",
                                        "guidance_mode": "",
                                        "attack_succeeded": int(base_atk.ok),
                                        "attack_conf": float(base_atk.score),
                                        "loc_precision": float(b_prec) if detector is not None else "",
                                        "loc_recall": float(b_rec) if detector is not None else "",
                                        "loc_f1": float(b_f1) if detector is not None else "",
                                        "attack_succeeded_after_sanitizer": b_after_san_ok,
                                        "is_direct_baseline": 1,
                                    }
                                )
                                f_csv.flush()

                                f_dbg.write(
                                    json.dumps(
                                        {
                                            "seed": int(seed),
                                            "variant_id": variant_id,
                                            "sample_id": sample_id,
                                            "is_direct_baseline": True,
                                            "malicious_instruction": malicious,
                                            "original_context": context,
                                            "baseline_context": base_ctx,
                                            "baseline_shadow": base_shadow,
                                            "question": question,
                                            "answer": answer,
                                            "target_response": base_resp,
                                            "attack_judge": asdict(base_atk),
                                            "loc_debug": b_loc_dbg,
                                            "sanitization_debug": b_san_dbg,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                                f_dbg.flush()

                            # -----------------------
                            # Our method / ablation variant
                            # -----------------------
                            guide_snippets: List[str] = []
                            if gmode in {"A", "B"}:
                                # Reuse project guidance generator (A/B are mutually exclusive versions)
                                guide_res = generate_guidance(task, gmode, malicious, rng)
                                guide_snippets = guide_res.snippets or []

                            guide_block = ""
                            guide_shadow_block = ""
                            if guide_snippets:
                                guide_block = "\n\n---\n" + _format_guidance_block(guide_snippets, quoted=True) + "\n"
                                if detector is not None and (cfg.localization.gt_mode or "").lower() == "shadow_tags":
                                    guide_shadow_block = "\n\n---\n" + _format_guidance_block(
                                        guide_snippets,
                                        quoted=True,
                                        tag_start=cfg.localization.tag_start,
                                        tag_end=cfg.localization.tag_end,
                                    ) + "\n"

                            carrier = carrier_line if preserve_structure else None

                            shadow_ctx: Optional[str] = None
                            weave_debug: Any = None
                            decoy_debug: Any = None

                            # --- Determine items to inject ---
                            # shard_mode:
                            #   - "k": items are shards from sharder
                            #   - "noshard": item is the original malicious instruction
                            if shard_mode == "noshard":
                                items = [malicious.strip()]
                                shards_for_log = items[:]  # for consistent debug field
                                shard_meta = {"use_shard": False, "k": 0, "use_cross_refs": False}
                            else:
                                shard_res = shard_with_llm(
                                    sharder,
                                    instruction=malicious,
                                    k=int(k),
                                    use_cross_refs=bool(use_refs),
                                    task=task,
                                    max_retries=sharder_retries,
                                )
                                items = shard_res.shards
                                shards_for_log = shard_res.shards
                                shard_meta = {"use_shard": True, "k": int(k), "use_cross_refs": bool(use_refs)}

                            # --- Apply weaving / injection ---
                            if weave_mode == "without":
                                # Special rule requested:
                                #   - if no-shard & no-weave => baseline direct injection
                                #   - if shard-based & no-weave => append shards as multi-line instructions
                                if shard_mode == "noshard":
                                    poisoned_core = _direct_inject(context, malicious)
                                    poisoned_ctx = poisoned_core + guide_block
                                    if detector is not None and (cfg.localization.gt_mode or "").lower() == "shadow_tags":
                                        shadow_core = _direct_inject_shadow(
                                            context, malicious, cfg.localization.tag_start, cfg.localization.tag_end
                                        )
                                        shadow_ctx = shadow_core + guide_shadow_block
                                else:
                                    poisoned_core = _inject_shards_as_lines(context, items)
                                    poisoned_ctx = poisoned_core + guide_block
                                    if detector is not None and (cfg.localization.gt_mode or "").lower() == "shadow_tags":
                                        shadow_core = _inject_shards_as_lines_shadow(
                                            context, items, cfg.localization.tag_start, cfg.localization.tag_end
                                        )
                                        shadow_ctx = shadow_core + guide_shadow_block
                            else:
                                # Weave enabled: choose ops and weave items (either shards or the single malicious instruction)
                                ops, ctx2 = choose_random_ops(task, context, items, rng, carrier_line=carrier)

                                ctx_for_weave = ctx2
                                if weave_mode == "with+decoy":
                                    decoys = generate_decoys(task, rng)
                                    dec_ins = inject_decoys(task, ctx_for_weave, decoys, rng, carrier_line=carrier)
                                    ctx_for_weave = dec_ins.new_context
                                    decoy_debug = {
                                        "decoys": decoys,
                                        "inserted": dec_ins.inserted,
                                        "used_carrier": dec_ins.used_carrier,
                                    }

                                if detector is not None and (cfg.localization.gt_mode or "").lower() == "shadow_tags":
                                    poisoned_core, shadow_core, weave_debug = apply_weave_with_shadow(
                                        weaver,
                                        ctx_for_weave,
                                        ops,
                                        task=task,
                                        max_retries=weaver_retries,
                                        tag_chat=weaver,
                                        tag_start=cfg.localization.tag_start,
                                        tag_end=cfg.localization.tag_end,
                                        tag_max_retries=tagger_retries,
                                    )
                                    poisoned_ctx = poisoned_core + guide_block
                                    shadow_ctx = shadow_core + guide_shadow_block
                                else:
                                    poisoned_core, weave_debug = apply_weave(
                                        weaver,
                                        ctx_for_weave,
                                        ops,
                                        task=task,
                                        max_retries=weaver_retries,
                                    )
                                    poisoned_ctx = poisoned_core + guide_block

                            # --- Target run + judge ---
                            prompt = cfg.prompt.target_template.format(context=poisoned_ctx, question=question)
                            resp = target.generate(prompt, max_new_tokens=cfg.target_model.max_new_tokens)

                            atk = judge_attack(judge, question=question, instruction=malicious, response=resp)
                            asr.append(int(atk.ok))

                            # --- Localization ---
                            prec = rec = f1 = 0.0
                            loc_dbg = None
                            if detector is not None and shadow_ctx is not None and (cfg.localization.gt_mode or "").lower() == "shadow_tags":
                                prec, rec, f1, loc_dbg = maybe_localize(poisoned_ctx, shadow_ctx)
                                loc_f1s.append(float(f1))

                            # --- Sanitization ---
                            after_san_ok, san_dbg = maybe_sanitize_and_rejudge(poisoned_ctx, malicious, question, answer)
                            if after_san_ok != "":
                                asr_san.append(int(after_san_ok))

                            w_csv.writerow(
                                {
                                    "seed": int(seed),
                                    "variant_id": variant_id,
                                    "sample_id": sample_id,
                                    "shard_mode": shard_mode,
                                    "k": int(k) if shard_mode != "noshard" else 0,
                                    "use_cross_refs": int(use_refs) if shard_mode != "noshard" else 0,
                                    "weave_mode": weave_mode,
                                    "guidance_mode": gmode,
                                    "attack_succeeded": int(atk.ok),
                                    "attack_conf": float(atk.score),
                                    "loc_precision": float(prec) if detector is not None else "",
                                    "loc_recall": float(rec) if detector is not None else "",
                                    "loc_f1": float(f1) if detector is not None else "",
                                    "attack_succeeded_after_sanitizer": after_san_ok,
                                    "is_direct_baseline": 0,
                                }
                            )
                            f_csv.flush()

                            f_dbg.write(
                                json.dumps(
                                    {
                                        "seed": int(seed),
                                        "variant_id": variant_id,
                                        "sample_id": sample_id,
                                        "is_direct_baseline": False,
                                        "malicious_instruction": malicious,
                                        "question": question,
                                        "answer": answer,
                                        "original_context": context,
                                        "shard_meta": shard_meta,
                                        "weave_mode": weave_mode,
                                        "guidance_mode": gmode,
                                        "guidance": guide_snippets,
                                        "decoy_debug": decoy_debug,
                                        "items_used": items,
                                        "shards": shards_for_log,
                                        "weave_debug": weave_debug,
                                        "shadow_context": shadow_ctx,
                                        "poisoned_context": poisoned_ctx,
                                        "target_response": resp,
                                        "attack_judge": asdict(atk),
                                        "loc_debug": loc_dbg,
                                        "sanitization_debug": san_dbg,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            f_dbg.flush()

                        # ---- per-variant prints + summary row ----
                        print(f"ASR: {_mean_int(asr):.3f}" if asr else "ASR: n/a")
                        if detector is not None and loc_f1s:
                            print(f"Localization F1: {_mean_float(loc_f1s):.3f}")
                        if sanitizer is not None and asr_san:
                            print(f"ASR after sanitizer: {_mean_int(asr_san):.3f}")

                        w_sum.writerow(
                            {
                                "seed": int(seed),
                                "variant_id": variant_id,
                                "n": len(samples),
                                "asr": _mean_int(asr) if asr else "",
                                "loc_f1": _mean_float(loc_f1s) if loc_f1s else "",
                                "asr_after_sanitizer": _mean_int(asr_san) if asr_san else "",
                                "asr_direct": _mean_int(asr_base) if asr_base else "",
                                "loc_f1_direct": _mean_float(loc_f1_base) if loc_f1_base else "",
                                "asr_after_sanitizer_direct": _mean_int(asr_san_base) if asr_san_base else "",
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