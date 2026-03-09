from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis_emailqa_interp.common import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DEBUG_JSONL,
    DEFAULT_MAX_PAIRED,
    DEFAULT_SEED,
    DEFAULT_VARIANT_ID,
    build_prompt,
    ensure_dir,
    find_sections,
    load_analysis_stack,
    log_line,
    pair_rows_with_diagnostics,
    read_jsonl,
    save_run_config,
    set_seed,
    split_pairs_by_mode,
    token_char_spans,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prompt-side attention visualization for EmailQA FragWeave vs baseline.")
    ap.add_argument("--debug-jsonl", default=DEFAULT_DEBUG_JSONL)
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--variant-id", default=DEFAULT_VARIANT_ID)
    ap.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRED)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--mode", choices=["all_pairs", "success_only", "both"], default="both")
    ap.add_argument("--tail-tokens", type=int, default=8)
    ap.add_argument("--out-dir", default="analysis_emailqa_interp/outputs/attention_viz")
    return ap.parse_args()


def locate_span(text: str, needle: str) -> Tuple[int, int] | None:
    if not needle:
        return None
    idx = text.find(needle)
    if idx < 0:
        return None
    return (idx, idx + len(needle))


def summarize_prompt_tail_attention(chat, prompt: str, tail_tokens: int) -> Dict[str, np.ndarray | str]:
    enc = chat.encode_prompt_states(prompt, add_generation_prompt=True, output_attentions=True)
    atts = np.stack([a.numpy()[0] for a in enc["attentions"]])
    att_lh = atts.mean(axis=1)
    tail = att_lh[:, -tail_tokens:, :].mean(axis=1)
    input_ids = enc["input_ids"][0].tolist()
    text, spans, alignment_mode = token_char_spans(chat.tokenizer, input_ids)
    return {
        "prompt_tail_to_input_attention": tail.mean(axis=0),
        "layer_prompt_tail_to_input_attention": tail,
        "prompt_text": text,
        "token_spans": np.array(spans, dtype=np.int32),
        "char_alignment_mode": alignment_mode,
    }


def section_vector(prompt_text: str, token_spans: np.ndarray, score: np.ndarray, row: Dict) -> Dict[str, float]:
    sections = find_sections(prompt_text)
    frag_spans = [locate_span(prompt_text, str(s)) for s in (row.get("shards") or [])]
    guide_spans = [locate_span(prompt_text, str(s)) for s in (row.get("guidance") or [])]
    inj_span = locate_span(prompt_text, str(row.get("malicious_instruction") or ""))
    out = {"question": 0.0, "main_context": 0.0, "injection_like": 0.0, "other": 0.0}
    for i, (st, ed) in enumerate(token_spans.tolist()):
        label = "other"
        if inj_span and st >= inj_span[0] and ed <= inj_span[1]:
            label = "injection_like"
        elif any(span and st >= span[0] and ed <= span[1] for span in frag_spans):
            label = "injection_like"
        elif any(span and st >= span[0] and ed <= span[1] for span in guide_spans):
            label = "injection_like"
        elif "question" in sections and st >= sections["question"][0] and ed <= sections["question"][1]:
            label = "question"
        elif "main_context" in sections and st >= sections["main_context"][0] and ed <= sections["main_context"][1]:
            label = "main_context"
        out[label] += float(score[i])
    total = sum(out.values()) + 1e-12
    return {k: v / total for k, v in out.items()}


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting outputs. Please install matplotlib.") from exc

    out_dir = ensure_dir(args.out_dir)
    set_seed(args.seed)
    save_run_config(out_dir, vars(args))

    rows = read_jsonl(args.debug_jsonl)
    pairs, diag = pair_rows_with_diagnostics(rows, variant_id=args.variant_id, max_pairs=args.max_pairs)
    if not pairs:
        raise RuntimeError(
            "No valid paired samples after filtering. "
            f"variant_id={args.variant_id}, rows_seen_for_variant={diag.rows_seen_for_variant}, "
            f"grouped_sample_ids={diag.grouped_sample_ids}, complete_pairs={diag.complete_pairs}, "
            f"skipped_missing_fields={diag.skipped_missing_fields}. "
            f"Available variants (top): {diag.short_variant_hint()}"
        )
    cfg, chat = load_analysis_stack(args.config)

    run_modes = ["all_pairs", "success_only"] if args.mode == "both" else [args.mode]
    for mode in run_modes:
        baseline_pairs, fragweave_pairs, mode_diag = split_pairs_by_mode(pairs, mode)
        if mode == "success_only" and (mode_diag.baseline_pairs <= 2 or mode_diag.fragweave_pairs <= 2):
            raise RuntimeError(
                "success_only mode requires >2 samples per side; "
                f"got baseline={mode_diag.baseline_pairs}, fragweave={mode_diag.fragweave_pairs}."
            )

        base_scores: List[np.ndarray] = []
        fw_scores: List[np.ndarray] = []
        base_sections: List[Dict[str, float]] = []
        fw_sections: List[Dict[str, float]] = []
        alignment_modes = {"offset_mapping": 0, "approximate": 0}

        for pair in tqdm(baseline_pairs, desc=f"Attention pairs ({mode} baseline)", unit="pair"):
            p_b = build_prompt(cfg.prompt.target_template, pair.baseline_context, pair.question)
            b = summarize_prompt_tail_attention(chat, p_b, tail_tokens=args.tail_tokens)
            base_scores.append(np.asarray(b["prompt_tail_to_input_attention"]))
            base_sections.append(
                section_vector(
                    str(b["prompt_text"]),
                    np.asarray(b["token_spans"]),
                    np.asarray(b["prompt_tail_to_input_attention"]),
                    pair.baseline_row,
                )
            )
            alignment_modes[str(b["char_alignment_mode"])] = alignment_modes.get(str(b["char_alignment_mode"]), 0) + 1

        for pair in tqdm(fragweave_pairs, desc=f"Attention pairs ({mode} fragweave)", unit="pair"):
            p_f = build_prompt(cfg.prompt.target_template, pair.fragweave_context, pair.question)
            f = summarize_prompt_tail_attention(chat, p_f, tail_tokens=args.tail_tokens)
            fw_scores.append(np.asarray(f["prompt_tail_to_input_attention"]))
            fw_sections.append(
                section_vector(
                    str(f["prompt_text"]),
                    np.asarray(f["token_spans"]),
                    np.asarray(f["prompt_tail_to_input_attention"]),
                    pair.fragweave_row,
                )
            )
            alignment_modes[str(f["char_alignment_mode"])] = alignment_modes.get(str(f["char_alignment_mode"]), 0) + 1

        min_len = min(min(len(x) for x in base_scores), min(len(x) for x in fw_scores))
        base_trim = np.stack([x[:min_len] for x in base_scores])
        fw_trim = np.stack([x[:min_len] for x in fw_scores])

        np.save(out_dir / f"prompt_attention_summary_baseline_{mode}.npy", base_trim)
        np.save(out_dir / f"prompt_attention_summary_fragweave_{mode}.npy", fw_trim)

        labels = ["main_context", "question", "injection_like", "other"]
        base_vec = np.array([np.mean([x[l] for x in base_sections]) for l in labels])
        fw_vec = np.array([np.mean([x[l] for x in fw_sections]) for l in labels])

        mode_label = "All Pairs" if mode == "all_pairs" else "Success-Only"
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, base_vec, width=w, label="baseline")
        ax.bar(x + w / 2, fw_vec, width=w, label="fragweave")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Normalized attention mass")
        ax.set_title(f"Prompt attention by section ({mode_label})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"section_attention_barplot_{mode}.png", dpi=180)
        plt.close(fig)

        mat = np.stack([base_vec, fw_vec])
        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(mat, aspect="auto", cmap="Blues")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["baseline", "fragweave"])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(f"Prompt-side attention by prompt section ({mode_label})")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"section_attention_heatmap_{mode}.png", dpi=180)
        plt.close(fig)

        elapsed_s = time.perf_counter() - start
        stats = {
            "mode": mode,
            "mode_label": mode_label,
            "mode_rule": mode_diag.rule,
            "n_pairs_total": mode_diag.total_pairs,
            "n_pairs_baseline": mode_diag.baseline_pairs,
            "n_pairs_fragweave": mode_diag.fragweave_pairs,
            "tail_prompt_tokens": args.tail_tokens,
            "pairing_diagnostics": {
                "rows_seen_for_variant": diag.rows_seen_for_variant,
                "grouped_sample_ids": diag.grouped_sample_ids,
                "complete_pairs": diag.complete_pairs,
                "used_pairs": diag.used_pairs,
                "skipped_missing_fields": diag.skipped_missing_fields,
                "variant_hint": diag.short_variant_hint(),
            },
            "char_alignment_mode_counts": alignment_modes,
            "baseline_main_context_prompt_attention": float(base_vec[0]),
            "fragweave_main_context_prompt_attention": float(fw_vec[0]),
            "elapsed_seconds": elapsed_s,
        }
        (out_dir / f"attention_stats_{mode}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        log_line(
            out_dir,
            f"Done: mode={mode} baseline_pairs={mode_diag.baseline_pairs} "
            f"fragweave_pairs={mode_diag.fragweave_pairs} elapsed_seconds={elapsed_s:.2f}",
        )


if __name__ == "__main__":
    main()
