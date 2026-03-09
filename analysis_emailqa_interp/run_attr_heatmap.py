from __future__ import annotations

import argparse
import csv
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
    attack_anchor_texts,
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
    select_benign_carrier_text,
    set_seed,
    split_pairs_by_mode,
    token_char_spans,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Attribution heatmap analysis for EmailQA FragWeave vs baseline.")
    ap.add_argument("--debug-jsonl", default=DEFAULT_DEBUG_JSONL)
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--variant-id", default=DEFAULT_VARIANT_ID)
    ap.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRED)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--mode", choices=["all_pairs", "success_only", "both"], default="both")
    ap.add_argument("--method", choices=["grad_x_input", "integrated_gradients"], default="grad_x_input")
    ap.add_argument("--token-examples", type=int, default=0)
    ap.add_argument("--out-dir", default="analysis_emailqa_interp/outputs/attr_heatmap")
    return ap.parse_args()


def normalize_attr(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    return x / (x.sum() + 1e-12)


def locate_span(text: str, needle: str) -> Tuple[int, int] | None:
    if not needle:
        return None
    idx = text.find(needle)
    if idx < 0:
        return None
    return (idx, idx + len(needle))


def choose_target_text(pair) -> Tuple[str, str]:
    target_response = str(pair.fragweave_row.get("target_response") or "").strip()
    malicious_instruction = str(pair.malicious_instruction or "").strip()
    if target_response:
        return target_response, "target_response"
    if malicious_instruction:
        return malicious_instruction, "malicious_instruction"
    return "Please produce a concise answer using only the email content.", "fallback"


def region_summary(prompt_text: str, token_spans: List[Tuple[int, int]], attr: np.ndarray, row: Dict) -> Dict[str, float]:
    sections = find_sections(prompt_text)
    attack_spans = [locate_span(prompt_text, s) for s in attack_anchor_texts(row)]

    out = {"question": 0.0, "main_context": 0.0, "injection_like": 0.0, "other": 0.0}
    for i, (st, ed) in enumerate(token_spans):
        val = float(attr[i])
        label = "other"
        if any(span and st >= span[0] and ed <= span[1] for span in attack_spans):
            label = "injection_like"
        elif "question" in sections and st >= sections["question"][0] and ed <= sections["question"][1]:
            label = "question"
        elif "main_context" in sections and st >= sections["main_context"][0] and ed <= sections["main_context"][1]:
            label = "main_context"
        out[label] += val

    total = sum(out.values()) + 1e-12
    return {k: v / total for k, v in out.items()}


def span_summary(
    prompt_text: str,
    token_spans: List[Tuple[int, int]],
    attr: np.ndarray,
    row: Dict,
    benign_carrier_text: str,
) -> Dict[str, float]:
    attack_spans = [locate_span(prompt_text, s) for s in attack_anchor_texts(row)]
    woven_spans = [locate_span(prompt_text, str(s)) for s in (row.get("shards") or [])]
    carrier_span = locate_span(prompt_text, benign_carrier_text)
    is_baseline = bool(row.get("is_direct_baseline"))

    out = {
        "baseline_injection_span": 0.0,
        "fragweave_woven_span": 0.0,
        "benign_carrier_span": 0.0,
        "other": 0.0,
    }

    for i, (st, ed) in enumerate(token_spans):
        val = float(attr[i])
        label = "other"

        if is_baseline:
            if any(span and st >= span[0] and ed <= span[1] for span in attack_spans):
                label = "baseline_injection_span"
            elif carrier_span and st >= carrier_span[0] and ed <= carrier_span[1]:
                label = "benign_carrier_span"
        else:
            if any(span and st >= span[0] and ed <= span[1] for span in woven_spans):
                label = "fragweave_woven_span"
            elif carrier_span and st >= carrier_span[0] and ed <= carrier_span[1]:
                label = "benign_carrier_span"

        out[label] += val

    total = sum(out.values()) + 1e-12
    return {k: v / total for k, v in out.items()}


def save_token_lines(path: Path, payloads: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in payloads:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def concentration_metrics(region_dist: Dict[str, float], top_k: int = 2) -> Dict[str, float]:
    values = np.asarray(sorted(region_dist.values(), reverse=True), dtype=np.float64)
    values = values / (values.sum() + 1e-12)
    entropy = -float(np.sum(values * np.log(values + 1e-12)) / np.log(len(values) + 1e-12))
    return {
        "top2_concentration": float(values[:top_k].sum()),
        "section_entropy": entropy,
        "outside_injection_like": float(1.0 - region_dist.get("injection_like", 0.0)),
        "hhi": float(np.sum(values**2)),
    }


def span_concentration_metrics(region_dist: Dict[str, float], top_k: int = 2) -> Dict[str, float]:
    values = np.asarray(sorted(region_dist.values(), reverse=True), dtype=np.float64)
    values = values / (values.sum() + 1e-12)
    entropy = -float(np.sum(values * np.log(values + 1e-12)) / np.log(len(values) + 1e-12))
    return {
        "top2_concentration": float(values[:top_k].sum()),
        "span_entropy": entropy,
        "hhi": float(np.sum(values**2)),
    }


def plot_example(path: Path, tokens: List[str], attr: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    n = min(80, len(tokens))
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.imshow(attr[:n][None, :], aspect="auto", cmap="viridis")
    step = max(1, n // 12)
    xt = list(range(0, n, step))
    ax.set_xticks(xt)
    ax.set_xticklabels([tokens[i][:10] for i in xt], rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    try:
        import matplotlib.pyplot as plt  # noqa: F401
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
        if mode == "success_only" and (mode_diag.baseline_pairs < 2 or mode_diag.fragweave_pairs < 2):
            raise RuntimeError(
                "success_only mode requires >2 samples per side; "
                f"got baseline={mode_diag.baseline_pairs}, fragweave={mode_diag.fragweave_pairs}."
            )

        base_dump: List[Dict] = []
        fw_dump: List[Dict] = []
        section_rows: List[Dict] = []
        span_rows: List[Dict] = []
        target_source_counts = {"target_response": 0, "malicious_instruction": 0, "fallback": 0}
        alignment_modes = {"offset_mapping": 0, "approximate": 0}

        for i, pair in enumerate(tqdm(baseline_pairs, desc=f"Attribution pairs ({mode} baseline)", unit="pair")):
            target, target_source = choose_target_text(pair)
            target_source_counts[target_source] += 1
            base_prompt = build_prompt(cfg.prompt.target_template, pair.baseline_context, pair.question)
            b_attr = chat.compute_input_attribution(base_prompt, target, method=args.method)
            b_ids = b_attr["input_ids"][0].tolist()
            b_text, b_spans, b_alignment = token_char_spans(chat.tokenizer, b_ids)
            alignment_modes[b_alignment] = alignment_modes.get(b_alignment, 0) + 1
            b_norm = normalize_attr(np.asarray(b_attr["token_attribution"].tolist(), dtype=np.float32))
            b_regions = region_summary(b_text, b_spans, b_norm, pair.baseline_row)
            benign_carrier_text = select_benign_carrier_text(pair.baseline_context, pair.fragweave_context)
            b_span_regions = span_summary(b_text, b_spans, b_norm, pair.baseline_row, benign_carrier_text)

            base_dump.append(
                {
                    "sample_id": pair.sample_id,
                    "tokens": b_attr["token_text"],
                    "attr": b_norm.tolist(),
                    "target_text": target,
                    "target_source": target_source,
                    "char_alignment_mode": b_alignment,
                    "mode": mode,
                }
            )
            section_rows.append(
                {
                    "sample_id": pair.sample_id,
                    "target_source": target_source,
                    **{f"baseline_{k}": b_regions.get(k, 0.0) for k in b_regions},
                }
            )
            span_rows.append(
                {
                    "sample_id": pair.sample_id,
                    "target_source": target_source,
                    **{f"baseline_{k}": b_span_regions.get(k, 0.0) for k in b_span_regions},
                }
            )
            if i < args.token_examples:
                mode_label = "All Pairs" if mode == "all_pairs" else "Success-Only"
                plot_example(
                    out_dir / f"heatmap_token_example_{pair.sample_id}_baseline_{mode}.png",
                    b_attr["token_text"],
                    b_norm,
                    f"Baseline token attribution ({pair.sample_id}, {mode_label})",
                )

        for i, pair in enumerate(tqdm(fragweave_pairs, desc=f"Attribution pairs ({mode} fragweave)", unit="pair")):
            target, target_source = choose_target_text(pair)
            target_source_counts[target_source] += 1
            fw_prompt = build_prompt(cfg.prompt.target_template, pair.fragweave_context, pair.question)
            f_attr = chat.compute_input_attribution(fw_prompt, target, method=args.method)
            f_ids = f_attr["input_ids"][0].tolist()
            f_text, f_spans, f_alignment = token_char_spans(chat.tokenizer, f_ids)
            alignment_modes[f_alignment] = alignment_modes.get(f_alignment, 0) + 1
            f_norm = normalize_attr(np.asarray(f_attr["token_attribution"].tolist(), dtype=np.float32))
            f_regions = region_summary(f_text, f_spans, f_norm, pair.fragweave_row)
            benign_carrier_text = select_benign_carrier_text(pair.baseline_context, pair.fragweave_context)
            f_span_regions = span_summary(f_text, f_spans, f_norm, pair.fragweave_row, benign_carrier_text)

            fw_dump.append(
                {
                    "sample_id": pair.sample_id,
                    "tokens": f_attr["token_text"],
                    "attr": f_norm.tolist(),
                    "target_text": target,
                    "target_source": target_source,
                    "char_alignment_mode": f_alignment,
                    "mode": mode,
                }
            )

            existing = next((r for r in section_rows if r["sample_id"] == pair.sample_id), None)
            if existing is None:
                existing = {"sample_id": pair.sample_id, "target_source": target_source}
                section_rows.append(existing)
            for k, v in f_regions.items():
                existing[f"fragweave_{k}"] = v

            existing_span = next((r for r in span_rows if r["sample_id"] == pair.sample_id), None)
            if existing_span is None:
                existing_span = {"sample_id": pair.sample_id, "target_source": target_source}
                span_rows.append(existing_span)
            for k, v in f_span_regions.items():
                existing_span[f"fragweave_{k}"] = v

            if i < args.token_examples:
                mode_label = "All Pairs" if mode == "all_pairs" else "Success-Only"
                plot_example(
                    out_dir / f"heatmap_token_example_{pair.sample_id}_fragweave_{mode}.png",
                    f_attr["token_text"],
                    f_norm,
                    f"FragWeave token attribution ({pair.sample_id}, {mode_label})",
                )

        save_token_lines(out_dir / f"attribution_tokens_baseline_{mode}.jsonl", base_dump)
        save_token_lines(out_dir / f"attribution_tokens_fragweave_{mode}.jsonl", fw_dump)

        section_keys = sorted(k for k in section_rows[0].keys() if k != "sample_id")
        with (out_dir / f"section_attr_summary_{mode}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", *section_keys])
            writer.writeheader()
            writer.writerows(section_rows)

        span_keys = sorted(k for k in span_rows[0].keys() if k != "sample_id")
        with (out_dir / f"span_attr_summary_{mode}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", *span_keys])
            writer.writeheader()
            writer.writerows(span_rows)

        baseline_section_cols = [k for k in section_keys if k.startswith("baseline_")]
        frag_section_cols = [k for k in section_keys if k.startswith("fragweave_")]
        base_section_mean = np.array([np.mean([r.get(k, 0.0) for r in section_rows]) for k in baseline_section_cols])
        fw_section_mean = np.array([np.mean([r.get(k, 0.0) for r in section_rows]) for k in frag_section_cols])
        section_labels = [k.replace("baseline_", "") for k in baseline_section_cols]

        mode_label = "All Pairs" if mode == "all_pairs" else "Success-Only"
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(section_labels))
        w = 0.35
        ax.bar(x - w / 2, base_section_mean, width=w, label="baseline")
        ax.bar(x + w / 2, fw_section_mean, width=w, label="fragweave")
        ax.set_xticks(x)
        ax.set_xticklabels(section_labels, rotation=30, ha="right")
        ax.set_ylabel("Normalized attribution")
        ax.set_title(f"Section-level attribution summary ({mode_label})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"section_attr_barplot_{mode}.png", dpi=180)
        plt.close(fig)

        baseline_span_cols = [k for k in span_keys if k.startswith("baseline_")]
        frag_span_cols = [k for k in span_keys if k.startswith("fragweave_")]
        base_span_mean = np.array([np.mean([r.get(k, 0.0) for r in span_rows]) for k in baseline_span_cols])
        fw_span_mean = np.array([np.mean([r.get(k, 0.0) for r in span_rows]) for k in frag_span_cols])
        span_labels = [k.replace("baseline_", "") for k in baseline_span_cols]

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(span_labels))
        w = 0.35
        ax.bar(x - w / 2, base_span_mean, width=w, label="baseline")
        ax.bar(x + w / 2, fw_span_mean, width=w, label="fragweave")
        ax.set_xticks(x)
        ax.set_xticklabels(span_labels, rotation=30, ha="right")
        ax.set_ylabel("Normalized attribution")
        ax.set_title(f"Span-level attribution summary ({mode_label})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"span_attr_barplot_{mode}.png", dpi=180)
        plt.close(fig)

        base_metrics = [
            concentration_metrics({k.replace("baseline_", ""): r.get(k, 0.0) for k in baseline_section_cols})
            for r in section_rows
        ]
        fw_metrics = [
            concentration_metrics({k.replace("fragweave_", ""): r.get(k, 0.0) for k in frag_section_cols})
            for r in section_rows
        ]
        metric_labels = ["top2_concentration", "section_entropy", "outside_injection_like", "hhi"]
        base_metric_mean = np.array([np.mean([m[k] for m in base_metrics]) for k in metric_labels])
        fw_metric_mean = np.array([np.mean([m[k] for m in fw_metrics]) for k in metric_labels])

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(metric_labels))
        w = 0.35
        ax.bar(x - w / 2, base_metric_mean, width=w, label="baseline")
        ax.bar(x + w / 2, fw_metric_mean, width=w, label="fragweave")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=20, ha="right")
        ax.set_ylabel("Metric value")
        ax.set_title(f"Section attribution concentration and dispersion ({mode_label})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"attribution_concentration_plot_{mode}.png", dpi=180)
        plt.close(fig)

        span_metric_labels = ["top2_concentration", "span_entropy", "hhi"]
        base_span_metrics = [
            span_concentration_metrics({k.replace("baseline_", ""): r.get(k, 0.0) for k in baseline_span_cols})
            for r in span_rows
        ]
        fw_span_metrics = [
            span_concentration_metrics({k.replace("fragweave_", ""): r.get(k, 0.0) for k in frag_span_cols})
            for r in span_rows
        ]

        elapsed_s = time.perf_counter() - start
        stats = {
            "mode": mode,
            "mode_label": mode_label,
            "mode_rule": mode_diag.rule,
            "attack_region_rule": (
                "attack_anchor_texts(row): shards/guidance/loc_debug.snippets, "
                "with malicious_instruction only as fallback"
            ),
            "n_pairs_total": mode_diag.total_pairs,
            "n_pairs_baseline": mode_diag.baseline_pairs,
            "n_pairs_fragweave": mode_diag.fragweave_pairs,
            "method": args.method,
            "pairing_diagnostics": {
                "rows_seen_for_variant": diag.rows_seen_for_variant,
                "grouped_sample_ids": diag.grouped_sample_ids,
                "complete_pairs": diag.complete_pairs,
                "used_pairs": diag.used_pairs,
                "skipped_missing_fields": diag.skipped_missing_fields,
                "variant_hint": diag.short_variant_hint(),
            },
            "target_source_counts": target_source_counts,
            "char_alignment_mode_counts": alignment_modes,
            "section_metric_means": {
                "baseline": {k: float(np.mean([m[k] for m in base_metrics])) for k in metric_labels},
                "fragweave": {k: float(np.mean([m[k] for m in fw_metrics])) for k in metric_labels},
            },
            "span_metric_means": {
                "baseline": {k: float(np.mean([m[k] for m in base_span_metrics])) for k in span_metric_labels},
                "fragweave": {k: float(np.mean([m[k] for m in fw_span_metrics])) for k in span_metric_labels},
            },
            "elapsed_seconds": elapsed_s,
        }
        (out_dir / f"attr_stats_{mode}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        log_line(
            out_dir,
            f"Done: mode={mode} baseline_pairs={mode_diag.baseline_pairs} "
            f"fragweave_pairs={mode_diag.fragweave_pairs} method={args.method} elapsed_seconds={elapsed_s:.2f}",
        )


if __name__ == "__main__":
    main()
