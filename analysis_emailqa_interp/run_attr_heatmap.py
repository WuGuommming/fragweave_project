from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import csv
import json
from typing import Dict, List, Tuple

import numpy as np

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
    pair_rows,
    read_jsonl,
    save_run_config,
    set_seed,
    token_char_spans,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Attribution heatmap analysis for EmailQA FragWeave vs baseline.")
    ap.add_argument("--debug-jsonl", default=DEFAULT_DEBUG_JSONL)
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--variant-id", default=DEFAULT_VARIANT_ID)
    ap.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRED)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--method", choices=["grad_x_input", "integrated_gradients"], default="grad_x_input")
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


def region_summary(prompt_text: str, token_spans: List[Tuple[int, int]], attr: np.ndarray, row: Dict) -> Dict[str, float]:
    sections = find_sections(prompt_text)
    frag_spans = [locate_span(prompt_text, str(s)) for s in (row.get("shards") or [])]
    guide_spans = [locate_span(prompt_text, str(s)) for s in (row.get("guidance") or [])]
    inj_span = locate_span(prompt_text, str(row.get("malicious_instruction") or ""))

    out = {"question": 0.0, "main_context": 0.0, "guidance": 0.0, "injection_like": 0.0, "other": 0.0}
    for i, (st, ed) in enumerate(token_spans):
        val = float(attr[i])
        label = "other"
        if inj_span and st >= inj_span[0] and ed <= inj_span[1]:
            label = "injection_like"
        elif any(span and st >= span[0] and ed <= span[1] for span in frag_spans):
            label = "injection_like"
        elif any(span and st >= span[0] and ed <= span[1] for span in guide_spans):
            label = "guidance"
        elif "question" in sections and st >= sections["question"][0] and ed <= sections["question"][1]:
            label = "question"
        elif "main_context" in sections and st >= sections["main_context"][0] and ed <= sections["main_context"][1]:
            label = "main_context"
        out[label] += val

    total = sum(out.values()) + 1e-12
    return {k: v / total for k, v in out.items()}


def save_token_lines(path: Path, payloads: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in payloads:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting outputs. Please install matplotlib.") from exc

    out_dir = ensure_dir(args.out_dir)
    set_seed(args.seed)
    save_run_config(out_dir, vars(args))

    rows = read_jsonl(args.debug_jsonl)
    pairs = pair_rows(rows, variant_id=args.variant_id, max_pairs=args.max_pairs)
    cfg, chat = load_analysis_stack(args.config)

    base_dump: List[Dict] = []
    fw_dump: List[Dict] = []
    span_rows: List[Dict] = []

    for i, pair in enumerate(pairs):
        target = pair.malicious_instruction or "Follow the malicious instruction."
        base_prompt = build_prompt(cfg.prompt.target_template, pair.baseline_context, pair.question)
        fw_prompt = build_prompt(cfg.prompt.target_template, pair.fragweave_context, pair.question)

        b_attr = chat.compute_input_attribution(base_prompt, target, method=args.method)
        f_attr = chat.compute_input_attribution(fw_prompt, target, method=args.method)

        b_ids = b_attr["input_ids"][0].tolist()
        f_ids = f_attr["input_ids"][0].tolist()
        b_text, b_spans = token_char_spans(chat.tokenizer, b_ids)
        f_text, f_spans = token_char_spans(chat.tokenizer, f_ids)

        b_norm = normalize_attr(np.asarray(b_attr["token_attribution"].tolist(), dtype=np.float32))
        f_norm = normalize_attr(np.asarray(f_attr["token_attribution"].tolist(), dtype=np.float32))

        b_regions = region_summary(b_text, b_spans, b_norm, pair.baseline_row)
        f_regions = region_summary(f_text, f_spans, f_norm, pair.fragweave_row)

        base_dump.append({"sample_id": pair.sample_id, "tokens": b_attr["token_text"], "attr": b_norm.tolist()})
        fw_dump.append({"sample_id": pair.sample_id, "tokens": f_attr["token_text"], "attr": f_norm.tolist()})

        row = {"sample_id": pair.sample_id}
        for k in sorted(set(b_regions) | set(f_regions)):
            row[f"baseline_{k}"] = b_regions.get(k, 0.0)
            row[f"fragweave_{k}"] = f_regions.get(k, 0.0)
        span_rows.append(row)

        if i < 3:
            plot_example(
                out_dir / f"heatmap_token_example_{pair.sample_id}_baseline.png",
                b_attr["token_text"],
                b_norm,
                f"Baseline token attribution ({pair.sample_id})",
            )
            plot_example(
                out_dir / f"heatmap_token_example_{pair.sample_id}_fragweave.png",
                f_attr["token_text"],
                f_norm,
                f"FragWeave token attribution ({pair.sample_id})",
            )

    save_token_lines(out_dir / "attribution_tokens_baseline.jsonl", base_dump)
    save_token_lines(out_dir / "attribution_tokens_fragweave.jsonl", fw_dump)

    keys = sorted(k for k in span_rows[0].keys() if k != "sample_id")
    with (out_dir / "span_attr_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", *keys])
        writer.writeheader()
        writer.writerows(span_rows)

    baseline_cols = [k for k in keys if k.startswith("baseline_")]
    frag_cols = [k for k in keys if k.startswith("fragweave_")]
    base_mean = np.array([np.mean([r[k] for r in span_rows]) for k in baseline_cols])
    fw_mean = np.array([np.mean([r[k] for r in span_rows]) for k in frag_cols])
    labels = [k.replace("baseline_", "") for k in baseline_cols]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, base_mean, width=w, label="baseline")
    ax.bar(x + w / 2, fw_mean, width=w, label="fragweave")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Normalized attribution")
    ax.set_title("Span-level attribution summary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_span_summary.png", dpi=180)
    plt.close(fig)

    stats = {
        "n_pairs": len(pairs),
        "method": args.method,
        "baseline_injection_like_mean": float(np.mean([r.get("baseline_injection_like", 0.0) for r in span_rows])),
        "fragweave_injection_like_mean": float(np.mean([r.get("fragweave_injection_like", 0.0) for r in span_rows])),
    }
    (out_dir / "attribution_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log_line(out_dir, f"Done: n_pairs={len(pairs)} method={args.method}")


if __name__ == "__main__":
    main()
