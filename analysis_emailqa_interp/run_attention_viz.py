from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from typing import Dict, List

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
    ap = argparse.ArgumentParser(description="Attention visualization for EmailQA FragWeave vs baseline.")
    ap.add_argument("--debug-jsonl", default=DEFAULT_DEBUG_JSONL)
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--variant-id", default=DEFAULT_VARIANT_ID)
    ap.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRED)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--tail-tokens", type=int, default=8)
    ap.add_argument("--out-dir", default="analysis_emailqa_interp/outputs/attention_viz")
    return ap.parse_args()


def summarize_attention(chat, prompt: str, tail_tokens: int) -> Dict[str, np.ndarray]:
    enc = chat.encode_prompt_states(prompt, add_generation_prompt=True, output_attentions=True)
    atts = np.stack([a.numpy()[0] for a in enc["attentions"]])  # [L, H, T, T]
    att_lh = atts.mean(axis=1)
    tail = att_lh[:, -tail_tokens:, :].mean(axis=1)
    input_ids = enc["input_ids"][0].tolist()
    text, spans = token_char_spans(chat.tokenizer, input_ids)
    return {
        "tail_query_to_input": tail.mean(axis=0),
        "layer_tail_query_to_input": tail,
        "prompt_text": text,
        "token_spans": np.array(spans, dtype=np.int32),
        "n_layers": np.array([atts.shape[0]], dtype=np.int32),
    }


def section_vector(prompt_text: str, token_spans: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    sections = find_sections(prompt_text)
    out = {"question": 0.0, "main_context": 0.0, "other": 0.0}
    for i, (st, ed) in enumerate(token_spans.tolist()):
        label = "other"
        if "question" in sections and st >= sections["question"][0] and ed <= sections["question"][1]:
            label = "question"
        elif "main_context" in sections and st >= sections["main_context"][0] and ed <= sections["main_context"][1]:
            label = "main_context"
        out[label] += float(score[i])
    total = sum(out.values()) + 1e-12
    return {k: v / total for k, v in out.items()}


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

    base_scores: List[np.ndarray] = []
    fw_scores: List[np.ndarray] = []
    base_sections: List[Dict[str, float]] = []
    fw_sections: List[Dict[str, float]] = []

    for pair in pairs:
        p_b = build_prompt(cfg.prompt.target_template, pair.baseline_context, pair.question)
        p_f = build_prompt(cfg.prompt.target_template, pair.fragweave_context, pair.question)
        b = summarize_attention(chat, p_b, tail_tokens=args.tail_tokens)
        f = summarize_attention(chat, p_f, tail_tokens=args.tail_tokens)

        base_scores.append(b["tail_query_to_input"])
        fw_scores.append(f["tail_query_to_input"])
        base_sections.append(section_vector(b["prompt_text"], b["token_spans"], b["tail_query_to_input"]))
        fw_sections.append(section_vector(f["prompt_text"], f["token_spans"], f["tail_query_to_input"]))

    min_len = min(min(len(x) for x in base_scores), min(len(x) for x in fw_scores))
    base_trim = np.stack([x[:min_len] for x in base_scores])
    fw_trim = np.stack([x[:min_len] for x in fw_scores])

    np.save(out_dir / "attention_summary_baseline.npy", base_trim)
    np.save(out_dir / "attention_summary_fragweave.npy", fw_trim)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(base_trim.mean(axis=0), label="baseline")
    ax.plot(fw_trim.mean(axis=0), label="fragweave")
    ax.set_title("Tail-query to input attention (mean)")
    ax.set_xlabel("Token index (trimmed)")
    ax.set_ylabel("Attention mass")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "attention_tail_query_plot.png", dpi=180)
    plt.close(fig)

    labels = ["main_context", "question", "other"]
    base_vec = np.array([np.mean([x[l] for x in base_sections]) for l in labels])
    fw_vec = np.array([np.mean([x[l] for x in fw_sections]) for l in labels])

    mat = np.stack([base_vec, fw_vec])
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["baseline", "fragweave"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Span-level attention summary")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "attention_span_heatmap.png", dpi=180)
    plt.close(fig)

    stats = {
        "n_pairs": len(pairs),
        "tail_tokens": args.tail_tokens,
        "baseline_main_context_attention": float(base_vec[0]),
        "fragweave_main_context_attention": float(fw_vec[0]),
    }
    (out_dir / "attention_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log_line(out_dir, f"Done: n_pairs={len(pairs)} tail_tokens={args.tail_tokens}")


if __name__ == "__main__":
    main()
