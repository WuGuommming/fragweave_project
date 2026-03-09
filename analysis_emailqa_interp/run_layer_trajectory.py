from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

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
    load_analysis_stack,
    log_line,
    pair_rows_with_diagnostics,
    read_jsonl,
    save_run_config,
    set_seed,
    to_numpy,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Layer-wise representation trajectory analysis.")
    ap.add_argument("--debug-jsonl", default=DEFAULT_DEBUG_JSONL)
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--variant-id", default=DEFAULT_VARIANT_ID)
    ap.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRED)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default="analysis_emailqa_interp/outputs/layer_trajectory")
    return ap.parse_args()


def encode_layers(chat, prompt: str) -> np.ndarray:
    enc = chat.encode_prompt_states(prompt, add_generation_prompt=True, output_attentions=False)
    pooled = []
    for h in enc["hidden_states"]:
        pooled.append(to_numpy(h[0, -1]))
    return np.stack(pooled)


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

    all_orig, all_base, all_fw = [], [], []
    for pair in tqdm(pairs, desc="Layer trajectory pairs", unit="pair"):
        p_o = build_prompt(cfg.prompt.target_template, pair.original_context, pair.question)
        p_b = build_prompt(cfg.prompt.target_template, pair.baseline_context, pair.question)
        p_f = build_prompt(cfg.prompt.target_template, pair.fragweave_context, pair.question)
        all_orig.append(encode_layers(chat, p_o))
        all_base.append(encode_layers(chat, p_b))
        all_fw.append(encode_layers(chat, p_f))

    orig = np.stack(all_orig)
    base = np.stack(all_base)
    fw = np.stack(all_fw)

    np.savez(out_dir / "layer_features.npz", original=orig, baseline=base, fragweave=fw)

    base_dist = np.linalg.norm(base - orig, axis=-1)
    fw_dist = np.linalg.norm(fw - orig, axis=-1)
    layers = list(range(base_dist.shape[1]))

    with (out_dir / "layer_trajectory_distances.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "baseline_mean_distance", "fragweave_mean_distance"],
        )
        writer.writeheader()
        for l in layers:
            writer.writerow(
                {
                    "layer": l,
                    "baseline_mean_distance": float(base_dist[:, l].mean()),
                    "fragweave_mean_distance": float(fw_dist[:, l].mean()),
                }
            )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, base_dist.mean(axis=0), label="baseline->original")
    ax.plot(layers, fw_dist.mean(axis=0), label="fragweave->original")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 distance")
    ax.set_title("Layer-wise distance trajectory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "layer_trajectory_plot.png", dpi=180)
    plt.close(fig)

    heat = np.stack([base_dist.mean(axis=0), fw_dist.mean(axis=0)])
    fig, ax = plt.subplots(figsize=(7, 2.6))
    im = ax.imshow(heat, aspect="auto", cmap="magma")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["baseline", "fragweave"])
    ax.set_xlabel("Layer")
    ax.set_title("Mean distance-to-original by layer")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "layer_heatmap.png", dpi=180)
    plt.close(fig)

    elapsed_s = time.perf_counter() - start
    stats = {
        "n_pairs": len(pairs),
        "n_layers": int(base_dist.shape[1]),
        "baseline_overall_mean_distance": float(base_dist.mean()),
        "fragweave_overall_mean_distance": float(fw_dist.mean()),
        "pairing_diagnostics": {
            "rows_seen_for_variant": diag.rows_seen_for_variant,
            "grouped_sample_ids": diag.grouped_sample_ids,
            "complete_pairs": diag.complete_pairs,
            "used_pairs": diag.used_pairs,
            "skipped_missing_fields": diag.skipped_missing_fields,
            "variant_hint": diag.short_variant_hint(),
        },
        "elapsed_seconds": elapsed_s,
    }
    (out_dir / "layer_trajectory_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log_line(out_dir, f"Done: n_pairs={len(pairs)} n_layers={base_dist.shape[1]} elapsed_seconds={elapsed_s:.2f}")


if __name__ == "__main__":
    main()
