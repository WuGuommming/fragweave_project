from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import csv
import json

import numpy as np

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
    pair_rows,
    read_jsonl,
    save_run_config,
    set_seed,
    to_numpy,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prompt-side representation analysis for EmailQA FragWeave.")
    ap.add_argument("--debug-jsonl", default=DEFAULT_DEBUG_JSONL)
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--variant-id", default=DEFAULT_VARIANT_ID)
    ap.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRED)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default="analysis_emailqa_interp/outputs/repr_prompt")
    return ap.parse_args()


def pooled_features(chat, prompt: str) -> dict:
    enc = chat.encode_prompt_states(prompt, add_generation_prompt=True, output_attentions=False)
    hs = enc["hidden_states"]
    last = hs[-1][0]
    return {
        "last_token": to_numpy(last[-1]),
        "mean_pool": to_numpy(last.mean(dim=0)),
        "n_tokens": int(last.shape[0]),
    }


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

    labels = []
    feats = []
    meta_rows = []

    for pair in pairs:
        prompts = {
            "original": build_prompt(cfg.prompt.target_template, pair.original_context, pair.question),
            "baseline": build_prompt(cfg.prompt.target_template, pair.baseline_context, pair.question),
            "fragweave": build_prompt(cfg.prompt.target_template, pair.fragweave_context, pair.question),
        }
        for category, prompt in prompts.items():
            info = pooled_features(chat, prompt)
            labels.append(category)
            feats.append(info["last_token"])
            meta_rows.append(
                {
                    "sample_id": pair.sample_id,
                    "category": category,
                    "n_tokens": info["n_tokens"],
                }
            )

    X = np.stack(feats)
    y = np.array(labels)
    np.savez(out_dir / "features_prompt_repr.npz", features=X, labels=y)

    with (out_dir / "metadata_prompt_repr.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "category", "n_tokens"])
        writer.writeheader()
        writer.writerows(meta_rows)

    X2 = X[:, :2] if X.shape[1] == 2 else None
    if X2 is None:
        try:
            from sklearn.decomposition import PCA

            X2 = PCA(n_components=2, random_state=args.seed).fit_transform(X)
        except Exception:
            X2 = X[:, :2]

    fig, ax = plt.subplots(figsize=(6, 5))
    for cat in ["original", "baseline", "fragweave"]:
        idx = np.where(y == cat)[0]
        ax.scatter(X2[idx, 0], X2[idx, 1], label=cat, s=20)
    ax.set_title("Prompt representation projection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "umap_or_pca_prompt_repr.png", dpi=180)
    plt.close(fig)

    centroids = {cat: X[y == cat].mean(axis=0) for cat in ["original", "baseline", "fragweave"]}
    cats = list(centroids.keys())
    sim = np.zeros((len(cats), len(cats)), dtype=np.float32)
    for i, c1 in enumerate(cats):
        for j, c2 in enumerate(cats):
            a = centroids[c1]
            b = centroids[c2]
            sim[i, j] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cats)), cats, rotation=30)
    ax.set_yticks(range(len(cats)), cats)
    ax.set_title("Centroid cosine similarity")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "similarity_matrix_prompt_repr.png", dpi=180)
    plt.close(fig)

    dist_b = float(np.linalg.norm(centroids["baseline"] - centroids["original"]))
    dist_f = float(np.linalg.norm(centroids["fragweave"] - centroids["original"]))
    stats = {
        "n_pairs": len(pairs),
        "n_points": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "distance_baseline_to_original": dist_b,
        "distance_fragweave_to_original": dist_f,
        "fragweave_minus_baseline_distance": dist_f - dist_b,
    }
    (out_dir / "prompt_repr_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log_line(out_dir, f"Done: n_pairs={len(pairs)}, feature_dim={X.shape[1]}")


if __name__ == "__main__":
    main()
