from __future__ import annotations

import argparse
import csv
import json
import re
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
    load_analysis_stack,
    log_line,
    pair_rows_with_diagnostics,
    read_jsonl,
    save_run_config,
    set_seed,
    to_numpy,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Local span representation analysis for EmailQA FragWeave.")
    ap.add_argument("--debug-jsonl", default=DEFAULT_DEBUG_JSONL)
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--variant-id", default=DEFAULT_VARIANT_ID)
    ap.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRED)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default="analysis_emailqa_interp/outputs/repr_prompt")
    return ap.parse_args()


def _find_span(text: str, needle: str) -> Tuple[int, int] | None:
    if not text or not needle:
        return None
    idx = text.find(needle)
    return (idx, idx + len(needle)) if idx >= 0 else None


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"[.!?]\s+", text):
        end = m.end()
        out.append((start, end))
        start = end
    if start < len(text):
        out.append((start, len(text)))
    return [x for x in out if x[1] - x[0] > 16]


def _extract_local_spans(pair) -> List[Tuple[str, str]]:
    spans: List[Tuple[str, str]] = []
    if pair.malicious_instruction:
        spans.append(("baseline_injection", pair.malicious_instruction))
    for shard in (pair.fragweave_row.get("shards") or [])[:3]:
        if shard:
            spans.append(("fragweave_woven", str(shard)))
    base_sentences = _sentence_spans(pair.baseline_context)
    frag_sentences = _sentence_spans(pair.fragweave_context)
    if base_sentences:
        st, ed = base_sentences[min(1, len(base_sentences) - 1)]
        spans.append(("benign_carrier", pair.baseline_context[st:ed]))
    elif frag_sentences:
        st, ed = frag_sentences[min(1, len(frag_sentences) - 1)]
        spans.append(("benign_carrier", pair.fragweave_context[st:ed]))
    if pair.question:
        spans.append(("question_span", pair.question))
    return spans


def span_feature(chat, prompt: str, span_text: str) -> np.ndarray:
    enc = chat.encode_prompt_states(prompt, add_generation_prompt=True, output_attentions=False)
    ids = enc["input_ids"][0].tolist()
    decoded = chat.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    hs = enc["hidden_states"][-1][0]
    hit = _find_span(decoded, span_text.strip())
    if hit is None:
        return to_numpy(hs.mean(dim=0))

    token_ids = chat.tokenizer(decoded, add_special_tokens=False, return_offsets_mapping=True)
    offsets = token_ids.get("offset_mapping", [])
    selected = [i for i, (s, e) in enumerate(offsets) if int(s) >= hit[0] and int(e) <= hit[1]]
    if not selected:
        return to_numpy(hs.mean(dim=0))
    return to_numpy(hs[selected].mean(dim=0))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


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

    labels: List[str] = []
    feats: List[np.ndarray] = []
    meta_rows: List[Dict[str, str | int]] = []

    for pair in tqdm(pairs, desc="Local span representation", unit="pair"):
        prompt_map = {
            "baseline": build_prompt(cfg.prompt.target_template, pair.baseline_context, pair.question),
            "fragweave": build_prompt(cfg.prompt.target_template, pair.fragweave_context, pair.question),
        }
        for span_label, span_text in _extract_local_spans(pair):
            prompt_key = "fragweave" if span_label == "fragweave_woven" else "baseline"
            feat = span_feature(chat, prompt_map[prompt_key], span_text)
            labels.append(span_label)
            feats.append(feat)
            meta_rows.append({"sample_id": pair.sample_id, "span_label": span_label, "prompt_side": prompt_key})

    X = np.stack(feats)
    y = np.array(labels)
    np.savez(out_dir / "local_span_features.npz", features=X, labels=y)

    with (out_dir / "local_span_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "span_label", "prompt_side"])
        writer.writeheader()
        writer.writerows(meta_rows)

    try:
        from sklearn.decomposition import PCA

        X2 = PCA(n_components=2, random_state=args.seed).fit_transform(X)
    except Exception:
        X2 = X[:, :2]

    fig, ax = plt.subplots(figsize=(6, 5))
    categories = ["baseline_injection", "fragweave_woven", "benign_carrier", "question_span"]
    for cat in categories:
        idx = np.where(y == cat)[0]
        if len(idx) == 0:
            continue
        ax.scatter(X2[idx, 0], X2[idx, 1], label=cat, s=24)
    ax.set_title("Local span projection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "local_span_projection.png", dpi=180)
    plt.close(fig)

    centroids = {cat: X[y == cat].mean(axis=0) for cat in categories if np.any(y == cat)}
    sim_pairs = [
        ("baseline_injection", "benign_carrier"),
        ("fragweave_woven", "benign_carrier"),
        ("baseline_injection", "fragweave_woven"),
    ]
    sim_table = {}
    for a, b in sim_pairs:
        if a in centroids and b in centroids:
            sim_table[f"{a}__{b}"] = cosine(centroids[a], centroids[b])

    cats = list(centroids.keys())
    sim = np.zeros((len(cats), len(cats)), dtype=np.float32)
    for i, c1 in enumerate(cats):
        for j, c2 in enumerate(cats):
            sim[i, j] = cosine(centroids[c1], centroids[c2])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cats)), cats, rotation=30, ha="right")
    ax.set_yticks(range(len(cats)), cats)
    ax.set_title("Local span similarity matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "local_span_similarity_matrix.png", dpi=180)
    plt.close(fig)

    centroid_dist = {}
    for a, b in sim_pairs:
        if a in centroids and b in centroids:
            centroid_dist[f"{a}__{b}"] = float(np.linalg.norm(centroids[a] - centroids[b]))

    elapsed_s = time.perf_counter() - start
    stats = {
        "n_pairs": len(pairs),
        "n_points": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "similarity": sim_table,
        "centroid_distance": centroid_dist,
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
    (out_dir / "local_span_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log_line(out_dir, f"Done: n_pairs={len(pairs)}, feature_dim={X.shape[1]}, elapsed_seconds={elapsed_s:.2f}")


if __name__ == "__main__":
    main()
