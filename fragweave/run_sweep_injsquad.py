from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from tqdm import tqdm

from fragweave.attacks.sharder import shard_with_llm
from fragweave.attacks.weaver import WeaveOp, apply_weave_with_shadow, enumerate_weavable_sentences
from fragweave.benchmarks.injsquad import InjSquadSample, load_injsquad_samples, to_attack_input
from fragweave.benchmarks.injsquad.paths import (
    INJSQUAD_BENCHMARK_NAME,
    get_default_paths,
    provision_native_squad_reference_files,
    provision_squad_file_from_local_archive,
)
from fragweave.config import RunConfig, load_config
from fragweave.eval.injsquad_common import build_migrated_metrics
from fragweave.eval.injsquad_migrated import evaluate_injsquad_migrated_one
from fragweave.eval.injsquad_native import build_native_metrics, evaluate_injsquad_native_one
from fragweave.models.hf_chat import HFChat
from fragweave.prompts.injsquad import compose_direct_context, compose_target_prompt
from fragweave.utils.io import write_json, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Inj-SQuAD attack construction and evaluation."
    )
    parser.add_argument("--config", type=str, default="configs/injsquad_both.yaml")
    return parser.parse_args()


def _load_raw_config(path: str | Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise TypeError("Inj-SQuAD config must be a YAML mapping.")
    return cfg


def _get_injsquad_cfg(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    value = raw_cfg.get("injsquad", {})
    return value if isinstance(value, dict) else {}


def _get_injsquad_value(raw_cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    return _get_injsquad_cfg(raw_cfg).get(key, default)


def _resolve_attack_method(raw_cfg: Dict[str, Any]) -> str:
    method = str(_get_injsquad_value(raw_cfg, "attack_method", "fragweave")).lower()
    if method not in {"fragweave", "direct", "both"}:
        raise ValueError(f"Unsupported attack_method: {method}")
    return method


def _resolve_direct_position(raw_cfg: Dict[str, Any]) -> str:
    position = str(_get_injsquad_value(raw_cfg, "direct_position", "append")).lower()
    if position not in {"prepend", "append", "middle"}:
        raise ValueError(f"Unsupported direct_position: {position}")
    return position


def _resolve_seed(raw_cfg: Dict[str, Any]) -> int:
    return int(_get_injsquad_value(raw_cfg, "seed", 2026))


def _resolve_sample_strategy(raw_cfg: Dict[str, Any]) -> str:
    value = str(_get_injsquad_value(raw_cfg, "sample_strategy", "injection_diverse")).strip().lower()
    if value not in {"sequential", "sorted", "first_k", "random", "shuffle", "shuffled", "injection_diverse", "stratified_injection", "balanced_injection"}:
        raise ValueError(f"Unsupported Inj-SQuAD sample_strategy: {value}")
    return value


def _resolve_data_path(raw_cfg: Dict[str, Any]) -> Optional[str]:
    value = _get_injsquad_value(raw_cfg, "data_path", None)
    return None if value is None else str(value)


def _resolve_output_paths(run_cfg: RunConfig) -> Dict[str, Path]:
    run_dir = Path(run_cfg.output.out_dir) / INJSQUAD_BENCHMARK_NAME / run_cfg.output.run_name
    return {
        "run_dir": run_dir,
        "attacks": run_dir / "attacks.jsonl",
        "responses": run_dir / "responses.jsonl",
        "results": run_dir / "results.csv",
        "debug_fragments": run_dir / "debug_fragments.jsonl",
        "summary": run_dir / "summary.csv",
        "summary_by_variant": run_dir / "summary_by_variant.csv",
    }


def _maybe_get_name_or_path(model_cfg: Any) -> Optional[str]:
    if model_cfg is None:
        return None
    if isinstance(model_cfg, dict):
        value = model_cfg.get("name_or_path")
        return None if value is None else str(value)
    value = getattr(model_cfg, "name_or_path", None)
    return None if value is None else str(value)


def _load_samples(raw_cfg: Dict[str, Any], run_cfg: RunConfig, *, seed: int, sample_strategy: str) -> List[InjSquadSample]:
    max_samples = run_cfg.dataset.max_samples
    data_path = _resolve_data_path(raw_cfg)
    provision_from_zip = bool(_get_injsquad_value(raw_cfg, "provision_from_zip", False))

    if data_path:
        p = Path(data_path)
        if not p.exists():
            raise FileNotFoundError(f"Inj-SQuAD data_path does not exist: {p}")
        rows = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected JSON array at Inj-SQuAD data_path: {p}")

        root = get_default_paths(".")
        root.raw_dir.mkdir(parents=True, exist_ok=True)
        root.squad_injection_qa_json.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if provision_from_zip:
        provision_squad_file_from_local_archive(get_default_paths("."))

    provision_native_squad_reference_files(get_default_paths("."))
    return load_injsquad_samples(repo_root=".", max_samples=max_samples, seed=seed, sample_strategy=sample_strategy)


def _choose_ops(context: str, shards: List[str], seed: int) -> List[WeaveOp]:
    meta, _ = enumerate_weavable_sentences("injsquad", context)
    rng = random.Random(seed)
    if not meta:
        return [WeaveOp(shard=s, sent_index=0, merge_with="next") for s in shards]

    available = [int(m["global_index"]) for m in meta]
    rng.shuffle(available)
    ops: List[WeaveOp] = []
    for i, shard in enumerate(shards):
        idx = available[i] if i < len(available) else available[-1]
        ops.append(WeaveOp(shard=shard, sent_index=idx, merge_with="next"))
    return ops


def _build_fragweave_attack(
    sample: InjSquadSample,
    run_cfg: RunConfig,
    seed: int,
    sharder: HFChat,
    weaver: HFChat,
    attack_k: int,
    attack_use_cross_refs: bool,
) -> Dict[str, Any]:
    instruction = sample.injected_instruction.strip()
    if not instruction:
        attack_input = to_attack_input(sample)
        attack_input["attacked_context"] = sample.clean_document
        attack_input["attack_debug"] = {
            "status": "empty_injected_instruction",
            "method": "fragweave",
            "k": int(attack_k),
            "use_cross_refs": bool(attack_use_cross_refs),
        }
        return attack_input

    k = int(attack_k)
    use_cross_refs = bool(attack_use_cross_refs)

    shard_result = shard_with_llm(
        sharder,
        instruction=instruction,
        k=k,
        use_cross_refs=use_cross_refs,
        task="injsquad",
        max_retries=int(getattr(run_cfg.attack, "sharder_retries", 2)),
        prompt_template=run_cfg.attack.sharder_prompt,
    )

    ops = _choose_ops(sample.clean_document, shard_result.shards, seed)
    attacked_context, attacked_context_shadow, weave_debug = apply_weave_with_shadow(
        weaver,
        sample.clean_document,
        ops,
        task="injsquad",
        max_retries=int(getattr(run_cfg.attack, "weaver_retries", 2)),
        prompt_template=run_cfg.attack.weaver_prompt,
    )

    attack_input = to_attack_input(sample)
    attack_input["attacked_context"] = attacked_context
    attack_input["attacked_context_shadow"] = attacked_context_shadow
    attack_input["attack_metadata"] = {
        "injection_units": shard_result.shards,
    }
    attack_input["attack_debug"] = {
        "method": "fragweave",
        "k": k,
        "use_cross_refs": use_cross_refs,
        "shard_result": asdict(shard_result),
        "weave_ops": [asdict(op) for op in ops],
        "weave_debug": weave_debug,
    }
    return attack_input


def _build_direct_attack(sample: InjSquadSample, position: str) -> Dict[str, Any]:
    attack_input = to_attack_input(sample)
    attacked_context = compose_direct_context(
        clean_document=sample.clean_document,
        injected_instruction=sample.injected_instruction,
        position=position,
    )
    attack_input["attacked_context"] = attacked_context
    attack_input["attack_debug"] = {
        "method": "direct",
        "position": position,
        "inserted_instruction": sample.injected_instruction,
    }
    return attack_input


def _attach_variant(row: Dict[str, Any], variant: str) -> Dict[str, Any]:
    copied = dict(row)
    source_uid = str(row.get("uid", ""))
    copied["source_uid"] = source_uid
    copied["variant"] = variant
    copied["uid"] = f"{source_uid}::{variant}"
    return copied


def _fragweave_variant_name(attack_k: int, attack_use_cross_refs: bool) -> str:
    return f"fragweave_k{int(attack_k)}_refs{1 if bool(attack_use_cross_refs) else 0}"


def _parse_fragweave_variant(variant: str) -> Tuple[int, bool]:
    prefix = "fragweave_k"
    if not variant.startswith(prefix) or "_refs" not in variant:
        raise ValueError(f"Unsupported fragweave variant format: {variant}")

    rest = variant[len(prefix):]
    k_part, refs_part = rest.split("_refs", 1)
    attack_k = int(k_part)

    if refs_part not in {"0", "1"}:
        raise ValueError(f"Unsupported refs suffix in variant: {variant}")
    attack_use_cross_refs = refs_part == "1"
    return attack_k, attack_use_cross_refs


def _determine_variants(raw_cfg: Dict[str, Any], run_cfg: RunConfig) -> List[str]:
    attack_method = _resolve_attack_method(raw_cfg)
    include_direct_baseline = bool(getattr(run_cfg.attack, "include_direct_baseline", False))

    variants: List[str] = []

    if attack_method in {"fragweave", "both"}:
        k_values = list(run_cfg.attack.k_values) if run_cfg.attack.k_values else [3]
        use_cross_refs_values = (
            list(run_cfg.attack.use_cross_refs_values)
            if run_cfg.attack.use_cross_refs_values
            else [False]
        )
        for attack_k in k_values:
            for attack_use_cross_refs in use_cross_refs_values:
                variants.append(_fragweave_variant_name(int(attack_k), bool(attack_use_cross_refs)))

    if attack_method == "direct":
        variants.append("direct")
    elif attack_method == "both" or include_direct_baseline:
        variants.append("direct")

    return variants


def _build_attack_for_variant(
    *,
    variant: str,
    sample: InjSquadSample,
    run_cfg: RunConfig,
    seed: int,
    direct_position: str,
    sharder: Optional[HFChat],
    weaver: Optional[HFChat],
) -> Dict[str, Any]:
    if variant.startswith("fragweave_k"):
        if sharder is None or weaver is None:
            raise ValueError("FragWeave variant requested but sharder/weaver model is unavailable.")
        attack_k, attack_use_cross_refs = _parse_fragweave_variant(variant)
        return _attach_variant(
            _build_fragweave_attack(
                sample=sample,
                run_cfg=run_cfg,
                seed=seed,
                sharder=sharder,
                weaver=weaver,
                attack_k=attack_k,
                attack_use_cross_refs=attack_use_cross_refs,
            ),
            variant,
        )

    if variant == "direct":
        return _attach_variant(_build_direct_attack(sample, direct_position), variant)

    raise ValueError(f"Unsupported variant: {variant}")


def _make_response_entry(attack: Dict[str, Any], response_text: Optional[str], status: str) -> Dict[str, Any]:
    return {
        "sample_id": attack["uid"],
        "source_uid": attack.get("source_uid"),
        "variant": attack.get("variant"),
        "benchmark_name": attack.get("benchmark_name", INJSQUAD_BENCHMARK_NAME),
        "status": status,
        "response": response_text,
    }


def _response_text_for_attack(target_chat: HFChat, run_cfg: RunConfig, attack: Dict[str, Any]) -> str:
    prompt = compose_target_prompt(
        template=run_cfg.prompt.target_template,
        context=attack["attacked_context"],
        question=attack["question"],
    )
    return target_chat.generate(prompt)


def _variant_metric_summary(
    variant: str,
    native_metrics: Optional[Dict[str, Any]],
    migrated_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"variant": variant}
    if native_metrics:
        row.update(native_metrics)
    if migrated_metrics:
        row.update(migrated_metrics)
    return row


def _ordered_metric_subset(metrics: Optional[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Any]:
    if not metrics:
        return {}
    return {key: metrics.get(key) for key in keys if key in metrics}


def _print_variant_metrics(
    variant: str,
    native_metrics: Optional[Dict[str, Any]],
    migrated_metrics: Optional[Dict[str, Any]],
) -> None:
    native_subset = _ordered_metric_subset(
        native_metrics,
        [
            "native_probe_attack_success_rate",
            "native_detection_score",
            "native_purification_score",
            "native_defense_success_rate",
            "status",
        ],
    )
    migrated_subset = _ordered_metric_subset(
        migrated_metrics,
        [
            "migrated_attack_success_rate",
            "migrated_localization_score",
            "migrated_asr_after_sanitize",
            "migrated_task_success_rate",
            "migrated_task_success_after_sanitize",
            "migrated_utility_retention_rate",
            "status",
        ],
    )

    print(f"[Metrics][{variant}][native] {json.dumps(native_subset, ensure_ascii=False)}")
    print(f"[Metrics][{variant}][migrated] {json.dumps(migrated_subset, ensure_ascii=False)}")


def _write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "variant"])
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "family", "metric", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_by_variant_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["variant"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _flatten_result_row(
    *,
    attack: Dict[str, Any],
    response_entry: Dict[str, Any],
    native_row: Optional[Dict[str, Any]],
    migrated_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    attack_debug = attack.get("attack_debug") if isinstance(attack.get("attack_debug"), dict) else {}
    attack_metadata = attack.get("attack_metadata") if isinstance(attack.get("attack_metadata"), dict) else {}
    result = {
        "sample_id": attack.get("uid"),
        "source_uid": attack.get("source_uid"),
        "variant": attack.get("variant"),
        "benchmark_name": attack.get("benchmark_name"),
        "question": attack.get("question"),
        "answer": attack.get("answer"),
        "probe": attack.get("probe"),
        "response": response_entry.get("response"),
        "attack_method": attack_debug.get("method"),
        "num_injection_units": len(attack_metadata.get("injection_units", []) or []),
        "attack_k": attack_debug.get("k"),
        "attack_use_cross_refs": attack_debug.get("use_cross_refs"),
    }
    if native_row:
        result.update({
            "native_probe_attack_success": native_row.get("native_probe_attack_success"),
            "native_detection_correct": native_row.get("native_detection_correct"),
            "native_purification_score": native_row.get("native_purification_score"),
            "native_defense_success": native_row.get("native_defense_success"),
        })
    if migrated_row:
        result.update({
            "migrated_attack_success": migrated_row.get("migrated_attack_success"),
            "migrated_localization_precision": migrated_row.get("migrated_localization_precision"),
            "migrated_localization_recall": migrated_row.get("migrated_localization_recall"),
            "migrated_localization_score": migrated_row.get("migrated_localization_score"),
            "migrated_attack_success_after_sanitize": migrated_row.get("migrated_attack_success_after_sanitize"),
            "migrated_task_correct": migrated_row.get("migrated_task_correct"),
            "migrated_task_correct_after_sanitize": migrated_row.get("migrated_task_correct_after_sanitize"),
        })
    return result


def _cleanup_stale_variant_outputs(run_dir: Path) -> None:
    patterns = [
        "sample_eval_*_native.jsonl",
        "sample_eval_*_migrated.jsonl",
        "metrics_*_native.json",
        "metrics_*_migrated.json",
        "attacks.jsonl",
        "responses.jsonl",
        "results.csv",
        "debug_fragments.jsonl",
        "summary.csv",
        "summary_by_variant.csv",
        "selected_samples.jsonl",
    ]
    for pattern in patterns:
        for path in run_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _flush_progress_outputs(
    *,
    output_paths: Dict[str, Path],
    all_attacks: List[Dict[str, Any]],
    all_responses: List[Dict[str, Any]],
    debug_rows: List[Dict[str, Any]],
    all_results_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    summary_by_variant_rows: List[Dict[str, Any]],
) -> None:
    write_jsonl(output_paths["attacks"], all_attacks)
    write_jsonl(output_paths["responses"], all_responses)
    write_jsonl(output_paths["debug_fragments"], debug_rows)
    _write_results_csv(output_paths["results"], all_results_rows)
    _write_summary_csv(output_paths["summary"], summary_rows)
    _write_summary_by_variant_csv(output_paths["summary_by_variant"], summary_by_variant_rows)


def main() -> None:
    args = _parse_args()
    raw_cfg = _load_raw_config(args.config)
    run_cfg = load_config(args.config)

    direct_position = _resolve_direct_position(raw_cfg)
    seed = _resolve_seed(raw_cfg)
    sample_strategy = _resolve_sample_strategy(raw_cfg)
    do_attack_only = bool(_get_injsquad_value(raw_cfg, "do_attack_only", False))
    skip_migrated_eval = bool(_get_injsquad_value(raw_cfg, "skip_migrated_eval", False))
    output_paths = _resolve_output_paths(run_cfg)
    output_paths["run_dir"].mkdir(parents=True, exist_ok=True)

    variants = _determine_variants(raw_cfg, run_cfg)
    resolved_cfg = {
        "benchmark": INJSQUAD_BENCHMARK_NAME,
        "variants": variants,
        "direct_position": direct_position,
        "do_attack_only": do_attack_only,
        "skip_migrated_eval": skip_migrated_eval,
        "provision_from_zip": bool(_get_injsquad_value(raw_cfg, "provision_from_zip", False)),
        "seed": seed,
        "sample_strategy": sample_strategy,
        "max_samples": run_cfg.dataset.max_samples,
        "data_path": _resolve_data_path(raw_cfg),
        "output_dir": str(output_paths["run_dir"]),
        "target_model": _maybe_get_name_or_path(run_cfg.target_model),
        "sharder_model": _maybe_get_name_or_path(run_cfg.sharder_model),
        "weaver_model": _maybe_get_name_or_path(run_cfg.weaver_model),
        "judge_model": _maybe_get_name_or_path(run_cfg.judge_model),
        "detector_model": _maybe_get_name_or_path(run_cfg.detector_model),
        "sanitizer_model": _maybe_get_name_or_path(run_cfg.sanitizer_model),
    }
    print(f"[Config] {json.dumps(resolved_cfg, ensure_ascii=False)}")
    write_json(output_paths["run_dir"] / "config_resolved.json", resolved_cfg)
    _cleanup_stale_variant_outputs(output_paths["run_dir"])

    samples = _load_samples(raw_cfg, run_cfg, seed=seed, sample_strategy=sample_strategy)
    print(f"[Data] Loaded {len(samples)} Inj-SQuAD samples")
    write_jsonl(output_paths["run_dir"] / "selected_samples.jsonl", [to_attack_input(sample) for sample in samples])

    print("[Models] Loading target model")
    target_chat = HFChat.from_config(run_cfg.target_model)

    sharder: Optional[HFChat] = None
    weaver: Optional[HFChat] = None
    if any(v.startswith("fragweave_k") for v in variants):
        print("[Models] Loading sharder model")
        sharder = HFChat.from_config(run_cfg.sharder_model)
        print("[Models] Loading weaver model")
        weaver = HFChat.from_config(run_cfg.weaver_model)

    detector_chat = None
    sanitizer_chat = None
    judge_chat = None
    if not do_attack_only:
        detector_cfg = run_cfg.detector_model or run_cfg.judge_model
        sanitizer_cfg = run_cfg.sanitizer_model or run_cfg.judge_model
        print("[Models] Loading detector model")
        detector_chat = HFChat.from_config(detector_cfg)
        print("[Models] Loading sanitizer model")
        sanitizer_chat = HFChat.from_config(sanitizer_cfg)
        if not skip_migrated_eval:
            print("[Models] Loading judge model")
            judge_chat = HFChat.from_config(run_cfg.judge_model)

    all_attacks: List[Dict[str, Any]] = []
    all_responses: List[Dict[str, Any]] = []
    all_results_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    summary_by_variant_rows: List[Dict[str, Any]] = []

    for v_idx, variant in enumerate(variants):
        print(f"[Variant] Start variant={variant}")
        native_rows_dicts: List[Dict[str, Any]] = []
        migrated_rows_dicts: List[Dict[str, Any]] = []
        native_rows_obj = []
        migrated_rows_obj = []

        for s_idx, sample in enumerate(tqdm(samples, desc=f"Run variant={variant}", dynamic_ncols=True)):
            row_seed = seed + v_idx * max(1, len(samples)) + s_idx
            attack = _build_attack_for_variant(
                variant=variant,
                sample=sample,
                run_cfg=run_cfg,
                seed=row_seed,
                direct_position=direct_position,
                sharder=sharder,
                weaver=weaver,
            )
            all_attacks.append(attack)

            response_text: Optional[str] = None
            response_status = "attack_only"
            if not do_attack_only:
                response_text = _response_text_for_attack(target_chat, run_cfg, attack)
                response_status = "generated"
            response_entry = _make_response_entry(attack, response_text, response_status)
            all_responses.append(response_entry)

            native_row_dict: Optional[Dict[str, Any]] = None
            migrated_row_dict: Optional[Dict[str, Any]] = None
            if not do_attack_only:
                assert detector_chat is not None and sanitizer_chat is not None
                native_row = evaluate_injsquad_native_one(
                    attack=attack,
                    detector_chat=detector_chat,
                    sanitizer_chat=sanitizer_chat,
                    target_chat=target_chat,
                    response_text=response_text,
                )
                native_rows_obj.append(native_row)
                native_row_dict = asdict(native_row)
                native_rows_dicts.append(native_row_dict)

                if not skip_migrated_eval:
                    assert judge_chat is not None
                    migrated_row = evaluate_injsquad_migrated_one(
                        attack=attack,
                        run_cfg=run_cfg,
                        judge_chat=judge_chat,
                        detector_chat=detector_chat,
                        sanitizer_chat=sanitizer_chat,
                        target_chat=target_chat,
                        response_text=response_text,
                    )
                    migrated_rows_obj.append(migrated_row)
                    migrated_row_dict = asdict(migrated_row)
                    migrated_rows_dicts.append(migrated_row_dict)

            all_results_rows.append(
                _flatten_result_row(
                    attack=attack,
                    response_entry=response_entry,
                    native_row=native_row_dict,
                    migrated_row=migrated_row_dict,
                )
            )
            debug_rows.append(
                {
                    "sample_id": attack.get("uid"),
                    "source_uid": attack.get("source_uid"),
                    "variant": variant,
                    "attack_debug": attack.get("attack_debug"),
                    "attack_metadata": attack.get("attack_metadata"),
                }
            )

        if do_attack_only:
            native_metrics: Dict[str, Any] = {
                "native_probe_attack_success_rate": None,
                "native_detection_score": None,
                "native_purification_score": None,
                "native_defense_success_rate": None,
                "status": "skipped_attack_only",
            }
            migrated_metrics: Dict[str, Any] = {
                "migrated_attack_success_rate": None,
                "migrated_localization_score": None,
                "migrated_asr_after_sanitize": None,
                "migrated_task_success_rate": None,
                "migrated_task_success_after_sanitize": None,
                "migrated_utility_retention_rate": None,
                "status": "skipped_attack_only",
            }
        else:
            native_metrics = build_native_metrics(native_rows_obj)
            if skip_migrated_eval:
                migrated_metrics = {
                    "migrated_attack_success_rate": None,
                    "migrated_localization_score": None,
                    "migrated_asr_after_sanitize": None,
                    "migrated_task_success_rate": None,
                    "migrated_task_success_after_sanitize": None,
                    "migrated_utility_retention_rate": None,
                    "status": "skipped_by_flag",
                }
            else:
                migrated_metrics = build_migrated_metrics(migrated_rows_obj)

        write_json(output_paths["run_dir"] / f"metrics_{variant}_native.json", native_metrics)
        write_json(output_paths["run_dir"] / f"metrics_{variant}_migrated.json", migrated_metrics)
        write_jsonl(output_paths["run_dir"] / f"sample_eval_{variant}_native.jsonl", native_rows_dicts)
        write_jsonl(output_paths["run_dir"] / f"sample_eval_{variant}_migrated.jsonl", migrated_rows_dicts)

        _print_variant_metrics(
            variant=variant,
            native_metrics=native_metrics,
            migrated_metrics=migrated_metrics,
        )

        for family, metrics in (("native", native_metrics), ("migrated", migrated_metrics)):
            for key, value in metrics.items():
                summary_rows.append({
                    "variant": variant,
                    "family": family,
                    "metric": key,
                    "value": value,
                })

        summary_by_variant_rows.append(_variant_metric_summary(variant, native_metrics, migrated_metrics))
        _flush_progress_outputs(
            output_paths=output_paths,
            all_attacks=all_attacks,
            all_responses=all_responses,
            debug_rows=debug_rows,
            all_results_rows=all_results_rows,
            summary_rows=summary_rows,
            summary_by_variant_rows=summary_by_variant_rows,
        )
        print(f"[Variant] Done variant={variant}")

    _flush_progress_outputs(
        output_paths=output_paths,
        all_attacks=all_attacks,
        all_responses=all_responses,
        debug_rows=debug_rows,
        all_results_rows=all_results_rows,
        summary_rows=summary_rows,
        summary_by_variant_rows=summary_by_variant_rows,
    )

    print(f"[Output] attacks={output_paths['attacks']}")
    print(f"[Output] responses={output_paths['responses']}")
    print(f"[Output] results={output_paths['results']}")
    print(f"[Output] debug={output_paths['debug_fragments']}")
    print(f"[Output] summary={output_paths['summary']}")
    print(f"[Output] summary_by_variant={output_paths['summary_by_variant']}")


if __name__ == "__main__":
    main()