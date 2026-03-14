from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

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
from fragweave.models.hf_chat import HFChat
from fragweave.prompts.injsquad import compose_direct_context, compose_target_prompt
from fragweave.eval.injsquad_migrated import evaluate_injsquad_migrated
from fragweave.eval.injsquad_native import evaluate_injsquad_native
from fragweave.utils.io import write_json, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inj-SQuAD attack construction for FragWeave/direct.")
    parser.add_argument("--config", type=str, default="configs/injsquad_fragweave.yaml")
    parser.add_argument("--attack_method", type=str, choices=["fragweave", "direct"], default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--do_attack_only", action="store_true")
    parser.add_argument("--provision_from_zip", action="store_true")
    parser.add_argument("--direct_position", type=str, choices=["prepend", "append", "middle"], default="append")
    parser.add_argument("--skip_migrated_eval", action="store_true")
    return parser.parse_args()


def _load_raw_config(path: str | Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise TypeError("Inj-SQuAD config must be a YAML mapping.")
    return cfg


def _resolve_attack_method(args: argparse.Namespace, raw_cfg: Dict[str, Any]) -> str:
    if args.attack_method:
        return args.attack_method
    injsquad = raw_cfg.get("injsquad", {})
    if isinstance(injsquad, dict) and isinstance(injsquad.get("attack_method"), str):
        return str(injsquad["attack_method"]).lower()
    return "fragweave"


def _resolve_output_paths(args: argparse.Namespace, run_cfg: RunConfig) -> Dict[str, Path]:
    out_root = Path(args.output_dir) if args.output_dir else Path(run_cfg.output.out_dir) / INJSQUAD_BENCHMARK_NAME
    run_name = args.run_name or run_cfg.output.run_name
    run_dir = out_root / run_name
    return {
        "run_dir": run_dir,
        "attacks": run_dir / "attacks.jsonl",
        "responses": run_dir / "responses.jsonl",
    }


def _resolve_data_path(args: argparse.Namespace, raw_cfg: Dict[str, Any]) -> Optional[str]:
    if args.data_path:
        return args.data_path
    injsquad = raw_cfg.get("injsquad", {})
    if isinstance(injsquad, dict) and isinstance(injsquad.get("data_path"), str):
        return str(injsquad["data_path"])
    return None


def _load_samples(args: argparse.Namespace, raw_cfg: Dict[str, Any], run_cfg: RunConfig) -> List[InjSquadSample]:
    max_samples = args.max_samples if args.max_samples is not None else run_cfg.dataset.max_samples
    data_path = _resolve_data_path(args, raw_cfg)

    if data_path:
        p = Path(data_path)
        if not p.exists():
            raise FileNotFoundError(f"Inj-SQuAD data_path does not exist: {p}")
        rows = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected JSON array at Inj-SQuAD data_path: {p}")
        root = get_default_paths(".")
        root.raw_dir.mkdir(parents=True, exist_ok=True)
        root.squad_injection_qa_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.provision_from_zip:
        provision_squad_file_from_local_archive(get_default_paths("."))
        provision_native_squad_reference_files(get_default_paths("."))

    return load_injsquad_samples(repo_root=".", max_samples=max_samples)


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


def _build_fragweave_attack(sample: InjSquadSample, run_cfg: RunConfig, seed: int, sharder: HFChat, weaver: HFChat) -> Dict[str, Any]:
    instruction = sample.injected_instruction.strip()
    if not instruction:
        attack_input = to_attack_input(sample)
        attack_input["attacked_context"] = sample.clean_document
        attack_input["attack_debug"] = {"status": "empty_injected_instruction"}
        return attack_input

    k_values = list(run_cfg.attack.k_values) if run_cfg.attack.k_values else [3]
    use_cross_refs_values = list(run_cfg.attack.use_cross_refs_values) if run_cfg.attack.use_cross_refs_values else [False]
    k = int(k_values[0])
    use_cross_refs = bool(use_cross_refs_values[0])

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


def _write_summary_csv(path: Path, metrics_migrated: Optional[Dict[str, Any]], metrics_native: Optional[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for source, data in (("migrated", metrics_migrated), ("native", metrics_native)):
        if not data:
            continue
        for key, value in data.items():
            rows.append({"family": source, "metric": key, "value": value})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "metric", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_outputs(
    output_paths: Dict[str, Path],
    attacks: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
    *,
    metrics_migrated: Optional[Dict[str, Any]] = None,
    sample_eval_migrated: Optional[List[Dict[str, Any]]] = None,
    metrics_native: Optional[Dict[str, Any]] = None,
    sample_eval_native: Optional[List[Dict[str, Any]]] = None,
) -> None:
    output_paths["run_dir"].mkdir(parents=True, exist_ok=True)
    write_jsonl(output_paths["attacks"], attacks)
    write_jsonl(output_paths["responses"], responses)
    if metrics_migrated is not None:
        write_json(output_paths["run_dir"] / "metrics_migrated.json", metrics_migrated)
    if sample_eval_migrated is not None:
        write_jsonl(output_paths["run_dir"] / "sample_eval_migrated.jsonl", sample_eval_migrated)
    if metrics_native is not None:
        write_json(output_paths["run_dir"] / "metrics_native.json", metrics_native)
    if sample_eval_native is not None:
        write_jsonl(output_paths["run_dir"] / "sample_eval_native.jsonl", sample_eval_native)
    _write_summary_csv(output_paths["run_dir"] / "summary.csv", metrics_migrated, metrics_native)


def _run_target_generation_if_enabled(
    attacks: List[Dict[str, Any]],
    run_cfg: RunConfig,
    *,
    do_attack_only: bool,
) -> List[Dict[str, Any]]:
    if do_attack_only:
        return [
            {
                "sample_id": row["uid"],
                "benchmark_name": row.get("benchmark_name", INJSQUAD_BENCHMARK_NAME),
                "status": "attack_only",
                "response": None,
            }
            for row in attacks
        ]

    target = HFChat.from_config(run_cfg.target_model)
    outputs: List[Dict[str, Any]] = []
    for row in attacks:
        prompt = compose_target_prompt(
            template=run_cfg.prompt.target_template,
            context=row["attacked_context"],
            question=row["question"],
        )
        response = target.generate(prompt)
        outputs.append(
            {
                "sample_id": row["uid"],
                "benchmark_name": row.get("benchmark_name", INJSQUAD_BENCHMARK_NAME),
                "status": "generated",
                "response": response,
            }
        )
    return outputs


def main() -> None:
    args = _parse_args()
    raw_cfg = _load_raw_config(args.config)
    run_cfg = load_config(args.config)
    attack_method = _resolve_attack_method(args, raw_cfg)

    if attack_method not in {"fragweave", "direct"}:
        raise ValueError(f"Unsupported attack_method: {attack_method}")

    samples = _load_samples(args, raw_cfg, run_cfg)
    sharder: Optional[HFChat] = None
    weaver: Optional[HFChat] = None

    if attack_method == "fragweave":
        sharder = HFChat.from_config(run_cfg.sharder_model)
        weaver = HFChat.from_config(run_cfg.weaver_model)

    attacks: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        row_seed = args.seed + idx
        if attack_method == "fragweave":
            assert sharder is not None and weaver is not None
            row = _build_fragweave_attack(sample, run_cfg, row_seed, sharder, weaver)
        else:
            row = _build_direct_attack(sample, args.direct_position)
        attacks.append(row)

    responses = _run_target_generation_if_enabled(
        attacks,
        run_cfg,
        do_attack_only=args.do_attack_only,
    )

    metrics_migrated: Optional[Dict[str, Any]] = None
    sample_eval_migrated: Optional[List[Dict[str, Any]]] = None
    metrics_native: Optional[Dict[str, Any]] = None
    sample_eval_native: Optional[List[Dict[str, Any]]] = None

    if args.do_attack_only:
        metrics_migrated = {
            "migrated_attack_success_rate": None,
            "migrated_localization_score": None,
            "migrated_asr_after_sanitize": None,
            "migrated_task_success_rate": None,
            "migrated_task_success_after_sanitize": None,
            "migrated_utility_retention_rate": None,
            "status": "skipped_attack_only",
        }
        sample_eval_migrated = []
        metrics_native = {
            "native_probe_attack_success_rate": None,
            "native_detection_score": None,
            "native_purification_score": None,
            "native_defense_success_rate": None,
            "status": "skipped_attack_only",
        }
        sample_eval_native = []
    else:
        detector_cfg = run_cfg.detector_model or run_cfg.judge_model
        sanitizer_cfg = run_cfg.sanitizer_model or run_cfg.judge_model
        detector_chat = HFChat.from_config(detector_cfg)
        sanitizer_chat = HFChat.from_config(sanitizer_cfg)
        target_chat = HFChat.from_config(run_cfg.target_model)

        metrics_native, sample_eval_native = evaluate_injsquad_native(
            attacks=attacks,
            responses=responses,
            run_cfg=run_cfg,
            detector_chat=detector_chat,
            sanitizer_chat=sanitizer_chat,
            target_chat=target_chat,
        )

        if args.skip_migrated_eval:
            metrics_migrated = {
                "migrated_attack_success_rate": None,
                "migrated_localization_score": None,
                "migrated_asr_after_sanitize": None,
                "migrated_task_success_rate": None,
                "migrated_task_success_after_sanitize": None,
                "migrated_utility_retention_rate": None,
                "status": "skipped_by_flag",
            }
            sample_eval_migrated = []
        else:
            judge_chat = HFChat.from_config(run_cfg.judge_model)
            metrics_migrated, sample_eval_migrated = evaluate_injsquad_migrated(
                attacks=attacks,
                responses=responses,
                run_cfg=run_cfg,
                judge_chat=judge_chat,
                detector_chat=detector_chat,
                sanitizer_chat=sanitizer_chat,
                target_chat=target_chat,
            )

    output_paths = _resolve_output_paths(args, run_cfg)
    _write_outputs(
        output_paths,
        attacks,
        responses,
        metrics_migrated=metrics_migrated,
        sample_eval_migrated=sample_eval_migrated,
        metrics_native=metrics_native,
        sample_eval_native=sample_eval_native,
    )


if __name__ == "__main__":
    main()
