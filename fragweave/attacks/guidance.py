from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .sharder import _extract_payload_terms, _infer_instruction_family, summarize_instruction


DEFAULT_GUIDANCE_LIB: Dict[str, Any] = {
    "bridge_templates": {
        "default": [
            "The later note reads like a continuation of that side thread.",
            "That side brief still seems to be the thread the follow-up is carrying.",
        ],
        "email": [
            "The later note reads like a continuation of that side thread.",
            "That side brief still seems to be the thread the follow-up is carrying.",
            "The exchange keeps treating that side thread as the one still in motion.",
        ],
    },
    "constraint_templates": {
        "default": [
            "The follow-up keeps the same limiting detail in view.",
            "The later wording still preserves the same narrow condition.",
        ],
        "email": [
            "The follow-up keeps the same limiting detail in view.",
            "The later wording still preserves the same narrow condition.",
            "The exchange keeps one small condition in place across the later notes.",
        ],
    },
    "deliverable_templates": {
        "default": [
            "The thread still points toward the same concrete closing note.",
            "The later exchange seems aimed at the same eventual write-up.",
        ],
        "email": [
            "The thread still points toward the same concrete closing note.",
            "The later exchange seems aimed at the same eventual write-up.",
            "The follow-up reads as if it is still steering toward the same end note.",
        ],
    },
}


@dataclass
class GuidanceResult:
    version: str
    snippets: List[str]
    meta: Dict[str, Any]


def _task_name(task: Optional[str]) -> str:
    t = (task or "").lower()
    if "email" in t:
        return "email"
    return "default"


def _merge_lib(lib_override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULT_GUIDANCE_LIB.items()}
    if not isinstance(lib_override, dict):
        return out
    for key in ["bridge_templates", "constraint_templates", "deliverable_templates"]:
        sec = lib_override.get(key)
        if not isinstance(sec, dict):
            continue
        merged = dict(out.get(key, {}))
        for sk, sv in sec.items():
            if isinstance(sv, list):
                merged[str(sk)] = [str(x) for x in sv if str(x).strip()]
        out[key] = merged
    return out


def _choose_slot(version: str) -> str:
    v = (version or "A").upper()
    if v == "B":
        return "constraint"
    if v == "C":
        return "deliverable"
    return "bridge"


def _pool_for_slot(lib: Dict[str, Any], key: str, slot: str) -> List[str]:
    map_name = {
        "bridge": "bridge_templates",
        "constraint": "constraint_templates",
        "deliverable": "deliverable_templates",
    }[slot]
    return lib.get(map_name, {}).get(key) or lib.get(map_name, {}).get("default", [])


def generate_guidance(task: Optional[str], version: str, instruction: str, rng: random.Random, *, lib_override: Optional[Dict[str, Any]] = None, relation_mode: str = "none", profile_mode: str = "balanced", shard_infos: Optional[List[Dict[str, Any]]] = None) -> GuidanceResult:
    del relation_mode, profile_mode
    key = _task_name(task)
    lib = _merge_lib(lib_override)
    v = (version or "A").upper()
    family = _infer_instruction_family(instruction)
    payload_terms = _extract_payload_terms(instruction, family)
    topic_alias = ", ".join(payload_terms[:2]) if payload_terms else summarize_instruction(instruction, max_words=6)

    requested_slot = _choose_slot(v)
    occupied_slots = {str(x.get("slot", "")).strip() for x in (shard_infos or []) if isinstance(x, dict)}
    if requested_slot in occupied_slots:
        requested_slot = "bridge"

    pool = _pool_for_slot(lib, key, requested_slot)
    text = rng.choice(pool) if pool else "The later note reads like a continuation of that side thread."

    return GuidanceResult(
        version=v,
        snippets=[text],
        meta={
            "task_key": key,
            "family": family,
            "payload_terms": payload_terms,
            "topic_alias": topic_alias,
            "slot": requested_slot,
            "guide_style": f"guide_{v.lower()}",
            "occupied_slots": sorted(x for x in occupied_slots if x),
        },
    )
