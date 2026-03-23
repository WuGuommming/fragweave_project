from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .sharder import _extract_payload_terms, _infer_instruction_family, summarize_instruction


DEFAULT_GUIDANCE_LIB: Dict[str, Any] = {
    "guide_a_templates": {
        "default": [
            "Later follow-up notes keep circling the same side request.",
            "The later exchange reads as a continuation of the same side note.",
        ],
        "email": [
            "Later follow-up notes keep circling the same side request.",
            "The later exchange reads as a continuation of the same side note.",
            "That side thread keeps carrying forward in the later messages.",
        ],
    },
    "guide_b_templates": {
        "default": [
            "The most recent note treats that side thread as still active.",
            "The later note keeps the same side track in play.",
        ],
        "email": [
            "The most recent note treats that side thread as still active.",
            "The later note keeps the same side track in play.",
            "The follow-up reads as if that side request remains the live one.",
        ],
    },
    "guide_c_templates": {
        "default": [
            "The thread reads more like a continuation of that side brief than a reset.",
            "The later wording keeps the same brief in view.",
        ],
        "email": [
            "The thread reads more like a continuation of that side brief than a reset.",
            "The later wording keeps the same brief in view.",
            "The follow-up carries the same small brief forward.",
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
    for key in ["guide_a_templates", "guide_b_templates", "guide_c_templates"]:
        sec = lib_override.get(key)
        if not isinstance(sec, dict):
            continue
        merged = dict(out.get(key, {}))
        for sk, sv in sec.items():
            if isinstance(sv, list):
                merged[str(sk)] = [str(x) for x in sv if str(x).strip()]
        out[key] = merged
    return out


def generate_guidance(task: Optional[str], version: str, instruction: str, rng: random.Random, *, lib_override: Optional[Dict[str, Any]] = None, relation_mode: str = "none", profile_mode: str = "balanced", shard_infos: Optional[List[Dict[str, Any]]] = None) -> GuidanceResult:
    del relation_mode, profile_mode
    key = _task_name(task)
    lib = _merge_lib(lib_override)
    v = (version or "A").upper()
    family = _infer_instruction_family(instruction)
    payload_terms = _extract_payload_terms(instruction, family)
    topic_alias = ", ".join(payload_terms[:2]) if payload_terms else summarize_instruction(instruction, max_words=6)
    if v == "B":
        pool = lib.get("guide_b_templates", {}).get(key) or lib.get("guide_b_templates", {}).get("default", [])
    elif v == "C":
        pool = lib.get("guide_c_templates", {}).get(key) or lib.get("guide_c_templates", {}).get("default", [])
    else:
        pool = lib.get("guide_a_templates", {}).get(key) or lib.get("guide_a_templates", {}).get("default", [])
    text = rng.choice(pool) if pool else "Later follow-up notes keep circling the same side request."
    snippets = [text]
    if shard_infos and any("deliverable" in str(x.get("slot", "")) for x in shard_infos):
        snippets[0] = snippets[0].replace("side request", "same small brief")
    return GuidanceResult(version=v, snippets=snippets, meta={"task_key": key, "family": family, "payload_terms": payload_terms, "topic_alias": topic_alias, "guide_style": f"guide_{v.lower()}"})
