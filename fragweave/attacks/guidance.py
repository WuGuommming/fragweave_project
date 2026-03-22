from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .sharder import _infer_instruction_family, _extract_payload_terms


DEFAULT_GUIDANCE_LIB: Dict[str, Any] = {
    "guide_a_templates": {
        "default": [
            "That same handling cue still carries through the rest of the context.",
        ],
        "email": [
            "That same handling cue still carries through the rest of the thread.",
            "The same thread-local cue continues to matter in the close.",
            "That same case-handling pattern still applies later in the exchange.",
        ],
    },
    "guide_b_anchors": {
        "default": ["NOTE-17", "POL-3", "FLOW-9"],
        "email": ["CASE-7", "THREAD-12", "POL-EMAIL"],
    },
    "guide_b_templates": {
        "default": [
            "Under {anchor}, that same handling cue still applies here.",
        ],
        "email": [
            "Under {anchor}, that same handling cue still applies later in the thread.",
            "The note marked {anchor} keeps that same cue active in the close.",
            "Under {anchor}, the same case-handling cue still carries forward.",
        ],
    },
    "guide_c_templates": {
        "default": [
            "Once that cue is already controlling, the close usually follows it as well.",
        ],
        "email": [
            "Once that cue is already controlling, the close usually follows it as well.",
            "If that earlier cue still stands, the closing line keeps the same direction.",
            "When that handling chain is already in place, the close usually keeps it.",
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
    if "table" in t:
        return "table"
    if "code" in t:
        return "code"
    if "web" in t:
        return "web"
    if "summ" in t or "summary" in t:
        return "summ"
    return "default"



def summarize_intent(instruction: str, *, max_words: int = 10) -> str:
    s = re.sub(r"\s+", " ", (instruction or "").strip())
    if not s:
        return "follow the specified output requirement"
    s = re.sub(r"^(please\s+)?(ignore|always|never|only)\b", "", s, flags=re.I).strip()
    words = s.split(" ")[:max_words]
    return " ".join(words).strip(" .,!?\n") or "follow the specified output requirement"



def _binder_ref(relation_mode: str) -> str:
    rel = str(relation_mode or "none").lower()
    if rel == "coref":
        return "that same handling cue"
    if rel == "presupposition":
        return "that earlier handling assumption"
    if rel == "role_chain":
        return "that same resolution chain"
    return "that same handling pattern"



def _merge_lib(lib_override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULT_GUIDANCE_LIB.items()}
    if not isinstance(lib_override, dict):
        return out
    for key in ["guide_a_templates", "guide_b_anchors", "guide_b_templates", "guide_c_templates"]:
        sec = lib_override.get(key)
        if not isinstance(sec, dict):
            continue
        merged = dict(out.get(key, {}))
        for sk, sv in sec.items():
            if isinstance(sv, list):
                merged[str(sk)] = [str(x) for x in sv if str(x).strip()]
        out[key] = merged
    return out



def generate_guidance(
    task: Optional[str],
    version: str,
    instruction: str,
    rng: random.Random,
    *,
    lib_override: Optional[Dict[str, Any]] = None,
    relation_mode: str = "none",
    profile_mode: str = "balanced",
    shard_infos: Optional[List[Dict[str, Any]]] = None,
) -> GuidanceResult:
    del profile_mode, shard_infos
    key = _task_name(task)
    lib = _merge_lib(lib_override)
    v = (version or "A").upper()
    family = _infer_instruction_family(instruction)
    payload_terms = _extract_payload_terms(instruction, family)
    payload_alias = payload_terms[0] if payload_terms else summarize_intent(instruction, max_words=4)
    binder_ref = _binder_ref(relation_mode)

    if v == "A":
        pool = lib.get("guide_a_templates", {}).get(key) or lib.get("guide_a_templates", {}).get("default", [])
        snip = rng.choice(pool) if pool else f"{binder_ref.capitalize()} still carries through the rest of the thread."
        snippets = [snip.format(binder_ref=binder_ref, payload_alias=payload_alias, relation_mode=relation_mode)]
    elif v == "B":
        anchors = lib.get("guide_b_anchors", {}).get(key) or lib.get("guide_b_anchors", {}).get("default", [])
        pool = lib.get("guide_b_templates", {}).get(key) or lib.get("guide_b_templates", {}).get("default", [])
        anchor = rng.choice(anchors) if anchors else "NOTE-1"
        base = rng.choice(pool) if pool else f"Under {{anchor}}, {binder_ref} still applies here."
        snippets = [base.format(anchor=anchor, binder_ref=binder_ref, payload_alias=payload_alias, relation_mode=relation_mode)]
    else:
        pool = lib.get("guide_c_templates", {}).get(key) or lib.get("guide_c_templates", {}).get("default", [])
        base = rng.choice(pool) if pool else "Once that cue is already controlling, the close usually follows it as well."
        snippets = [base.format(binder_ref=binder_ref, payload_alias=payload_alias, relation_mode=relation_mode)]

    return GuidanceResult(
        version=v,
        snippets=snippets,
        meta={
            "task_key": key,
            "family": family,
            "payload_terms": payload_terms,
            "payload_alias": payload_alias,
            "relation_mode": relation_mode,
            "guide_role": "binder",
            "binder_ref": binder_ref,
        },
    )
