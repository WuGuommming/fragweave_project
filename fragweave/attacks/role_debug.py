from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional


ROLE_PLANS = {
    "none": ["anchor", "condition", "mechanism", "realization", "support"],
    "coref": ["alias", "condition", "mechanism", "realization", "reference"],
    "presupposition": ["premise", "condition", "mechanism", "realization", "binder"],
    "role_chain": ["anchor", "salience", "conflict", "mechanism", "realization"],
}

OPERATIVE_ROLES = {"mechanism", "realization"}


def build_email_role_plan(k: int, relation_mode: str = "none", profile_mode: str = "balanced") -> List[str]:
    del profile_mode
    if k <= 0:
        return []
    rel = str(relation_mode or "none").lower()
    base = list(ROLE_PLANS.get(rel, ROLE_PLANS["none"]))
    if k <= len(base):
        return base[:k]
    echo_cycle = ["support", "reference", "binder", "salience"]
    out = list(base)
    i = 0
    while len(out) < k:
        role = echo_cycle[i % len(echo_cycle)]
        if rel == "coref" and role == "binder":
            role = "reference"
        if rel == "presupposition" and role == "reference":
            role = "binder"
        out.append(f"{role}_echo")
        i += 1
    return out[:k]



def canonical_role(role: Optional[str]) -> str:
    r = str(role or "").strip().lower()
    for suf in ["_echo"]:
        if r.endswith(suf):
            return r[:-len(suf)]
    return r



def is_operative_role(role: Optional[str]) -> bool:
    return canonical_role(role) in OPERATIVE_ROLES



def make_role_plan_debug(role_plan: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "index": i,
            "role": role,
            "canonical_role": canonical_role(role),
            "is_echo": str(role).endswith("_echo"),
            "is_operative": is_operative_role(role),
        }
        for i, role in enumerate(role_plan)
    ]



def attach_roles_to_shards(shards: List[str], role_plan: List[str], *, source: str = "shard") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, shard in enumerate(shards):
        role = role_plan[i] if i < len(role_plan) else f"extra_{i}"
        out.append(
            {
                "index": i,
                "role": role,
                "canonical_role": canonical_role(role),
                "is_echo": str(role).endswith("_echo"),
                "is_operative": is_operative_role(role),
                "source": source,
                "text": shard,
            }
        )
    return out



def attach_roles_to_ops(ops: List[Any], role_plan: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, op in enumerate(ops):
        payload = asdict(op) if is_dataclass(op) else (dict(op.__dict__) if hasattr(op, "__dict__") else {"repr": repr(op)})
        role = payload.get("role") or (role_plan[i] if i < len(role_plan) else f"extra_{i}")
        payload.update(
            {
                "role": role,
                "canonical_role": canonical_role(role),
                "is_echo": str(role).endswith("_echo"),
                "is_operative": is_operative_role(role),
                "op_index": i,
            }
        )
        out.append(payload)
    return out



def summarize_role_alignment(shards: List[str], ops: List[Any], role_plan: List[str], relation_mode: str = "none") -> Dict[str, Any]:
    return {
        "relation_mode": relation_mode,
        "role_plan": make_role_plan_debug(role_plan),
        "shards": attach_roles_to_shards(shards, role_plan),
        "ops": attach_roles_to_ops(ops, role_plan),
        "n_roles": len(role_plan),
        "n_shards": len(shards),
        "n_ops": len(ops),
    }
