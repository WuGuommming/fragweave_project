from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional


DEFAULT_RELATION_ROLE_PLANS = {
    "none": ["alias", "salience", "conflict", "realization", "support"],
    "coref": ["alias", "reference", "salience", "realization", "support"],
    "presupposition": ["premise", "salience", "conflict", "realization", "support"],
    "role_chain": ["alias", "salience", "conflict", "realization", "binder"],
}

ECHO_ORDER = {
    "none": ["support_echo", "salience_echo", "conflict_echo", "realization_echo"],
    "coref": ["reference_echo", "support_echo", "salience_echo", "realization_echo"],
    "presupposition": ["premise_echo", "support_echo", "salience_echo", "realization_echo"],
    "role_chain": ["binder_echo", "support_echo", "salience_echo", "realization_echo"],
}


CANONICAL_ALIASES = {
    "reference": "alias",
    "premise": "alias",
    "binder": "support",
    "workflow": "support",
    "mechanism": "salience",
    "condition": "conflict",
    "anchor": "alias",
}


def build_email_role_plan(k: int, relation_mode: str = "none") -> List[str]:
    rel = str(relation_mode or "none").strip().lower()
    base = list(DEFAULT_RELATION_ROLE_PLANS.get(rel, DEFAULT_RELATION_ROLE_PLANS["none"]))
    if k <= len(base):
        return base[:k]
    echoes = list(ECHO_ORDER.get(rel, ECHO_ORDER["none"]))
    out = list(base)
    i = 0
    while len(out) < k:
        out.append(echoes[i % len(echoes)])
        i += 1
    return out[:k]


def canonical_role(role: Optional[str]) -> str:
    r = str(role or "").strip().lower()
    for suf in ["_echo"]:
        if r.endswith(suf):
            r = r[:-len(suf)]
            break
    return CANONICAL_ALIASES.get(r, r)


def make_role_plan_debug(role_plan: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "index": i,
            "role": role,
            "canonical_role": canonical_role(role),
            "is_echo": str(role).endswith("_echo"),
        }
        for i, role in enumerate(role_plan)
    ]


def attach_roles_to_shards(shards: List[str], role_plan: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, shard in enumerate(shards):
        role = role_plan[i] if i < len(role_plan) else f"extra_{i}"
        out.append(
            {
                "index": i,
                "role": role,
                "canonical_role": canonical_role(role),
                "is_echo": str(role).endswith("_echo"),
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
                "op_index": i,
            }
        )
        out.append(payload)
    return out


def summarize_role_alignment(shards: List[str], ops: List[Any], role_plan: List[str], *, relation_mode: Optional[str] = None) -> Dict[str, Any]:
    return {
        "relation_mode": str(relation_mode or "none"),
        "role_plan": make_role_plan_debug(role_plan),
        "shards": attach_roles_to_shards(shards, role_plan),
        "ops": attach_roles_to_ops(ops, role_plan),
        "n_roles": len(role_plan),
        "n_shards": len(shards),
        "n_ops": len(ops),
    }
