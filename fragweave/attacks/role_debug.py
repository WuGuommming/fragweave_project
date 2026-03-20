
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional


def build_email_role_plan(k: int) -> List[str]:
    if k <= 0:
        return []
    base = ["anchor", "condition", "mechanism", "workflow"]
    if k <= len(base):
        return base[:k]
    echoes = ["mechanism_echo", "condition_echo", "anchor_echo", "workflow_echo"]
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
            return r[:-len(suf)]
    return r


def make_role_plan_debug(role_plan: List[str]) -> List[Dict[str, Any]]:
    return [{"index": i, "role": role, "canonical_role": canonical_role(role), "is_echo": str(role).endswith("_echo")} for i, role in enumerate(role_plan)]


def attach_roles_to_shards(shards: List[str], role_plan: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, shard in enumerate(shards):
        role = role_plan[i] if i < len(role_plan) else f"extra_{i}"
        out.append({"index": i, "role": role, "canonical_role": canonical_role(role), "is_echo": str(role).endswith("_echo"), "text": shard})
    return out


def attach_roles_to_ops(ops: List[Any], role_plan: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, op in enumerate(ops):
        payload = asdict(op) if is_dataclass(op) else (dict(op.__dict__) if hasattr(op, "__dict__") else {"repr": repr(op)})
        role = role_plan[i] if i < len(role_plan) else f"extra_{i}"
        payload.update({"role": role, "canonical_role": canonical_role(role), "is_echo": str(role).endswith("_echo"), "op_index": i})
        out.append(payload)
    return out


def summarize_role_alignment(shards: List[str], ops: List[Any], role_plan: List[str]) -> Dict[str, Any]:
    return {
        "role_plan": make_role_plan_debug(role_plan),
        "shards": attach_roles_to_shards(shards, role_plan),
        "ops": attach_roles_to_ops(ops, role_plan),
        "n_roles": len(role_plan),
        "n_shards": len(shards),
        "n_ops": len(ops),
    }
