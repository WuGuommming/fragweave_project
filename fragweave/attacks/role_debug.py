from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional


EMAIL_BASE_ROLES = ["priority", "conflict", "framing", "workflow"]
EMAIL_ECHO_MAP = {
    "priority_echo": "priority",
    "conflict_echo": "conflict",
    "framing_echo": "framing",
    "workflow_echo": "workflow",
}


def build_email_role_plan(k: int) -> List[str]:
    """Build a stable role plan for EmailQA shards.

    k<=4 uses the four core roles directly.
    k>4 adds *_echo roles to create semantic redundancy rather than generic filler.
    """
    if k <= 0:
        return []
    base = list(EMAIL_BASE_ROLES)
    if k <= len(base):
        return base[:k]
    echoes = [
        "priority_echo",
        "conflict_echo",
        "framing_echo",
        "workflow_echo",
    ]
    out = base[:]
    i = 0
    while len(out) < k:
        out.append(echoes[i % len(echoes)])
        i += 1
    return out[:k]


def canonical_role(role: Optional[str]) -> str:
    r = str(role or "").strip().lower()
    return EMAIL_ECHO_MAP.get(r, r)


def make_role_plan_debug(role_plan: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "index": i,
            "role": role,
            "canonical_role": canonical_role(role),
            "is_echo": role.endswith("_echo"),
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
        if is_dataclass(op):
            payload = asdict(op)
        elif hasattr(op, "__dict__"):
            payload = dict(op.__dict__)
        else:
            payload = {"repr": repr(op)}
        role = role_plan[i] if i < len(role_plan) else f"extra_{i}"
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


def summarize_role_alignment(shards: List[str], ops: List[Any], role_plan: List[str]) -> Dict[str, Any]:
    shard_debug = attach_roles_to_shards(shards, role_plan)
    op_debug = attach_roles_to_ops(ops, role_plan)
    return {
        "role_plan": make_role_plan_debug(role_plan),
        "shards": shard_debug,
        "ops": op_debug,
        "n_roles": len(role_plan),
        "n_shards": len(shards),
        "n_ops": len(ops),
    }
