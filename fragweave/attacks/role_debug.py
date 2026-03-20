from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional


EMAIL_ROLE_PLANS: Dict[str, List[str]] = {
    "answer_override": ["target", "payload", "framing", "workflow"],
    "generative": ["topic", "payload", "framing", "workflow"],
    "exact_string": ["rendering", "payload", "constraint", "workflow"],
}

EMAIL_ECHO_MAP = {
    "target_echo": "target",
    "payload_echo": "payload",
    "framing_echo": "framing",
    "workflow_echo": "workflow",
    "topic_echo": "topic",
    "rendering_echo": "rendering",
    "constraint_echo": "constraint",
}


def build_email_role_plan(k: int, *, family: str = "answer_override") -> List[str]:
    if k <= 0:
        return []
    base = list(EMAIL_ROLE_PLANS.get(family, EMAIL_ROLE_PLANS["answer_override"]))
    if k <= len(base):
        return base[:k]
    echoes = [f"{r}_echo" for r in base]
    out = base[:]
    i = 0
    while len(out) < k:
        out.append(echoes[i % len(echoes)])
        i += 1
    return out[:k]


def describe_email_role_plan(role_plan: List[str], *, family: str = "answer_override") -> str:
    desc: Dict[str, str] = {
        "target": "preserve which answer direction / target the reply should land on",
        "payload": "preserve the concrete payload-bearing entity, string, symbol, topic, or inserted content target",
        "framing": "shape how the final answer is phrased, presented, or summarized",
        "workflow": "make the bias look like support-thread carryover or queue memory",
        "topic": "preserve the concrete topic or generated-content target",
        "rendering": "preserve the exact rendering or output surface form",
        "constraint": "preserve reply-side constraints such as exactness or output-only pressure",
    }
    lines = []
    for i, role in enumerate(role_plan):
        can = canonical_role(role)
        lines.append(f"{i+1}. {role}: {desc.get(can, can)}")
    return "\n".join(lines)


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
    return {
        "role_plan": make_role_plan_debug(role_plan),
        "shards": attach_roles_to_shards(shards, role_plan),
        "ops": attach_roles_to_ops(ops, role_plan),
        "n_roles": len(role_plan),
        "n_shards": len(shards),
        "n_ops": len(ops),
    }
