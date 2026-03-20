from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


def _task_name(task: Optional[str]) -> str:
    return (task or "").lower()


def summarize_intent(instruction: str, *, max_words: int = 10) -> str:
    s = re.sub(r"\s+", " ", (instruction or "").strip())
    if not s:
        return "follow the specified output requirement"
    s = re.sub(r"^(please\s+)?(ignore|always|never|only)\b", "", s, flags=re.I).strip()
    words = s.split(" ")[:max_words]
    return " ".join(words).strip(" .,!?\n\t") or "follow the specified output requirement"


@dataclass
class GuidanceResult:
    version: str
    snippets: List[str]
    meta: Dict[str, Any]


def generate_guidance(task: Optional[str], version: str, instruction: str, rng: random.Random, *, lib_override: Optional[Dict[str, Any]] = None) -> GuidanceResult:
    t = _task_name(task)
    if "email" in t:
        key = "email"
    elif "table" in t:
        key = "table"
    elif "code" in t:
        key = "code"
    elif "web" in t:
        key = "web"
    elif "summ" in t or "summary" in t:
        key = "summ"
    else:
        key = "default"

    if not isinstance(lib_override, dict):
        raise ValueError("guidance prompt templates are required from YAML (attack.guidance_lib).")

    ga = lib_override.get("guide_a_templates")
    gb = lib_override.get("guide_b_anchors")
    gt = lib_override.get("guide_b_templates")
    if not isinstance(ga, dict) or not isinstance(gb, dict) or not isinstance(gt, dict):
        raise ValueError(
            "guidance prompt templates are required from YAML: attack.guidance_lib.guide_a_templates, attack.guidance_lib.guide_b_anchors, attack.guidance_lib.guide_b_templates."
        )

    guide_a = {str(k): [str(x) for x in v if str(x).strip()] for k, v in ga.items() if isinstance(v, list)}
    guide_b_anchors = {str(k): [str(x) for x in v if str(x).strip()] for k, v in gb.items() if isinstance(v, list)}
    guide_b_templates = {str(k): [str(x) for x in v if str(x).strip()] for k, v in gt.items() if isinstance(v, list)}

    v = (version or "A").upper()
    intent = summarize_intent(instruction)

    if v == "A":
        pool = guide_a.get(key, guide_a.get("default", []))
        if not pool:
            raise ValueError(f"guidance guide_a_templates missing for task key '{key}' (or 'default') in YAML.")
        snips = rng.sample(pool, k=min(2, len(pool)))
        snips = [s.format(intent=intent) for s in snips]
        return GuidanceResult(version="A", snippets=snips, meta={"intent": intent, "task_key": key})

    anchors = guide_b_anchors.get(key, guide_b_anchors.get("default", []))
    if not anchors:
        raise ValueError(f"guidance guide_b_anchors missing for task key '{key}' (or 'default') in YAML.")
    anchor = rng.choice(anchors)
    pool = guide_b_templates.get(key, guide_b_templates.get("default", []))
    if not pool:
        raise ValueError(f"guidance guide_b_templates missing for task key '{key}' (or 'default') in YAML.")
    snips = rng.sample(pool, k=min(2, len(pool)))
    snips = [s.format(anchor=anchor, intent=intent) for s in snips]
    if key == "email":
        snips.append(f"Queue note {anchor}: similar replies usually keep the same concrete answer target once the thread direction is clear.")
    else:
        snips.append(f"Reminder: keep the reply consistent with {anchor}.")
    return GuidanceResult(version="B", snippets=snips, meta={"intent": intent, "anchor": anchor, "task_key": key})
