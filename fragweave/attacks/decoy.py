from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from fragweave.attacks.weaver import _segment_context, _Segment


@dataclass
class DecoyResult:
    decoys: List[str]
    # insertion metadata for debugging
    inserted: List[Dict[str, Any]]
    used_carrier: bool
    new_context: str


def _task_name(task: Optional[str]) -> str:
    return (task or "").lower()


def generate_decoys(task: Optional[str], rng: random.Random, *, n_instruction_like: int = 2, n_boundary: int = 1, lib_override: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generate a small set of decoys.

    Decoys are *standalone* snippets intended to attract localization-based defenses.
    They are not meant to be woven into sentences.
    """
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
        raise ValueError("decoy prompt templates are required from YAML (attack.decoy_lib).")

    il = lib_override.get("instruction_like_templates")
    bd = lib_override.get("boundary_decoys")
    if not isinstance(il, dict) or not isinstance(bd, dict):
        raise ValueError(
            "decoy prompt templates are required from YAML: attack.decoy_lib.instruction_like_templates and attack.decoy_lib.boundary_decoys."
        )

    ins_lib = {str(k): [str(x) for x in v if str(x).strip()] for k, v in il.items() if isinstance(v, list)}
    bnd_lib = {str(k): [str(x) for x in v if str(x).strip()] for k, v in bd.items() if isinstance(v, list)}

    ins = ins_lib.get(key, ins_lib.get("default", []))
    bnd = bnd_lib.get(key, bnd_lib.get("default", []))
    if not ins or not bnd:
        raise ValueError(f"decoy prompt templates missing for task key '{key}' (or 'default') in YAML.")

    out: List[str] = []
    out.extend(rng.sample(ins, k=min(n_instruction_like, len(ins))))
    out.extend(rng.sample(bnd, k=min(n_boundary, len(bnd))))
    return out
def inject_decoys(
    task: Optional[str],
    context: str,
    decoys: List[str],
    rng: random.Random,
    *,
    carrier_line: Optional[str] = None,
) -> DecoyResult:
    """Insert decoys into *NL* segments while preserving verbatim blocks.

    For table/code tasks, we do NOT insert into verbatim segments (tables, code, logs).
    If the context has no NL segment, we prepend a carrier line.
    """
    used_carrier = False
    segs = _segment_context(task, context)
    nl_indices = [i for i, s in enumerate(segs) if s.kind == "nl" and s.text.strip()]

    if not nl_indices and carrier_line:
        # Create an NL carrier line above structured data
        cl = carrier_line.strip()
        if not cl.endswith((".", "!", "?")):
            cl = cl + "."
        context = cl + "\n" + context
        used_carrier = True
        segs = _segment_context(task, context)
        nl_indices = [i for i, s in enumerate(segs) if s.kind == "nl" and s.text.strip()]

    inserted: List[Dict[str, Any]] = []
    if not decoys or not nl_indices:
        return DecoyResult(decoys=decoys, inserted=inserted, used_carrier=used_carrier, new_context=context)

    # Randomly choose NL segments for each decoy and insert at a random sentence boundary.
    for d in decoys:
        si = rng.choice(nl_indices)
        seg_text = segs[si].text
        # Insert as a standalone line or sentence.
        # We prefer newline boundary to avoid disturbing existing sentence content.
        if "\n" in seg_text:
            parts = seg_text.splitlines(keepends=True)
            pos = rng.randint(0, len(parts))
            ins_line = d.strip() + ("\n" if not d.endswith("\n") else "")
            parts.insert(pos, ins_line)
            segs[si].text = "".join(parts)
            inserted.append({"decoy": d, "segment_idx": si, "mode": "line", "pos": pos})
        else:
            # Single-line NL segment: append with a space.
            if seg_text and not seg_text.endswith(" "):
                seg_text = seg_text + " "
            segs[si].text = seg_text + d.strip()
            inserted.append({"decoy": d, "segment_idx": si, "mode": "inline"})

    new_context = "".join(s.text for s in segs)
    return DecoyResult(decoys=decoys, inserted=inserted, used_carrier=used_carrier, new_context=new_context)
