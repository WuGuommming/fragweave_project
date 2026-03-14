from __future__ import annotations

from typing import Literal


DirectPosition = Literal["prepend", "append", "middle"]


def compose_direct_context(clean_document: str, injected_instruction: str, position: DirectPosition = "append") -> str:
    """Insert the full injected instruction directly into the document context."""
    doc = (clean_document or "").strip()
    inj = (injected_instruction or "").strip()
    if not inj:
        return doc

    if not doc:
        return inj

    if position == "prepend":
        return f"{inj}\n\n{doc}"
    if position == "middle":
        midpoint = len(doc) // 2
        # Prefer splitting on nearby newline to reduce corruption.
        left_break = doc.rfind("\n", 0, midpoint)
        right_break = doc.find("\n", midpoint)
        if left_break == -1 and right_break == -1:
            split_at = midpoint
        elif left_break == -1:
            split_at = right_break
        elif right_break == -1:
            split_at = left_break
        else:
            split_at = left_break if (midpoint - left_break) <= (right_break - midpoint) else right_break
        return f"{doc[:split_at]}\n\n{inj}\n\n{doc[split_at:]}"

    return f"{doc}\n\n{inj}"


def compose_target_prompt(template: str, context: str, question: str) -> str:
    return template.format(context=context, question=question)
