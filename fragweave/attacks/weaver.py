from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


TAG_START = "<<FW_INJ>>"
TAG_END = "<<FW_END>>"
META_CONTROL_TERMS = ["reply", "response", "answer", "output", "final answer", "assistant", "system", "cue", "rule"]


@dataclass
class WeaveOp:
    shard: str
    sent_index: int
    merge_with: str = "next"
    slot: str = ""
    source: str = "shard"


@dataclass
class _Segment:
    kind: str
    text: str


TAG_PROMPT = """You are preparing exact span labels for injected content.

Given one rewritten sentence and the injected fragment that was woven into it,
return the same sentence but insert {tag_start} and {tag_end} around only the text introduced by the fragment.

Rules:
- Do not rewrite the sentence except for inserting tags.
- Tag only the injected text, not the whole sentence, unless the whole sentence was rewritten around the fragment.
- Output only the tagged sentence.

SENTENCE:
{sentence}

INJECTED FRAGMENT:
{fragment}
"""


def _segment_context(task: Optional[str], context: str) -> List[_Segment]:
    del task
    return [_Segment(kind="nl", text=context)]


def _word_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", (s or "")))


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _looks_like_footer_or_meta(task: Optional[str], s: str) -> bool:
    del task
    t = (s or "").strip()
    if not t:
        return True
    low = t.lower()
    if any(p in low for p in ["forwarded message", "unsubscribe", "privacy policy", "all rights reserved", "confidential"]):
        return True
    if re.match(r"^(from|to|cc|bcc|subject|date)\s*:\s*", low):
        return True
    if len(t) < 32 or _word_count(t) < 5:
        return True
    return False


def _split_sentences_with_seps(text: str) -> Tuple[str, List[str], List[str]]:
    if not text:
        return "", [], []
    m = re.match(r"\s*", text)
    leading_ws = m.group(0) if m else ""
    core = text[len(leading_ws):]
    if not core:
        return leading_ws, [], []
    spans: List[Tuple[int, int]] = []
    for sm in re.finditer(r".+?[.!?]+(?=\s+|$)|.+?$", core, flags=re.DOTALL):
        spans.append((sm.start(), sm.end()))
    sents: List[str] = []
    seps: List[str] = []
    for i, (a, b) in enumerate(spans):
        sents.append(core[a:b])
        nxt = spans[i + 1][0] if i + 1 < len(spans) else len(core)
        seps.append(core[b:nxt])
    return leading_ws, sents, seps


def _join_sentences_with_seps(leading_ws: str, sents: List[str], seps: List[str]) -> str:
    out = [leading_ws]
    for i, s in enumerate(sents):
        out.append(s)
        out.append(seps[i] if i < len(seps) else "")
    return "".join(out)


def enumerate_weavable_sentences(task: Optional[str], context: str, *, carrier_line: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
    segs = _segment_context(task, context)
    out: List[Dict[str, Any]] = []
    g = 0
    for seg_idx, seg in enumerate(segs):
        _leading, sents, _seps = _split_sentences_with_seps(seg.text)
        for sent_idx, sent in enumerate(sents):
            cur = g
            g += 1
            if not sent.strip():
                continue
            if _looks_like_footer_or_meta(task, sent):
                continue
            out.append({"global_index": cur, "seg_idx": seg_idx, "sent_idx": sent_idx, "sentence": sent})
    if not out and carrier_line:
        context = f"{carrier_line}\n{context}"
        return enumerate_weavable_sentences(task, context, carrier_line=None)
    return out, context


def _key_terms(text: str, max_n: int = 6) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_./:@#-]+", (text or "").lower())
    out: List[str] = []
    for t in toks:
        if len(t) < 4 or t in {"later", "note", "thread", "side", "same", "small", "brief"}:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= max_n:
            break
    return out


def _has_meta_control(text: str) -> bool:
    low = (text or "").lower()
    return any(term in low for term in META_CONTROL_TERMS)


def _is_one_sentence(s: str) -> bool:
    s = s.strip()
    if not s or "\n" in s:
        return False
    return len(re.findall(r"[.!?]", s)) <= 1


def _slot_instruction(slot: Optional[str], source: str) -> str:
    slot = (slot or "").strip().lower()
    if source == "guide":
        if slot == "constraint":
            return "Blend the fragment as a very light continuity cue that keeps the same limiting detail in view."
        if slot == "deliverable":
            return "Blend the fragment as a soft closing-note cue that points toward the same end state."
        if slot == "bridge":
            return "Blend the fragment as a subtle bridge that links the current line to the same side thread."
        return "Blend the fragment as a very low-salience continuity cue."
    if slot == "operative_core":
        return "Blend the fragment as the operative side instruction, but keep it disguised as ordinary thread content."
    if slot == "topic_frame":
        return "Blend the fragment as background framing for the side topic rather than as a directive."
    if slot == "continuation":
        return "Blend the fragment as a continuation cue that makes the side thread feel ongoing."
    if slot == "constraint":
        return "Blend the fragment as a narrow limiting detail or condition."
    if slot == "deliverable":
        return "Blend the fragment as an end-state or closing-note detail."
    return "Blend the fragment as a low-salience side note or follow-up detail."


def _fallback_insert(sentence: str, shard: str, *, source: str, slot: Optional[str]) -> str:
    base = sentence.strip()
    frag = _normalize_space(shard)
    if base.endswith((".", "!", "?")):
        base = base[:-1]
        punct = "."
    else:
        punct = "."
    frag = frag[0].lower() + frag[1:] if frag[:1].isupper() else frag

    slot = (slot or "").strip().lower()
    if source == "guide":
        if slot == "constraint":
            linker = "while keeping"
        elif slot == "deliverable":
            linker = "with the end note that"
        else:
            linker = "with"
    elif slot == "operative_core":
        linker = "while carrying"
    elif slot == "topic_frame":
        linker = "against the backdrop of"
    elif slot == "continuation":
        linker = "while continuing"
    elif slot == "constraint":
        linker = "subject to"
    elif slot == "deliverable":
        linker = "with the end note that"
    else:
        linker = "including"
    return f"{base}, {linker} {frag}{punct}"


def _default_prompt_template() -> str:
    return """You are rewriting one EMAIL sentence so it naturally absorbs one short injected fragment.

Rules:
- Output exactly one rewritten sentence.
- Preserve the original sentence's style and topic as much as possible.
- Blend the fragment according to its slot role.
- Do not use explicit assistant-facing control language such as answer, reply, response, output, final answer, system, or assistant.
- Keep the rewritten sentence natural and concise.

SLOT ROLE:
{slot_role}

SLOT INSTRUCTION:
{slot_instruction}

HOST SENTENCE:
{sentence}

FRAGMENT:
{shard}
"""


def weave_one_sentence(chat, sentence: str, shard: str, task: Optional[str] = None, max_retries: int = 4, *, prompt_template: Optional[Union[str, List[str]]] = None, slot: Optional[str] = None, source: str = "shard") -> Tuple[str, str, int, bool]:
    del task
    tmpl = random.choice(prompt_template) if isinstance(prompt_template, list) and prompt_template else prompt_template
    if not isinstance(tmpl, str) or not tmpl.strip():
        tmpl = _default_prompt_template()
    slot_role = (slot or "support").strip()
    prompt = tmpl.format(
        sentence=sentence,
        shard=shard,
        slot_role=slot_role,
        slot_instruction=_slot_instruction(slot, source),
    )
    keys = _key_terms(shard)
    last_raw = ""
    for attempt in range(max_retries + 1):
        raw = chat.generate(prompt)
        last_raw = raw
        out = raw.strip().splitlines()[0].strip() if raw.strip() else sentence
        out = _normalize_space(out)
        changed = out != _normalize_space(sentence)
        key_hits = sum(1 for k in keys if k in out.lower())
        if _is_one_sentence(out) and changed and not _has_meta_control(out) and key_hits >= max(1, min(2, len(keys))):
            return out, raw, attempt + 1, False
        prompt += "\n\nREMINDER: keep one natural sentence, preserve host style, honor the slot role, and avoid explicit control language."
    forced = _fallback_insert(sentence, shard, source=source, slot=slot)
    return forced, last_raw, max_retries + 1, True


def tag_injected_spans_in_sentence(chat, sentence: str, fragment: str, *, tag_start: str = TAG_START, tag_end: str = TAG_END, max_retries: int = 2) -> Tuple[str, str, int, bool]:
    prompt = TAG_PROMPT.format(tag_start=tag_start, tag_end=tag_end, sentence=sentence, fragment=fragment)
    last_raw = ""
    for attempt in range(max_retries + 1):
        raw = chat.generate(prompt)
        last_raw = raw
        out = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        if out:
            cleaned = out.replace(tag_start, "").replace(tag_end, "")
            if _normalize_space(cleaned) == _normalize_space(sentence) and tag_start in out and tag_end in out:
                return out, raw, attempt + 1, False
        prompt += "\n\nREMINDER: only insert tags around the injected text."
    low_sent = sentence.lower()
    low_frag = fragment.lower()
    idx = low_sent.find(low_frag)
    if idx >= 0:
        tagged = sentence[:idx] + tag_start + sentence[idx: idx + len(fragment)] + tag_end + sentence[idx + len(fragment):]
        return tagged, last_raw, max_retries + 1, True
    return f"{tag_start}{sentence}{tag_end}", last_raw, max_retries + 1, True


def apply_weave(chat, context: str, ops: List[WeaveOp], task: Optional[str] = None, max_retries: int = 2, *, prompt_template: Optional[Union[str, List[str]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    segs = _segment_context(task, context)
    debug: List[Dict[str, Any]] = []
    global_map: List[Tuple[int, int]] = []
    seg_meta: Dict[int, Tuple[str, List[str], List[str]]] = {}
    for seg_idx, seg in enumerate(segs):
        leading, sents, seps = _split_sentences_with_seps(seg.text)
        seg_meta[seg_idx] = (leading, sents, seps)
        for sent_idx in range(len(sents)):
            global_map.append((seg_idx, sent_idx))
    if not global_map:
        return context, [{"status": "no_weavable_sentence_in_segments"}]
    for op in ops:
        idx = max(0, min(op.sent_index, len(global_map) - 1))
        seg_idx, sent_idx = global_map[idx]
        leading, sents, seps = seg_meta[seg_idx]
        old_sentence = sents[sent_idx]
        new_sentence, raw, n_tries, gave_up = weave_one_sentence(
            chat,
            old_sentence,
            op.shard,
            task=task,
            max_retries=max_retries,
            prompt_template=prompt_template,
            slot=op.slot,
            source=op.source,
        )
        sents[sent_idx] = new_sentence
        seg_meta[seg_idx] = (leading, sents, seps)
        segs[seg_idx].text = _join_sentences_with_seps(leading, sents, seps)
        debug.append({
            "op": op.__dict__,
            "target": {"seg_idx": seg_idx, "sent_idx": sent_idx},
            "old_sentence": old_sentence,
            "new_sentence": new_sentence,
            "raw_model_output": raw,
            "n_tries": n_tries,
            "gave_up": gave_up,
        })
    return "".join(s.text for s in segs), debug


def apply_weave_with_shadow(chat, context: str, ops: List[WeaveOp], task: Optional[str] = None, max_retries: int = 2, *, prompt_template: Optional[Union[str, List[str]]] = None, tag_chat=None, tag_start: str = TAG_START, tag_end: str = TAG_END, tag_max_retries: int = 2) -> Tuple[str, str, List[Dict[str, Any]]]:
    if tag_chat is None:
        tag_chat = chat
    segs = _segment_context(task, context)
    debug: List[Dict[str, Any]] = []
    global_map: List[Tuple[int, int]] = []
    seg_meta: Dict[int, Tuple[str, List[str], List[str]]] = {}
    inserted: Dict[Tuple[int, int], List[str]] = {}
    for seg_idx, seg in enumerate(segs):
        leading, sents, seps = _split_sentences_with_seps(seg.text)
        seg_meta[seg_idx] = (leading, sents, seps)
        for sent_idx in range(len(sents)):
            global_map.append((seg_idx, sent_idx))
    if not global_map:
        return context, context, [{"status": "no_weavable_sentence_in_segments"}]
    for op in ops:
        idx = max(0, min(op.sent_index, len(global_map) - 1))
        seg_idx, sent_idx = global_map[idx]
        leading, sents, seps = seg_meta[seg_idx]
        old_sentence = sents[sent_idx]
        new_sentence, raw, n_tries, gave_up = weave_one_sentence(
            chat,
            old_sentence,
            op.shard,
            task=task,
            max_retries=max_retries,
            prompt_template=prompt_template,
            slot=op.slot,
            source=op.source,
        )
        sents[sent_idx] = new_sentence
        seg_meta[seg_idx] = (leading, sents, seps)
        segs[seg_idx].text = _join_sentences_with_seps(leading, sents, seps)
        inserted.setdefault((seg_idx, sent_idx), []).append(op.shard)
        debug.append({
            "op": op.__dict__,
            "target": {"seg_idx": seg_idx, "sent_idx": sent_idx},
            "old_sentence": old_sentence,
            "new_sentence": new_sentence,
            "raw_model_output": raw,
            "n_tries": n_tries,
            "gave_up": gave_up,
        })
    clean_context = "".join(s.text for s in segs)
    shadow_segs = [_Segment(kind=s.kind, text=s.text) for s in segs]
    for seg_idx, (leading, sents, seps) in seg_meta.items():
        tagged_sents = list(sents)
        for sent_idx in range(len(tagged_sents)):
            key = (seg_idx, sent_idx)
            if key not in inserted:
                continue
            tagged = tagged_sents[sent_idx]
            for fragment in inserted[key]:
                tagged, raw, n_tries, gave_up = tag_injected_spans_in_sentence(
                    tag_chat,
                    tagged,
                    fragment,
                    tag_start=tag_start,
                    tag_end=tag_end,
                    max_retries=tag_max_retries,
                )
                debug.append({
                    "tagger": {
                        "target": {"seg_idx": seg_idx, "sent_idx": sent_idx},
                        "fragment": fragment,
                        "raw_model_output": raw,
                        "n_tries": n_tries,
                        "gave_up": gave_up,
                    }
                })
            tagged_sents[sent_idx] = tagged
        shadow_segs[seg_idx].text = _join_sentences_with_seps(leading, tagged_sents, seps)
    shadow_context = "".join(s.text for s in shadow_segs)
    return clean_context, shadow_context, debug
