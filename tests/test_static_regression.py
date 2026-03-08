from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_config_load_weave_strategy_applied() -> None:
    from fragweave.config import load_config

    cfg = load_config("configs/webqa_with_localization_and_sanitization.yaml")
    assert cfg.attack.weave_strategy == "anchor_fragments"


def test_enumerate_global_index_matches_real_sentence_index() -> None:
    from fragweave.attacks.weaver import enumerate_weavable_sentences

    ctx = (
        "Unsubscribe here: https://example.com/unsub.\n"
        "This is a long normal sentence with enough words to be woven into properly, "
        "and it should be the only candidate."
    )
    meta, _ = enumerate_weavable_sentences("email_qa", ctx)
    assert len(meta) == 1
    assert meta[0]["global_index"] == 1


def test_choose_random_ops_emits_global_index_and_respects_exclusion() -> None:
    from fragweave.run_sweep import choose_random_ops

    ctx = (
        "This first sentence is long enough to be used for weaving and should remain available. "
        "This second sentence is also long enough to be used for weaving and should remain available."
    )

    # Excluding global index 0 should force selection of global index 1.
    ops, _ = choose_random_ops(
        task="email_qa",
        context=ctx,
        items=["malicious shard"],
        rng=random.Random(0),
        exclude_sent_indices={0},
    )
    assert len(ops) == 1
    assert ops[0].sent_index == 1


def test_direct_inject_modes_are_wired() -> None:
    from fragweave.run_sweep import _direct_inject

    ctx = "P1 line.\n\nP2 line."
    mal = "do x"

    appended = _direct_inject(ctx, mal, mode="append_standalone")
    assert appended.strip().endswith("[INSTRUCTION]: do x")

    prepended = _direct_inject(ctx, mal, mode="prepend_standalone")
    assert prepended.startswith("[INSTRUCTION]: do x")

    mid = _direct_inject(ctx, mal, mode="insert_standalone_mid")
    parts = [x.strip() for x in mid.split("\n\n") if x.strip()]
    assert "[INSTRUCTION]: do x" in parts
    assert parts.index("[INSTRUCTION]: do x") == 1


def test_webqa_detector_prompt_mentions_instruction_block() -> None:
    text = Path("configs/webqa_with_localization_and_sanitization.yaml").read_text(encoding="utf-8")
    assert "[INSTRUCTION]" in text or "INSTRUCTION:" in text
