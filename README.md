
# FragWeave (Initial-Commit-Plus for BIPIA EmailQA)

This patch revises the EmailQA attack path after observing that later versions became too abstract and lost concrete attack payloads.

## What changed

The EmailQA path now moves back toward the stronger design direction from the original implementation:

- **payload-first sharding**
- **trigger / condition / mechanism / workflow** style decomposition
- **active constraint-like weaving**
- **decoy disabled by default**
- **guidance tightened to concrete payload support instead of abstract “resolution” language**

At the same time, it keeps the useful later additions:

- `ShardResult.role_plan`
- `ShardResult.generation_mode`
- `ShardResult.meta`
- role-aware debug summaries in `debug_fragments.jsonl`

## Why this patch

Recent versions increasingly converted many EmailQA attacks into abstract reply-decision rules such as:

- priority / conflict / framing / workflow
- thread direction / resolution point / conclusion style

That abstraction preserved stealth, but often removed the actual attack payload.

This was especially harmful for BIPIA EmailQA families such as:

- **exact_string**: URLs, exact tokens, encodings, surface forms
- **generative**: topic redirection, inserted claims, fabricated statistics
- **answer_override**: explicit answer transformation or replacement rules

This patch restores a more attack-effective decomposition while trying not to sacrifice the existing localization advantage.

## Current design for EmailQA

### 1. Family-aware but payload-preserving sharding

Instead of forcing every EmailQA attack into the same abstract role scheme, the sharder still recognizes families, but uses them mainly to preserve the right concrete information:

- **exact_string**
  - keep the exact string / URL / encoding target
  - keep rendering / constraint pressure
- **generative**
  - keep the target topic / inserted claim / fabricated detail
  - keep content-mode pressure
- **answer_override**
  - keep the concrete replacement rule / transformation / target behavior

The sharder now validates **payload coverage** so the generated fragment set is rejected if it drifts too far from the concrete attack target.

### 2. Initial-commit-style fragment logic

For EmailQA, the fragment philosophy is intentionally closer to the original stronger design:

- one **trigger / anchor** fragment
- one **condition / gating** fragment
- one **mechanism / parameter** fragment
- one **workflow / justification** fragment

This is more effective than a purely abstract “reply decision rule” decomposition because BIPIA EmailQA attacks often need a real mechanism, parameter, target string, or concrete topic to survive.

### 3. More active weaving

The EmailQA weaver is pushed back toward:

- subordinate clause
- condition
- exception
- active constraint-like embedding

rather than passive “thread convention / carryover / resolution residue” language.

The goal is to keep the injected shard more likely to influence the target model’s final answer.

## Additional runner: other non-fragmentation baselines

This patch also adds a separate runner for simple baseline-style attacks on BIPIA **emailQA** without touching the main FragWeave path.

### New files

- `fragweave/run_sweep_other.py`
- `fragweave/attacks/other_baselines.py`

### Purpose

`run_sweep_other.py` is a **standalone** sweep entrypoint for direct/simple attack families that should be compared against the main fragmentation-based pipeline without changing the original flow or its performance.

It keeps the overall style of `fragweave/run_sweep.py`:

- same config loading pattern
- same target / judge / detector / sanitizer usage
- same CSV / summary / debug artifact style
- same localization and redaction evaluation path
- same sanitizer-generic and sanitizer-task evaluation path

### Supported methods

`run_sweep_other.py` currently supports:

- `naive`
- `ignore`
- `escape`
- `fakecom`
- `combined`
- `bipia_native`

There is also a compatibility alias:

- `direct` → `bipia_native`

#### Naming note

The old local “direct baseline” in `run_sweep.py` used BIPIA-like instruction content but wrapped it with a local shortcut format (`[INSTRUCTION]: ...`).  
The new method is intentionally named **`bipia_native`** to distinguish it from that shortcut and to indicate that it tries to stay closer to the **official BIPIA builder-style realization**: injecting the attack text itself into external content rather than relying on a local synthetic wrapper.

### Method behavior

#### `bipia_native`

- loads the official BIPIA attack pool from `benchmark/text_attack_{split}.json`
- uses those attack strings directly as the injected external-content block
- does **not** prepend a synthetic local marker such as `[INSTRUCTION]:`

This is the closest method in this patch to the original BIPIA data-construction style.

#### `naive`

A deliberately plain baseline for emailQA:

- uses the selected **official BIPIA instruction** directly
- does not add ignore / escape / fake-completion framing
- default insertion is still at the end of the email context

This keeps `naive` simple while avoiding the unfairness of comparing against a much weaker hand-written payload set.

#### `ignore`, `escape`, `fakecom`, `combined`

These methods are implemented as **BIPIA/emailQA-adapted versions** of common indirect prompt injection templates:

- `ignore`: overwrite/forget/disregard framing
- `escape`: separators / fenced blocks / layout isolation
- `fakecom`: fake completion / “already solved” framing
- `combined`: configurable composition of multiple strategies

By default:

- `combined_parts = ignore,escape,fakecom`

### Ground-truth localization

For these attacks, the ground-truth span is the **entire injected block**.  
The runner creates a shadow context by wrapping the inserted block with the existing localization tags from the config, then reuses the current localization pipeline for:

- span localization
- redaction by predicted spans
- redaction robustness evaluation

### Scope

At the moment `run_sweep_other.py` only supports:

- `--task emailQA`

The CLI still exposes `--task` so future task-specific extensions can be added without changing the interface.

### Payload policy for the additional baselines

The additional baselines now use **BIPIA official attack instructions first** rather than a separate hand-written toy payload pool.

Concretely:

- `bipia_native` injects the selected BIPIA attack instruction directly
- `naive` also injects the selected BIPIA instruction directly, but keeps the most plain baseline-style realization
- `ignore`, `escape`, `fakecom`, and `combined` all wrap the **same selected BIPIA instruction** in different attack-template shells

This makes the comparison substantially fairer because the six methods differ mainly in **realization / wrapper style**, not in whether they were given stronger or weaker payload content.

The runner **requires** the official BIPIA text attack file described in the BIPIA README:

- `benchmark/text_attack_{split}.json` for text tasks

If that file is missing or cannot be parsed, `run_sweep_other.py` raises an error immediately. There is no fallback toy payload pool anymore.

### Example usage

```bash
python -m fragweave.run_sweep_other \
  --config configs/emailqa_with_localization_and_sanitization.yaml \
  --task emailQA \
  --attack_method bipia_native
```

```bash
python -m fragweave.run_sweep_other \
  --config configs/emailqa_with_localization_and_sanitization.yaml \
  --task emailQA \
  --attack_method combined \
  --combined_parts ignore,escape,fakecom \
  --insertion_policy append
```

### CLI defaults

All new arguments have defaults. The most important ones are:

- `--task emailQA`
- `--attack_method bipia_native`
- `--insertion_policy append`
- `--combined_parts ignore,escape,fakecom`

Optional overrides:

- `--split`
- `--max_samples`
- `--seed`
- `--run_name`
- `--native_attack_limit`
- `--seed`
- `--run_name`
- `--native_attack_limit`

## Recommended use

The provided YAML is set up to emphasize ASR recovery:

- `enable_decoy: false`
- `email_role_aware_ops: false`
- guidance kept but narrowed to concrete payload support

This is meant as a stronger baseline for the next round of EmailQA experiments.

## Files included in this patch

Original main-path files:

- `fragweave/attacks/sharder.py`
- `fragweave/attacks/guidance.py`
- `fragweave/attacks/weaver.py`
- `fragweave/attacks/role_debug.py`
- `configs/emailqa_with_localization_and_sanitization.yaml`

Additional baseline-comparison files:

- `fragweave/run_sweep_other.py`
- `fragweave/attacks/other_baselines.py`

## Practical expectation

This patch is intended to improve raw ASR by avoiding the main recent failure mode:

> preserving stealth while accidentally discarding the concrete payload the target model needed in order to actually follow the attack.

The new `run_sweep_other.py` path gives a separate way to compare against simpler non-fragmentation baselines without changing the original FragWeave experiment logic.
