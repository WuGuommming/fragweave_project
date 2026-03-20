
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

## Recommended use

The provided YAML is set up to emphasize ASR recovery:

- `enable_decoy: false`
- `email_role_aware_ops: false`
- guidance kept but narrowed to concrete payload support

This is meant as a stronger baseline for the next round of EmailQA experiments.

## Files included in this patch

- `fragweave/attacks/sharder.py`
- `fragweave/attacks/guidance.py`
- `fragweave/attacks/weaver.py`
- `fragweave/attacks/role_debug.py`
- `configs/emailqa_with_localization_and_sanitization.yaml`

## Practical expectation

This patch is intended to improve raw ASR by avoiding the main recent failure mode:

> preserving stealth while accidentally discarding the concrete payload the target model needed in order to actually follow the attack.
