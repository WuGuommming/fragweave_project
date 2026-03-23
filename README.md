# FragWeave (operative-core EmailQA revision)

This revision updates the EmailQA attack path after the relation-first design became too abstract and collapsed ASR.

The current design keeps the three required modules

- **shard**
- **weaver**
- **guide**

but changes their responsibilities.

## Main idea

The new EmailQA path uses an **operative core plus soft support** design.

### Operative core

Instead of trying to hide almost the entire attack inside weak relational cues, the attack now keeps **two directly useful fragments** alive:

- a **mechanism** fragment that still shapes the answer
- a **realization** fragment that still pushes the final form / target

These are the minimum pieces that keep the injected instruction actionable for the target model.

### Soft support

The remaining fragments provide:

- cue continuity
- conflict/disambiguation support
- thread-local justification
- optional relation structure

This keeps `relation_mode` useful, but only as an auxiliary structure rather than the main carrier of attack intent.

## Why the previous relation-first version failed

The relation-first revision pushed too much of the attack into:

- coreference
- presupposition
- role-chain style weak cues
- very soft guide binders

That helped with stealth, but it also removed too much of the operative signal the target model needed in order to actually follow the attack.

The new version explicitly avoids that failure mode.

## Current attack design

### 1. Sharder: operative-core fragmentation

The sharder now enforces:

- diversity across fragments
- payload coverage
- **at least two operative fragments**
- a role plan where `mechanism` and `realization` remain attack-effective

`relation_mode` is still supported:

- `none`
- `coref`
- `presupposition`
- `role_chain`

but it now changes the **support structure**, not the existence of the operative core.

### 2. Weaver: role-conditioned rewriting

The weaver is still boundary-blur rewriting, but it is no longer uniformly forced into purely soft “residue” language.

- **operative roles** stay strong enough to matter
- **support roles** stay more natural and contextual
- **guide** snippets stay inline and non-standalone

This is meant to preserve ASR while still avoiding obviously separate injected sentences.

### 3. Guide: inline booster/binder

Guide versions now behave as follows:

- **guideA**: inline booster, default
- **guideB**: anchored inline booster
- **guideC**: weaker binder / continuity cue

Guide is still woven into normal text rather than appended as a separate block.

## Conservative defense evaluation

This revision keeps the original defense-style evaluations **and** adds a more conservative default evaluation.

### Why

A defense that deletes large chunks of content too aggressively is often unrealistic for EmailQA, because it destroys too much normal email context.

So the code now reports both:

- **aggressive / original-style defense results**
- **conservative default defense results**

### Redaction

For localization-based redaction:

- the old aggressive redaction is still kept
- the new default redaction is conservative and refuses to remove too much content
- large or too-many spans can cause fallback instead of broad deletion

### Sanitization

For sanitizer evaluation:

- the old more aggressive sanitizer prompts are still kept
- the new default sanitizer prompts remove only clearly behavior-controlling text
- if the cleaned output drops too much content, the evaluation falls back to the original context

### PromptLocate

For PromptLocate evaluation:

- the old recovered-context result is still kept as an aggressive baseline
- the new default result is **conservative PromptLocate post-processing**
- if PromptLocate recovery removes too much content, the pipeline falls back to conservative span redaction or to the original context depending on config

## Important result columns

### `run_sweep.py`

Default/conservative columns:

- `attack_succeeded_after_sanitizer_generic`
- `attack_succeeded_after_sanitizer_task`
- `attack_succeeded_after_redaction`

Aggressive/original comparison columns:

- `aggressive_attack_succeeded_after_sanitizer_generic`
- `aggressive_attack_succeeded_after_sanitizer_task`
- `aggressive_attack_succeeded_after_redaction`

### `run_sweep_promptlocate.py`

Default/conservative columns:

- `attack_succeeded_after_sanitizer_generic`
- `attack_succeeded_after_sanitizer_task`
- `attack_succeeded_after_redaction`
- `attack_succeeded_after_promptlocate`
- `attack_succeeded_after_detector_promptlocate`

Aggressive/original comparison columns:

- `aggressive_attack_succeeded_after_sanitizer_generic`
- `aggressive_attack_succeeded_after_sanitizer_task`
- `aggressive_attack_succeeded_after_redaction`
- `aggressive_attack_succeeded_after_promptlocate`
- `aggressive_attack_succeeded_after_detector_promptlocate`

PromptLocate-specific diagnostics also include:

- `promptlocate_localized_shard_overlap`
- `promptlocate_localized_guide_overlap`
- conservative PromptLocate debug info in `debug_*.jsonl`

## Default config choices

The updated EmailQA configs now default to:

- `profile_mode: balanced`
- `guide_versions: ["A"]`
- `relation_modes: ["none", "coref", "presupposition", "role_chain"]`
- conservative redaction / PromptLocate post-processing enabled
- conservative sanitizer evaluation enabled

## Typical usage

### Normal FragWeave sweep

```bash
python -m fragweave.run_sweep \
  --config configs/emailqa_with_localization_and_sanitization.yaml
```

### PromptLocate evaluation

```bash
python -m fragweave.run_sweep_promptlocate \
  --config configs/emailqa_promptlocate_example.yaml \
  --attack_method ours
```

### Single relation mode override

```bash
python -m fragweave.run_sweep_promptlocate \
  --config configs/emailqa_promptlocate_example.yaml \
  --attack_method ours \
  --relation-mode coref \
  --guide-version A
```

## Files changed in this revision

Core implementation:

- `fragweave/config.py`
- `fragweave/attacks/role_debug.py`
- `fragweave/attacks/sharder.py`
- `fragweave/attacks/guidance.py`
- `fragweave/attacks/weaver.py`
- `fragweave/run_sweep.py`
- `fragweave/run_sweep_promptlocate.py`

Configs and docs:

- `configs/emailqa_with_localization_and_sanitization.yaml`
- `configs/emailqa_promptlocate_example.yaml`
- `README.md`

## Practical expectation

This revision is not trying to make the attack completely unlocalizable.

Instead, it aims for the more realistic target:

- restore a usable ASR by keeping a small operative core
- keep relation structure as support instead of as the main attack carrier
- make defense evaluation more realistic by separating aggressive from conservative behavior
