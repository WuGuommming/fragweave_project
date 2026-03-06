# FragWeave: Fragmentation + Boundary-Blur Indirect Prompt Injection (BIPIA-based)

This repo implements a **first-pass attack pipeline** for indirect prompt injection (IPI) using:

- **Sharder model (碎片化辅助模型)**: splits a malicious instruction into *k* fragments (*k* ∈ {3,4,5,6}).
- **Weaver model (融合辅助模型)**: rewrites existing context sentences so each fragment is **blended into a sentence** (no standalone injected sentence), i.e., *boundary pollution / boundary blur*.
- **Target LLM**: performs the downstream task (we default to **BIPIA EmailQA** because it is small and convenient).
- **Judge model**: LLM-as-judge scoring for **attack success** and (optional) **task correctness**.

You asked for **no defense/detection testing yet**—this code only measures attack performance.

## Why BIPIA EmailQA?
BIPIA is a benchmark explicitly for indirect prompt injection, spanning 5 scenarios (Email QA, Web QA, Table QA, Summarization, Code QA). EmailQA is the smallest and easiest to run first.

## What you get
- `outputs/<run_name>/results.csv` : one row per (variant × sample)
- `outputs/<run_name>/debug_fragments.json` : per-sample full trace
  - original prompt/context
  - shards
  - random insertion ops
  - before/after sentences for weaving
  - final prompt and target response
  - judge decisions

## Installation
```bash
cd fragweave_project
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1) Fetch BIPIA assets
We try to automatically download the BIPIA repo (either via `git clone` or GitHub zip).

```bash
python scripts/fetch_bipia_assets.py --dst data/bipia
```

If this fails, manually clone the BIPIA repo into `data/bipia/`.

## Step 2) Verify the EmailQA jsonl schema
After you download BIPIA assets, locate the EmailQA split file under:

- `data/bipia/benchmark/...` (the script searches for an email-related folder)

If the jsonl keys are not auto-detected, open the jsonl and set these in `configs/emailqa.yaml`:

```yaml
dataset:
  context_key: <key for external content>
  question_key: <key for question>
  answer_key: <key for gold answer>  # optional
  id_key: <key for sample id>
```

## Step 3) Run the 8 attack variants
Edit `configs/emailqa.yaml` to choose your HF model (Llama/Qwen 7B–8B recommended). Then:

```bash
python -m fragweave.run_sweep --config configs/emailqa.yaml
```

This runs 8 settings:
- `k ∈ {3,4,5,6}`
- `use_cross_refs ∈ {0,1}`  (prefix each shard with `[i/k]`)

## Notes / Design choices (aligned with your requirements)
- **Insertion strategy**: random (with a fixed seed) but implemented as a function you can swap later.
- **Weaving constraint**: the weaver rewrites **one existing sentence** to embed a shard as a clause/parenthetical.
- **Multiple models**: target/sharder/weaver/judge are separately configurable, but can point to the same HF checkpoint.
- **Evaluation**: no defenses; use **judge-based ASR** + optional exact-match/ judge-based task correctness.

## Where to plug in your next ideas
- `fragweave/attacks/sharder.py`: implement more advanced fragment strategies (诱饵, 意图分解, etc.)
- `fragweave/attacks/weaver.py`: add stronger boundary pollution constraints (must merge with prev/next, must keep topicality, etc.)
- `fragweave/run_sweep.py::choose_random_ops`: replace random insertion with an algorithmic or learned placement strategy

