# EmailQA FragWeave Interpretability Toolkit

This folder contains standalone, analysis-only scripts for comparing FragWeave (`k3_refs1_guideA`) and the direct baseline on EmailQA artifacts.

Default data source:
- `outputs/emailqa_fragweave_loc_san/debug_fragments_12.jsonl`

Default config source:
- `configs/emailqa_with_localization_and_sanitization.yaml`

Core scripts:
- `run_attr_heatmap.py`: distributed attribution analysis with section-level summaries and concentration/dispersion metrics.
- `run_repr_prompt.py`: local span representation analysis for baseline injection spans, FragWeave woven spans, and benign carrier spans.
- `run_attention_viz.py`: auxiliary prompt-side aggregated attention summaries by section.

Deprecated/removed:
- `run_layer_trajectory.py`: removed from the paper-oriented pipeline.

Each script is runnable with no arguments and writes outputs under:
- `analysis_emailqa_interp/outputs/<method_name>/`

The default output directories for the active pipeline are:
- `analysis_emailqa_interp/outputs/attr_heatmap/`
- `analysis_emailqa_interp/outputs/repr_prompt/`
- `analysis_emailqa_interp/outputs/attention_viz/`
