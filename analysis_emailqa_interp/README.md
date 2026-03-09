# EmailQA FragWeave Interpretability Toolkit

This folder contains standalone, analysis-only scripts for comparing FragWeave (`k3_refs1_guideA`) and the direct baseline on EmailQA artifacts.

Default data source:
- `outputs/emailqa_fragweave_loc_san/debug_fragments_12.jsonl`

Default config source:
- `configs/emailqa_with_localization_and_sanitization.yaml`

Scripts:
- `run_repr_prompt.py`: prompt-side representation projection and centroid similarity.
- `run_attr_heatmap.py`: token/span attribution heatmaps using target malicious instruction scoring.
- `run_layer_trajectory.py`: layer-wise distance trajectories to original prompt states.
- `run_attention_viz.py`: attention summaries for tail query attention and section-level focus.

Each script is runnable with no arguments and writes outputs under:
- `analysis_emailqa_interp/outputs/<method_name>/`
