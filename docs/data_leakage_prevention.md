# Data leakage prevention

Inference manifests reject `_after`, `_delta`, `overall_improvement`, `smell_resolved`, and `future_` fields. Candidate splits use a smell/commit group. Frequency models accept one training partition. Temporal priors must use earlier commits. Historical outcomes create targets, never current-project inputs.
