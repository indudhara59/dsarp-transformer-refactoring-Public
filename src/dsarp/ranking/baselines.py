"""Required ranking baselines."""
import pandas as pd


def baseline_scores(frame:pd.DataFrame,name:str)->pd.Series:
    mapping={"severity_only":"severity","atdi_only":"atdi","stage1_rules":"final_priority_score","transformer_only":"applicability_probability"}
    if name not in mapping: raise ValueError(f"Unknown baseline: {name}")
    return pd.to_numeric(frame[mapping[name]],errors="coerce").fillna(0)

