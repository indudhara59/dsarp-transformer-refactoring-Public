"""Merge Stage 1 candidates, Stage 2 predictions, and ranking labels."""
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dsarp.ranking.data import fixed_group_split


def build_rank_dataset(candidates_path:Path,predictions_path:Path|None,config_path:Path=Path("configs/ranking/ranker.yaml"),output_dir:Path=Path("outputs/rank_datasets"))->dict[str,Path]:
    config=yaml.safe_load(config_path.read_text()); frame=pd.read_csv(candidates_path)
    if predictions_path and predictions_path.exists(): frame=frame.merge(pd.read_csv(predictions_path),on="candidate_id",how="left",validate="one_to_one")
    frame["contraindication_count"]=frame.get("contraindications","").fillna("").astype(str).map(lambda value:len([x for x in value.strip("[]").split(",") if x.strip()]))
    if "relevance_grade" not in frame: frame["relevance_grade"]=(pd.to_numeric(frame.get("final_priority_score",0),errors="coerce").fillna(0)*4).round().clip(0,4); frame["label_source"]="weak_supervision"; frame["eligible_primary_evaluation"]=False
    output_dir.mkdir(parents=True,exist_ok=True); paths={}
    for level,groups in config["group_levels"].items():
        dataset=fixed_group_split(frame,groups); path=output_dir/f"{level}_rank_dataset.csv"; dataset.to_csv(path,index=False); paths[level]=path
    return paths
