#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd,yaml
from dsarp.ranking.metrics import bootstrap_groups,evaluate_groups
if __name__=="__main__":
    frame=pd.read_csv("outputs/rank_datasets/candidate_rank_dataset.csv"); groups=yaml.safe_load(Path("configs/ranking/ranker.yaml").read_text())["group_levels"]["candidate"]; result={"metrics":evaluate_groups(frame,groups),"bootstrap":bootstrap_groups(frame,groups)}; Path("outputs/evaluations/ranker_evaluation.json").write_text(json.dumps(result,indent=2))
