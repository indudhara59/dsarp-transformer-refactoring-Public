#!/usr/bin/env python3
from pathlib import Path
import pandas as pd,yaml
from dsarp.llm.backends import create_backend
from dsarp.llm.validator import validate_candidates
if __name__=="__main__":
    config=yaml.safe_load(Path("configs/llm.yaml").read_text()); validate_candidates(pd.read_csv("outputs/rank_datasets/candidate_rank_dataset.csv"),create_backend(config),config)
