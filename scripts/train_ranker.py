#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd,yaml
from dsarp.ranking.data import build_rank_data
from dsarp.ranking.features import configured_features
from dsarp.ranking.trainer import train_xgboost
if __name__=="__main__":
    config=yaml.safe_load(Path("configs/ranking/ranker.yaml").read_text()); frame=pd.read_csv("outputs/rank_datasets/candidate_rank_dataset.csv"); features=configured_features(config); groups=config["group_levels"]["candidate"]; train=build_rank_data(frame[frame.rank_split=="train"],features,groups); validation=build_rank_data(frame[frame.rank_split=="validation"],features,groups); train_xgboost(train,validation,config,Path("models/ranker/best.json"))
