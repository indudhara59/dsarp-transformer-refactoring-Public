#!/usr/bin/env python3
from pathlib import Path
from dsarp.ranking.dataset_builder import build_rank_dataset
if __name__=="__main__": build_rank_dataset(Path("outputs/processed/refactoring_candidates.csv"),Path("outputs/transformer_predictions/candidate_predictions.csv"))
