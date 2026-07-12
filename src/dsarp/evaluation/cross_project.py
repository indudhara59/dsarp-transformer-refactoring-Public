"""Cross-project and cross-version evaluation manifests."""
from typing import Any

import pandas as pd

from dsarp.datasets.splitters import leave_one_project_out, temporal_split


def leave_one_project_out_manifest(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{"held_out_project":project,"train_candidates":train.candidate_id.tolist(),"test_candidates":test.candidate_id.tolist()} for project,train,test in leave_one_project_out(frame)]


def cross_version_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    return temporal_split(frame)[["candidate_id","project","version_id","split"]]

