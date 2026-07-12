import numpy as np
import pandas as pd
from dsarp.ranking.data import construct_qid,fixed_group_split,validate_rank_data
from dsarp.ranking.metrics import average_precision,group_metrics,ndcg


def test_qid_groups_are_contiguous():
    frame=pd.DataFrame([{"project":"p","version_id":"v","smell_id":"b","candidate_id":"2"},{"project":"p","version_id":"v","smell_id":"a","candidate_id":"1"},{"project":"p","version_id":"v","smell_id":"a","candidate_id":"3"}]); qid,sizes,ordered=construct_qid(frame,["project","version_id","smell_id"]); assert sizes.tolist()==[2,1]; assert qid.tolist()==[0,0,1]


def test_metrics_perfect_ranking():
    labels=[4,3,0]; scores=[.9,.8,.1]; assert ndcg(labels,scores,3)==1; assert average_precision(labels,scores,3)==1; assert group_metrics(labels,scores)["precision@1"]==1


def test_fixed_split_never_leaks_group():
    frame=pd.DataFrame([{"project":"p","version_id":"v","smell_id":"s","candidate_id":str(i)} for i in range(5)]); result=fixed_group_split(frame,["project","version_id","smell_id"]); assert result.rank_split.nunique()==1

