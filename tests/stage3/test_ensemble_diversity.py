from pathlib import Path
import pandas as pd,yaml
from dsarp.diversity.mmr import diversify
from dsarp.ensemble.scoring import apply_ensemble,minmax


def config(): return yaml.safe_load(Path("configs/ensemble.yaml").read_text())


def rows():
    return pd.DataFrame([{"candidate_id":"a","project":"p","version_id":"v","smell_id":"s","smell_type":"godComponent","affected_component_id":"c","recommendation_id":"split","recommendation_family":"split","ranker_score":.9,"predicted_relevance_grade":4,"rule_confidence":.8,"predicted_benefit":.8,"predicted_risk":.2,"valid":False,"confidence_score":.9,"warnings":["grounded contraindication"],"severity":1,"strength":1,"atdi":1,"fan_in":1,"fan_out":1},{"candidate_id":"b","project":"p","version_id":"v","smell_id":"s","smell_type":"godComponent","affected_component_id":"c","recommendation_id":"facade","recommendation_family":"boundary","ranker_score":.8,"predicted_relevance_grade":3,"rule_confidence":.7,"predicted_benefit":.7,"predicted_risk":.3,"valid":True,"confidence_score":.8,"warnings":[],"severity":1,"strength":1,"atdi":1,"fan_in":1,"fan_out":1}])


def test_normalization_ensemble_rejection_and_confidence():
    ranked,rejected=apply_ensemble(rows(),config()); assert ranked.final_score.between(0,1).all(); assert len(rejected)==1; assert rejected.iloc[0].validation_status=="LLM-rejected"; assert set(ranked.confidence_band)<={"High","Medium","Low"}


def test_mmr_limits_same_smell():
    ranked,_=apply_ensemble(rows(),config()); diversified=diversify(ranked,10,{**config()["diversity"],"max_per_smell":1}); assert len(diversified)==1

