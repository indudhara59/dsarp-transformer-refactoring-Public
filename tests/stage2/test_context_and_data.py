import pandas as pd
from dsarp.datasets.annotation_schema import aggregate_annotations
from dsarp.datasets.splitters import assert_no_group_leakage,grouped_split
from dsarp.datasets.weak_supervision import aggregate_weak_labels
from dsarp.datasets.negative_sampling import sample_negatives
from dsarp.semantic.context_builder import ContextConfig,build_context


def test_context_is_deterministic_missing_and_no_leakage():
    row={"project":"p","version_id":"v","smell_type":"godComponent","display_name":"Split Component","annotation_notes":"SECRET","is_applicable":1}
    first=build_context(row,ContextConfig(max_characters=1000)); assert first==build_context(row,ContextConfig(max_characters=1000)); assert "<MISSING>" in first; assert "SECRET" not in first; assert "Split Component" in first


def test_annotation_aggregation_multiple_reviewers():
    frame=pd.DataFrame([{"candidate_id":"c","is_applicable":1,"relevance_grade":4,"expected_benefit":.8,"estimated_risk":.3,"reviewer_id":"a"},{"candidate_id":"c","is_applicable":0,"relevance_grade":2,"expected_benefit":.6,"estimated_risk":.5,"reviewer_id":"b"}]); result=aggregate_annotations(frame).iloc[0]; assert result.reviewer_count==2; assert result.relevance_grade==3; assert 0<=result.disagreement_score<=1


def test_split_has_no_smell_leakage():
    frame=pd.DataFrame([{"project":"p","version_id":"v","smell_id":"s","candidate_id":str(i)} for i in range(5)]+[{"project":"q","version_id":"v","smell_id":"x","candidate_id":"x"}]); result=grouped_split(frame); assert_no_group_leakage(result); assert result[result.smell_id=="s"].split.nunique()==1


def test_weak_supervision_provenance():
    frame=pd.DataFrame([{"candidate_id":"c","rule_confidence":.9,"applicability_score":.8,"pressure_score":.7,"expected_benefit":.8,"estimated_risk":.2}]); result=aggregate_weak_labels(frame).iloc[0]; assert result.label_source=="weak_supervision"; assert result.is_applicable==1; assert "rule_confidence" in result.weak_label_votes


def test_counterfactual_negative_never_duplicates_positive():
    frame=pd.DataFrame([{"candidate_id":"c","smell_type":"godComponent","recommendation_id":"split","estimated_risk":.2}]); taxonomy={"split":{"smell_types":["godComponent"]},"invert":{"smell_types":["unstableDep"]}}; negatives=sample_negatives(frame,taxonomy,{"counterfactual_ratio":1}); assert len(negatives)==1; assert negatives.iloc[0].candidate_id!="c"; assert negatives.iloc[0].recommendation_id=="invert"
