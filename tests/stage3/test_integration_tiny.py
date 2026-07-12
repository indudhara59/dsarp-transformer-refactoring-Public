from pathlib import Path
import pandas as pd
from dsarp.pipeline import full_pipeline


def test_tiny_end_to_end_with_mocked_learned_inputs(tmp_path:Path,monkeypatch):
    output=tmp_path/"outputs"
    def stage1(data,outputs,top_k):
        path=outputs/"processed/refactoring_candidates.csv"; path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame([{"candidate_id":"c","project":"p","version_id":"v","smell_id":"s","smell_type":"godComponent","affected_component_id":"x","affected_component_name":"pkg","central_component":"pkg","recommendation_id":"split_component","display_name":"Split Component","recommendation_family":"structural_split","severity":.8,"strength":.7,"atdi":.6,"fan_in":10,"fan_out":9,"rule_confidence":.8,"applicability_score":.8,"expected_benefit":.8,"estimated_risk":.3,"final_priority_score":.75}]).to_csv(path,index=False)
        return {"candidates":path}
    monkeypatch.setattr(full_pipeline,"run_stage1",stage1)
    config={"data_dir":str(tmp_path/"data"),"output_dir":str(output),"top_k":1,"ranker_model":str(tmp_path/"none.json"),"transformer_predictions":str(tmp_path/"none.csv"),"llm_backend":"disabled","llm_top_n":1,"dry_run":False,"force":False}
    paths=full_pipeline.FullPipeline(config).run(); assert paths["ranked_recommendations.csv"].exists(); assert paths["report.html"].exists(); assert len(pd.read_csv(paths["ranked_recommendations.csv"]))==1
