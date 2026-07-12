import json
from pathlib import Path
import pandas as pd
import pytest
from dsarp.llm.backends import MockBackend
from dsarp.llm.cache import LLMCache
from dsarp.llm.parsing import validate_response
from dsarp.llm.validator import validate_candidates


def response(): return {"valid":True,"recommendation_id":"split","applicability_score":.8,"benefit_score":.7,"risk_score":.3,"semantic_preservation_score":.8,"evidence_strength":.6,"confidence_score":.7,"reasoning":["Metrics support it"],"warnings":[],"implementation_steps":["Inspect cohesion"],"insufficient_evidence":False,"requested_additional_context":[]}


def test_malformed_json_repair_and_schema():
    value=validate_response("```json\n"+json.dumps(response())[:-1]+",}\n```","split"); assert value.valid


def test_cache_prevents_second_call(tmp_path:Path):
    config={"cache_dir":str(tmp_path/"cache"),"responses":str(tmp_path/"responses.jsonl"),"errors":str(tmp_path/"errors.jsonl"),"top_n":1,"retry_limit":0,"prompt_version":"v","temperature":0,"max_tokens":10}; backend=MockBackend(response()); frame=pd.DataFrame([{"candidate_id":"c","recommendation_id":"split","project":"p","version_id":"v","ranker_score":1}]); validate_candidates(frame,backend,config); validate_candidates(frame,backend,config); assert backend.calls==1


def test_wrong_recommendation_rejected():
    with pytest.raises(ValueError): validate_response(json.dumps(response()),"other")

