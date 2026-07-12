"""Failure-isolated cached top-candidate LLM validation."""
from __future__ import annotations

from typing import Any

import pandas as pd

from dsarp.llm.backends import DisabledBackend,LLMBackend
from dsarp.llm.cache import LLMCache,append_jsonl
from dsarp.llm.parsing import validate_response
from dsarp.llm.prompt import build_validation_prompt


def validate_candidates(frame:pd.DataFrame,backend:LLMBackend,config:dict[str,Any],force:bool=False)->tuple[pd.DataFrame,pd.DataFrame]:
    if isinstance(backend,DisabledBackend): return pd.DataFrame(),pd.DataFrame()
    cache=LLMCache(__import__("pathlib").Path(config["cache_dir"])); successes=[]; errors=[]; decoding={"temperature":config.get("temperature",0),"max_tokens":config.get("max_tokens",1400),**config.get("decoding",{})}
    groups=["project","version_id"]
    top=frame.sort_values("ranker_score",ascending=False,kind="stable").groupby(groups,group_keys=False).head(int(config.get("top_n",20)))
    for _,row in top.iterrows():
        prompt=build_validation_prompt(row.to_dict(),config.get("prompt_version","dsarp-validation-v1")); key=cache.key(backend.identifier,config.get("prompt_version","v1"),str(row.candidate_id),prompt,decoding); cached=None if force else cache.get(key)
        if cached: successes.append(cached); continue
        last_error=None
        for attempt in range(int(config.get("retry_limit",2))+1):
            try:
                response=validate_response(backend.generate(prompt,decoding),str(row.recommendation_id)); record={"candidate_id":row.candidate_id,"response_hash":key,**response.model_dump()}; cache.put(key,record); append_jsonl(__import__("pathlib").Path(config["responses"]),record); successes.append(record); last_error=None; break
            except Exception as exc: last_error=str(exc)
        if last_error:
            record={"candidate_id":row.candidate_id,"response_hash":key,"error":last_error}; append_jsonl(__import__("pathlib").Path(config["errors"]),record); errors.append(record)
    return pd.DataFrame(successes),pd.DataFrame(errors)

