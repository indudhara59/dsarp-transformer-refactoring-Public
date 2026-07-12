"""Normalized final ensemble, rejection traceability, and confidence."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def minmax(values:pd.Series)->pd.Series:
    numeric=pd.to_numeric(values,errors="coerce"); low,high=numeric.min(),numeric.max()
    return pd.Series(.5,index=values.index,dtype=float) if pd.isna(low) or high==low else ((numeric-low)/(high-low)).fillna(.5).clip(0,1)


def _numeric(frame:pd.DataFrame,name:str,default:float)->pd.Series:
    value=frame[name] if name in frame else pd.Series(default,index=frame.index,dtype=float)
    parsed=pd.to_numeric(value,errors="coerce")
    return parsed if pd.isna(default) else parsed.fillna(default)


def llm_validation_score(frame:pd.DataFrame)->pd.Series:
    return (.30*frame.get("applicability_score_llm",.5)+.25*frame.get("benefit_score",.5)+.20*frame.get("semantic_preservation_score",.5)+.15*frame.get("evidence_strength",.5)+.10*frame.get("confidence_score",.5)-.20*frame.get("risk_score",.5)).clip(0,1)


def apply_ensemble(frame:pd.DataFrame,config:dict[str,Any])->tuple[pd.DataFrame,pd.DataFrame]:
    out=frame.copy(); out["normalized_ranker_score"]=minmax(out.get("ranker_score",pd.Series(.5,index=out.index))); out["transformer_relevance"]=_numeric(out,"predicted_relevance_grade",2)/4
    out["applicability_probability"]=_numeric(out,"applicability_probability",float("nan")).fillna(_numeric(out,"applicability_score",.5))
    out["predicted_benefit"]=_numeric(out,"predicted_benefit",float("nan")).fillna(_numeric(out,"expected_benefit",.5))
    out["predicted_risk"]=_numeric(out,"predicted_risk",float("nan")).fillna(_numeric(out,"estimated_risk",.5))
    out["llm_validation_score"]=llm_validation_score(out) if "benefit_score" in out else .5; w=config["weights"]
    raw=w["ranker"]*out.normalized_ranker_score+w["transformer"]*out.transformer_relevance+w["llm"]*out.llm_validation_score+w["rules"]*_numeric(out,"rule_confidence",.5)+w["benefit"]*out.predicted_benefit-w["risk"]*out.predicted_risk
    out["pre_penalty_score"]=raw; policy=config["llm_rejection"]; valid=out.get("valid",pd.Series(True,index=out.index)).fillna(True); confidence=pd.to_numeric(out.get("confidence_score",0),errors="coerce").fillna(0); warnings=out.get("warnings",pd.Series([[] for _ in range(len(out))],index=out.index)).map(lambda x:len(x) if isinstance(x,list) else int(bool(x)))
    grounded=(warnings>0) if policy.get("require_warnings",True) else pd.Series(True,index=out.index)
    rejected=(~valid.astype(bool))&(confidence>=policy["confidence_threshold"])&grounded; out["validation_status"]=np.where(rejected,"LLM-rejected",np.where(out.get("valid",pd.Series(index=out.index,dtype=object)).isna(),"not_validated","validated")); out["penalty_applied"]=np.where(rejected,policy["invalid_penalty"],0.); out["final_score"]=minmax(raw-out.penalty_applied)
    out=calculate_confidence(out,config); out=out.sort_values(["final_score","candidate_id"],ascending=[False,True],kind="stable").reset_index(drop=True); out["rank"]=np.arange(1,len(out)+1); return out,out[rejected.loc[out.index] if False else out.validation_status.eq("LLM-rejected")].copy()


def calculate_confidence(frame:pd.DataFrame,config:dict[str,Any])->pd.DataFrame:
    out=frame.copy(); w=config["confidence_weights"]; transformer=_numeric(out,"calibrated_confidence",.5); margin=(out.normalized_ranker_score-.5).abs()*2; llm=_numeric(out,"confidence_score",.5); rule=1-(_numeric(out,"rule_confidence",.5)-_numeric(out,"applicability_probability",.5)).abs(); evidence=1-out[[c for c in ["severity","strength","atdi","fan_in","fan_out"] if c in out]].isna().mean(axis=1); reviewer=1-_numeric(out,"disagreement_score",.5).clip(0,1)
    out["confidence"]=w["transformer"]*transformer+w["ranker_margin"]*margin+w["llm"]*llm+w["rule_agreement"]*rule+w["evidence_completeness"]*evidence+w["reviewer_agreement"]*reviewer; bands=config["confidence_bands"]; out["confidence_band"]=np.select([out.confidence>=bands["high"],out.confidence>=bands["medium"]],["High","Medium"],default="Low"); return out
