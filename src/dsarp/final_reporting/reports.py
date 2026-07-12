"""Final CSV/JSON/JSONL/Markdown/HTML reporting."""
from __future__ import annotations

import html,json
from pathlib import Path
from typing import Any

import pandas as pd

MAIN_COLUMNS=["rank","project","version_id","smell_id","smell_type","affected_component_name","central_component","recommendation_id","display_name","severity","strength","atdi","applicability_probability","predicted_benefit","predicted_risk","ranker_score","llm_validation_score","final_score","confidence","confidence_band","validation_status","concise_reason","warning_count"]


def _records(frame:pd.DataFrame)->list[dict[str,Any]]: return json.loads(frame.to_json(orient="records"))
def validate_final_schema(frame:pd.DataFrame)->None:
    missing=set(MAIN_COLUMNS)-set(frame.columns)
    if missing: raise ValueError(f"Final report missing columns: {sorted(missing)}")


def write_final_reports(raw:pd.DataFrame,diversified:pd.DataFrame,rejected:pd.DataFrame,output_dir:Path,metadata:dict[str,Any],experiment:dict[str,Any]|None=None)->dict[str,Path]:
    output_dir.mkdir(parents=True,exist_ok=True); raw=raw.copy(); raw["concise_reason"]=raw.get("reasoning","").map(lambda value:value[0] if isinstance(value,list) and value else str(value)[:300]); raw["warning_count"]=raw.get("warnings","").map(lambda value:len(value) if isinstance(value,list) else int(bool(value))); validate_final_schema(raw)
    paths={name:output_dir/name for name in ["ranked_recommendations.csv","ranked_recommendations.json","diversified_recommendations.csv","rejected_candidates.csv","smell_instance_ranking.csv","per_smell_candidate_ranking.csv","score_breakdowns.jsonl","recommendation_explanations.md","report.html","experiment_summary.json","run_metadata.json","model_manifest.json"]}
    raw[MAIN_COLUMNS].to_csv(paths["ranked_recommendations.csv"],index=False); paths["ranked_recommendations.json"].write_text(json.dumps(_records(raw),indent=2,default=str)); diversified.to_csv(paths["diversified_recommendations.csv"],index=False); rejected.to_csv(paths["rejected_candidates.csv"],index=False)
    smell=raw.groupby(["project","version_id","smell_id","smell_type"],as_index=False).agg(final_score=("final_score","max"),candidate_count=("candidate_id","count")).sort_values("final_score",ascending=False); smell["rank"]=range(1,len(smell)+1); smell.to_csv(paths["smell_instance_ranking.csv"],index=False)
    per=raw.sort_values(["project","version_id","smell_id","final_score"],ascending=[True,True,True,False]); per["rank_within_smell"]=per.groupby(["project","version_id","smell_id"]).cumcount()+1; per.to_csv(paths["per_smell_candidate_ranking.csv"],index=False)
    with paths["score_breakdowns.jsonl"].open("w") as stream:
        for _,row in raw.iterrows(): stream.write(json.dumps({"candidate_id":row.candidate_id,"ranker":row.get("ranker_score"),"transformer":row.get("transformer_relevance"),"llm":row.get("llm_validation_score"),"rules":row.get("rule_confidence"),"benefit":row.get("predicted_benefit"),"risk":row.get("predicted_risk"),"penalty":row.get("penalty_applied"),"final":row.final_score},default=str)+"\n")
    markdown=["# DSARP final recommendations","","> Recommendations without attached repository evidence are architecture-metric based.",""]
    for _,row in diversified.head(50).iterrows(): markdown.extend([f"## #{row.diversified_rank} {row.display_name}",f"- Component: `{row.affected_component_name}`",f"- Smell: `{row.smell_type}`",f"- Score / confidence: {row.final_score:.3f} / {row.confidence:.3f} ({row.confidence_band})",f"- Validation: {row.validation_status}",f"- Reason: {row.get('concise_reason','Evidence summarized in report.')}",f"- Warnings: {row.get('warnings',[])}",f"- Steps: {row.get('implementation_steps',[])}",""])
    paths["recommendation_explanations.md"].write_text("\n".join(markdown)); payload=experiment or {}; paths["experiment_summary.json"].write_text(json.dumps(payload,indent=2,default=str)); paths["run_metadata.json"].write_text(json.dumps(metadata,indent=2,default=str)); paths["model_manifest.json"].write_text(json.dumps(metadata.get("models",{}),indent=2,default=str))
    sections=["Executive summary","Input data summary","Data-quality report","Selected and excluded smell counts","Top smell instances","Global ranked recommendations","Diversified ranking","Per-smell alternatives","Evidence breakdown","Feature contributions","Transformer predictions","LLM validation","Expected benefits","Risks and warnings","Implementation steps","Model and dataset metadata","Limitations","Reproducibility information"]
    tables=raw.head(50).to_html(index=False,escape=True); nav="".join(f"<li>{html.escape(x)}</li>" for x in sections); paths["report.html"].write_text(f"<!doctype html><meta charset='utf-8'><title>DSARP final report</title><style>body{{font:14px system-ui;max-width:1400px;margin:auto;padding:2rem}}table{{border-collapse:collapse;display:block;overflow:auto}}th,td{{border:1px solid #ccc;padding:.35rem}}</style><h1>DSARP final report</h1><ol>{nav}</ol><p>Source-code context is optional; rows without it are architecture-metric based.</p>{tables}<h2>Reproducibility information</h2><pre>{html.escape(json.dumps(metadata,indent=2,default=str))}</pre>")
    return paths

