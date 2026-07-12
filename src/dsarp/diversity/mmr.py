"""Exact deduplication, saturation limits, and multi-signal MMR."""
from __future__ import annotations

import ast
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def _elements(value:Any)->set[str]:
    if isinstance(value,list): return set(map(str,value))
    try: parsed=ast.literal_eval(str(value)); return set(map(str,parsed)) if isinstance(parsed,list) else set()
    except (ValueError,SyntaxError): return set()


def candidate_similarity(left:pd.Series,right:pd.Series,config:dict[str,Any])->float:
    score=config["same_component_weight"]*int(left.get("affected_component_id")==right.get("affected_component_id"))+config["same_smell_weight"]*int(left.get("smell_id")==right.get("smell_id"))+config["same_family_weight"]*int(left.get("recommendation_family")==right.get("recommendation_family"))
    a,b=_elements(left.get("affected_elements")),_elements(right.get("affected_elements")); score+=config["affected_overlap_weight"]*(len(a&b)/len(a|b) if a|b else 0)
    va,vb=left.get("embedding"),right.get("embedding")
    if isinstance(va,(list,np.ndarray)) and isinstance(vb,(list,np.ndarray)):
        va,vb=np.asarray(va,float),np.asarray(vb,float); cosine=float(va@vb/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-9)); score+=config["semantic_similarity_weight"]*max(0,cosine)
    return min(1.,score)


def diversify(frame:pd.DataFrame,top_k:int,config:dict[str,Any])->pd.DataFrame:
    ordered=frame.drop_duplicates(["project","version_id","smell_id","affected_component_id","recommendation_id"],keep="first").sort_values(["final_score","candidate_id"],ascending=[False,True],kind="stable"); selected=[]; components=Counter(); smells=Counter(); families=Counter(); lam=float(config["mmr_lambda"])
    while len(selected)<top_k:
        options=[]
        for idx,row in ordered.iterrows():
            if idx in selected or components[str(row.affected_component_id)]>=config["max_per_component"] or smells[str(row.smell_id)]>=config["max_per_smell"]: continue
            if config.get("suppress_family") and families[(str(row.smell_id),str(row.recommendation_family))]>=1: continue
            similarity=max((candidate_similarity(row,ordered.loc[p],config) for p in selected),default=0); options.append((lam*float(row.final_score)-(1-lam)*similarity,str(row.candidate_id),idx))
        if not options: break
        winner=sorted(options,key=lambda x:(-x[0],x[1]))[0][2]; selected.append(winner); row=ordered.loc[winner]; components[str(row.affected_component_id)]+=1; smells[str(row.smell_id)]+=1; families[(str(row.smell_id),str(row.recommendation_family))]+=1
    result=ordered.loc[selected].reset_index(drop=True); result["diversified_rank"]=np.arange(1,len(result)+1); return result

