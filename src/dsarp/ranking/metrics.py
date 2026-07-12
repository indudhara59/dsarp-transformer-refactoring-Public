"""Ranking metrics and group-level bootstrap confidence intervals."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _ordered(labels:np.ndarray,scores:np.ndarray)->np.ndarray: return labels[np.argsort(-scores,kind="stable")]
def ndcg(labels:Any,scores:Any,k:int)->float:
    values=_ordered(np.asarray(labels,float),np.asarray(scores,float))[:k]; ideal=np.sort(np.asarray(labels,float))[::-1][:k]
    discount=1/np.log2(np.arange(2,len(values)+2)); dcg=float(np.sum((2**values-1)*discount)); idcg=float(np.sum((2**ideal-1)*discount)); return dcg/idcg if idcg else 0.
def average_precision(labels:Any,scores:Any,k:int,threshold:float=2)->float:
    relevant=_ordered(np.asarray(labels,float),np.asarray(scores,float))[:k]>=threshold; total=min(int((np.asarray(labels)>=threshold).sum()),k)
    return float(sum(relevant[i]*relevant[:i+1].mean() for i in range(len(relevant)))/total) if total else 0.
def reciprocal_rank(labels:Any,scores:Any,threshold:float=2)->float:
    hits=np.flatnonzero(_ordered(np.asarray(labels),np.asarray(scores))>=threshold); return 1/float(hits[0]+1) if len(hits) else 0.
def pairwise_accuracy(labels:Any,scores:Any)->float:
    labels=np.asarray(labels); scores=np.asarray(scores); comparable=correct=0
    for i in range(len(labels)):
        for j in range(i+1,len(labels)):
            if labels[i]==labels[j]: continue
            comparable+=1; correct+=int((scores[i]-scores[j])*(labels[i]-labels[j])>0)
    return correct/comparable if comparable else 0.


def group_metrics(labels:Any,scores:Any)->dict[str,float]:
    labels=np.asarray(labels); scores=np.asarray(scores); ordered=_ordered(labels,scores); relevant=labels>=2; total=int(relevant.sum())
    result={f"ndcg@{k}":ndcg(labels,scores,k) for k in [1,3,5,10]}; result.update({f"map@{k}":average_precision(labels,scores,k) for k in [5,10]}); result["mrr"]=reciprocal_rank(labels,scores)
    for k in [1,3,5]: result[f"precision@{k}"]=float((ordered[:k]>=2).sum()/min(k,len(ordered)))
    for k in [5,10]: result[f"recall@{k}"]=float((ordered[:k]>=2).sum()/total) if total else 0.
    result["hit_rate"]=float((ordered>=2).any()); result["pairwise_accuracy"]=pairwise_accuracy(labels,scores); return result


def evaluate_groups(frame:pd.DataFrame,group_columns:list[str],label:str="relevance_grade",score:str="ranker_score")->dict[str,float]:
    values=[group_metrics(group[label],group[score]) for _,group in frame.groupby(group_columns,dropna=False,sort=True)]
    return {key:float(np.mean([item[key] for item in values])) for key in values[0]} if values else {}


def bootstrap_groups(frame:pd.DataFrame,group_columns:list[str],iterations:int=1000,seed:int=42)->dict[str,dict[str,float]]:
    groups=[group for _,group in frame.groupby(group_columns,sort=True)]; rng=np.random.default_rng(seed); distributions:dict[str,list[float]]={}
    for _ in range(iterations):
        sampled=pd.concat([groups[i] for i in rng.integers(0,len(groups),len(groups))],ignore_index=True); metrics=evaluate_groups(sampled,group_columns)
        for key,value in metrics.items(): distributions.setdefault(key,[]).append(value)
    return {key:{"mean":float(np.mean(values)),"lower_95":float(np.quantile(values,.025)),"upper_95":float(np.quantile(values,.975))} for key,values in distributions.items()}

