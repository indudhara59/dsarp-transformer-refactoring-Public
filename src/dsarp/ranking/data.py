"""Leakage-safe learning-to-rank matrices and qids."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RankData:
    frame: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    qid: np.ndarray
    group_sizes: np.ndarray
    feature_names: list[str]
    weights: np.ndarray


def construct_qid(frame: pd.DataFrame, group_columns: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Sort groups contiguously and construct deterministic integer qids/groups."""
    missing=set(group_columns)-set(frame.columns)
    if missing: raise ValueError(f"Missing ranking group columns: {sorted(missing)}")
    ordered=frame.sort_values([*group_columns,"candidate_id"],kind="stable").reset_index(drop=True)
    keys=ordered[group_columns].astype(str).agg("\x1f".join,axis=1); codes,_=pd.factorize(keys,sort=False); sizes=pd.Series(codes).value_counts(sort=False).sort_index().to_numpy()
    return codes.astype(np.int64),sizes.astype(np.int64),ordered


def validate_rank_data(frame: pd.DataFrame,group_columns:list[str],label_column:str)->None:
    """Reject missing IDs/labels, duplicate candidates, and singleton-only datasets."""
    required={*group_columns,"candidate_id",label_column}; missing=required-set(frame.columns)
    if missing: raise ValueError(f"Rank data missing: {sorted(missing)}")
    if frame.candidate_id.duplicated().any(): raise ValueError("Duplicate candidate_id values in rank data")
    labels=pd.to_numeric(frame[label_column],errors="coerce")
    if labels.isna().any() or not labels.between(0,4).all(): raise ValueError("Ranking labels must be complete grades in [0,4]")
    if frame.groupby(group_columns,dropna=False).size().max()<2: raise ValueError("At least one ranking group must contain two items")


def build_rank_data(frame:pd.DataFrame,feature_names:list[str],group_columns:list[str],label_column:str="relevance_grade")->RankData:
    validate_rank_data(frame,group_columns,label_column); qid,sizes,ordered=construct_qid(frame,group_columns)
    available=[name for name in feature_names if name in ordered]; matrix=ordered[available].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    labels=pd.to_numeric(ordered[label_column]).to_numpy(np.float32); weights=pd.to_numeric(ordered.get("label_confidence",1),errors="coerce").fillna(1).clip(0,1).to_numpy(np.float32) if "label_confidence" in ordered else np.ones(len(ordered),np.float32)
    return RankData(ordered,matrix,labels,qid,sizes,available,weights)


def fixed_group_split(frame:pd.DataFrame,group_columns:list[str],seed:int=42,train:float=.7,validation:float=.15)->pd.DataFrame:
    """Hash complete ranking groups into persistent partitions."""
    result=frame.copy(); keys=result[group_columns].astype(str).agg("\x1f".join,axis=1)
    def choose(key:str)->str:
        fraction=int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:12],16)/16**12
        return "train" if fraction<train else "validation" if fraction<train+validation else "test"
    result["rank_split_group"]=keys; result["rank_split"]=keys.map(choose); return result

