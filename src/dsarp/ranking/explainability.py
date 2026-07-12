"""Built-in ranker importance with optional permutation/SHAP."""
from typing import Any

import pandas as pd


def xgboost_importance(model:Any,importance_types:tuple[str,...]=( "gain","cover","weight"))->dict[str,dict[str,float]]:
    return {kind:{k:float(v) for k,v in model.get_score(importance_type=kind).items()} for kind in importance_types}


def shap_values_optional(model:Any,features:pd.DataFrame)->Any|None:
    try:
        import shap
        return shap.TreeExplainer(model).shap_values(features)
    except ImportError: return None

