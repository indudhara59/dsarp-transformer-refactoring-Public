"""Regression metrics with correlation guards."""
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: Any,y_pred: Any) -> dict[str,float|None]:
    true=np.asarray(y_true,float); pred=np.asarray(y_pred,float); mask=np.isfinite(true)&np.isfinite(pred); true,pred=true[mask],pred[mask]
    if not len(true): return {"mae":None,"rmse":None,"spearman":None,"pearson":None,"r2":None}
    return {"mae":float(mean_absolute_error(true,pred)),"rmse":float(mean_squared_error(true,pred)**.5),"spearman":float(spearmanr(true,pred).statistic) if len(true)>1 else None,"pearson":float(pearsonr(true,pred).statistic) if len(true)>1 else None,"r2":float(r2_score(true,pred)) if len(true)>1 else None}

