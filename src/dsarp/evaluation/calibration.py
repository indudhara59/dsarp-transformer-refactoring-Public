"""Temperature, Platt, and isotonic calibration."""
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def calibration_metrics(y_true: Any, probabilities: Any, bins: int=10) -> dict[str,Any]:
    y=np.asarray(y_true,int); p=np.asarray(probabilities,float); edges=np.linspace(0,1,bins+1); ece=0.; reliability=[]
    for low,high in zip(edges[:-1],edges[1:]):
        mask=(p>=low)&(p<(high if high<1 else high+1e-12))
        if mask.any(): accuracy=float(y[mask].mean()); confidence=float(p[mask].mean()); count=int(mask.sum()); ece+=count/len(y)*abs(accuracy-confidence); reliability.append({"lower":float(low),"upper":float(high),"accuracy":accuracy,"confidence":confidence,"count":count})
    return {"ece":float(ece),"brier_score":float(brier_score_loss(y,p)),"reliability":reliability}


class TemperatureScaler:
    def __init__(self,temperature:float=1.): self.temperature=temperature
    def fit(self,logits:Any,labels:Any)->"TemperatureScaler":
        from scipy.optimize import minimize_scalar
        x=np.asarray(logits,float); y=np.asarray(labels,int)
        def objective(t:float)->float:
            p=1/(1+np.exp(-x/t)); return float(-np.mean(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9)))
        self.temperature=float(minimize_scalar(objective,bounds=(.05,10),method="bounded").x); return self
    def predict_proba(self,logits:Any)->np.ndarray: return 1/(1+np.exp(-np.asarray(logits,float)/self.temperature))


def fit_calibrator(method:str,scores:Any,labels:Any)->Any:
    if method=="temperature": return TemperatureScaler().fit(scores,labels)
    if method=="platt": return LogisticRegression().fit(np.asarray(scores).reshape(-1,1),labels)
    if method=="isotonic": return IsotonicRegression(out_of_bounds="clip").fit(scores,labels)
    raise ValueError(f"Unknown calibration method: {method}")

