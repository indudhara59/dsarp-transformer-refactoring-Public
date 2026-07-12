"""XGBoost LambdaMART with optional pointwise and LightGBM alternatives."""
from pathlib import Path
from typing import Any

import json
import numpy as np

from dsarp.ranking.data import RankData


def train_xgboost(train:RankData,validation:RankData,config:dict[str,Any],output:Path)->Any:
    """Train LambdaMART with qids, confidence weights, early stopping, and manifests."""
    import xgboost as xgb
    parameters=config["parameters"].copy(); rounds=int(parameters.pop("n_estimators")); early=int(parameters.pop("early_stopping_rounds")); parameters.update(objective=config["objective"],eval_metric=config["evaluation_metrics"])
    dtrain=xgb.DMatrix(train.features,label=train.labels,weight=train.weights,qid=train.qid,feature_names=train.feature_names); dvalid=xgb.DMatrix(validation.features,label=validation.labels,weight=validation.weights,qid=validation.qid,feature_names=validation.feature_names)
    model=xgb.train(parameters,dtrain,num_boost_round=rounds,evals=[(dtrain,"train"),(dvalid,"validation")],early_stopping_rounds=early,verbose_eval=False,xgb_model=str(output) if output.exists() else None)
    output.parent.mkdir(parents=True,exist_ok=True); model.save_model(output); output.with_suffix(".features.json").write_text(json.dumps(train.feature_names,indent=2)); return model


def pointwise_fit(train:RankData)->Any:
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=42).fit(train.features,train.labels,sample_weight=train.weights)


def predict_ranker(model:Any,data:RankData)->np.ndarray:
    try:
        import xgboost as xgb
        if isinstance(model,xgb.Booster): return model.predict(xgb.DMatrix(data.features,feature_names=data.feature_names))
    except ImportError: pass
    return np.asarray(model.predict(data.features),float)

