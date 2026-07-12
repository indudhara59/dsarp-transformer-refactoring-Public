#!/usr/bin/env python3
"""Optuna HPC tuning entry point; no study is run on import."""
import argparse,json
from pathlib import Path
import pandas as pd,yaml
from dsarp.ranking.data import build_rank_data
from dsarp.ranking.features import configured_features
from dsarp.ranking.metrics import evaluate_groups
from dsarp.ranking.trainer import predict_ranker,train_xgboost
def main():
    p=argparse.ArgumentParser(); p.add_argument("--trial-offset",type=int,default=0); p.add_argument("--trials",type=int); a=p.parse_args(); tuning=yaml.safe_load(Path("configs/ranking/tuning.yaml").read_text()); base=yaml.safe_load(Path("configs/ranking/ranker.yaml").read_text()); import optuna; study=optuna.create_study(study_name=tuning["study_name"],storage=tuning["storage"],direction=tuning["direction"],load_if_exists=True); frame=pd.read_csv("outputs/rank_datasets/candidate_rank_dataset.csv"); groups=base["group_levels"]["candidate"]; features=configured_features(base)
    def objective(trial):
        config={**base,"parameters":base["parameters"].copy()}; space=tuning["search_space"]; config["parameters"].update({"max_depth":trial.suggest_int("max_depth",*space["max_depth"]),"min_child_weight":trial.suggest_int("min_child_weight",*space["min_child_weight"]),"learning_rate":trial.suggest_float("learning_rate",*space["learning_rate"],log=True),"subsample":trial.suggest_float("subsample",*space["subsample"]),"colsample_bytree":trial.suggest_float("colsample_bytree",*space["colsample_bytree"]),"reg_alpha":trial.suggest_float("reg_alpha",*space["reg_alpha"],log=True),"reg_lambda":trial.suggest_float("reg_lambda",*space["reg_lambda"],log=True),"gamma":trial.suggest_float("gamma",*space["gamma"]),"n_estimators":trial.suggest_int("n_estimators",*space["n_estimators"])}); train=build_rank_data(frame[frame.rank_split=="train"],features,groups); validation=build_rank_data(frame[frame.rank_split=="validation"],features,groups); path=Path(f"models/ranker/trials/{trial.number}.json"); model=train_xgboost(train,validation,config,path); scored=validation.frame.copy(); scored["ranker_score"]=predict_ranker(model,validation); metrics=evaluate_groups(scored,groups); trial.set_user_attr("ndcg@5",metrics["ndcg@5"]); trial.set_user_attr("map@10",metrics["map@10"]); return metrics["ndcg@10"]
    study.optimize(objective,n_trials=a.trials or tuning["trials"]); output=Path("outputs/evaluations"); output.mkdir(parents=True,exist_ok=True); (output/"best_ranker_parameters.json").write_text(json.dumps(study.best_params,indent=2)); (output/"ranker_trial_summaries.json").write_text(json.dumps([{"number":t.number,"value":t.value,"params":t.params,"user_attrs":t.user_attrs} for t in study.trials],indent=2))
if __name__=="__main__": main()
