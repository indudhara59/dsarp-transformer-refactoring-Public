#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import joblib,pandas as pd,yaml
from dsarp.evaluation.calibration import calibration_metrics,fit_calibrator
def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--validation",type=Path,required=True); p.add_argument("--config",type=Path,default=Path("configs/calibration.yaml")); a=p.parse_args(); data=pd.read_csv(a.validation); config=yaml.safe_load(a.config.read_text()); best=None
    for method in config["methods"]:
        calibrator=fit_calibrator(method,data.applicability_logit,data.is_applicable); probability=calibrator.predict_proba(data.applicability_logit) if method=="temperature" else (calibrator.predict_proba(data[["applicability_logit"]])[:,1] if method=="platt" else calibrator.predict(data.applicability_logit)); metrics=calibration_metrics(data.is_applicable,probability,config["bins"]); candidate=(metrics[config["selection_metric"]],method,calibrator,metrics); best=candidate if best is None or candidate[0]<best[0] else best
    output=Path(config["output_dir"]); output.mkdir(parents=True,exist_ok=True); joblib.dump(best[2],output/"applicability_calibrator.joblib"); (output/"metrics.json").write_text(json.dumps({"method":best[1],**best[3]},indent=2))
if __name__=="__main__": main()
