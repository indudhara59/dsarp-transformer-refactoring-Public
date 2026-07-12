#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from dsarp.evaluation.reports import write_evaluation_report
def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--predictions",type=Path,required=True); p.add_argument("--output-dir",type=Path,default=Path("outputs/evaluations")); a=p.parse_args(); payload=json.loads(a.predictions.read_text()) if a.predictions.suffix==".json" else {"prediction_file":str(a.predictions)}; write_evaluation_report(payload,a.output_dir)
if __name__=="__main__": main()
