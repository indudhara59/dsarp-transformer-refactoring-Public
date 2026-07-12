"""Failure-bounded RefactoringMiner batch driver."""
import argparse, json
from pathlib import Path
import pandas as pd, yaml
from dsarp.mining.refactoring_miner import SubprocessRefactoringMiner

parser=argparse.ArgumentParser(); parser.add_argument("--commits",type=Path,required=True); parser.add_argument("--repository",type=Path,required=True); parser.add_argument("--config",type=Path,default=Path("configs/refactoring_miner.yaml")); parser.add_argument("--output-dir",type=Path,default=Path("data/historical")); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--force",action="store_true")
args=parser.parse_args(); cfg=yaml.safe_load(args.config.read_text()); miner=SubprocessRefactoringMiner(cfg["command"],cfg["timeout_seconds"],cfg["retries"]); failures=[]
for row in pd.read_csv(args.commits).to_dict("records"):
    target=args.output_dir/str(row["repository"])/str(row["commit"])/"refactoringminer.json"
    if args.dry_run: continue
    try: miner.mine(args.repository,str(row["commit"]),target,args.force)
    except Exception as exc: failures.append({"commit":row["commit"],"error":str(exc)})
(args.output_dir/"refactoringminer_failures.json").write_text(json.dumps(failures,indent=2))
