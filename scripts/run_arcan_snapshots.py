"""Configured Arcan execution is intentionally explicit and supports dry-run validation."""
import argparse
from pathlib import Path
import yaml
parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=Path("configs/arcan_runner.yaml")); parser.add_argument("--dry-run",action="store_true",required=True)
args=parser.parse_args(); cfg=yaml.safe_load(args.config.read_text()); print({"validated":True,"command":cfg["command"],"dry_run":args.dry_run})
