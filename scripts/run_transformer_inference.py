#!/usr/bin/env python3
"""Cluster inference entry point; intentionally never downloads weights."""
import argparse
from pathlib import Path
def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--dataset",type=Path,required=True); p.parse_args(); raise SystemExit("Use dsarp.inference.batch_predictor with the project dataloader; weights must exist locally.")
if __name__=="__main__": main()
