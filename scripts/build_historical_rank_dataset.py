"""Reuse the existing rank dataset builder for a split-safe historical candidate table."""
import argparse
from pathlib import Path
from dsarp.ranking.dataset_builder import build_rank_dataset
parser=argparse.ArgumentParser(); parser.add_argument("--candidates",type=Path,required=True); parser.add_argument("--predictions",type=Path)
args=parser.parse_args(); build_rank_dataset(args.candidates,args.predictions)
