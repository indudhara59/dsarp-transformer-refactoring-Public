#!/usr/bin/env python3
"""Run the deterministic Stage 1 baseline."""
from pathlib import Path

from dsarp.pipeline.stage1_pipeline import run_stage1

if __name__ == "__main__":
    run_stage1(Path("data"), top_k=20)

