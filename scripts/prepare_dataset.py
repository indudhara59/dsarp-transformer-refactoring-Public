#!/usr/bin/env python3
"""Prepare Stage 1 data artifacts."""
from pathlib import Path

from dsarp.pipeline.stage1_pipeline import run_stage1

if __name__ == "__main__":
    run_stage1(Path("data"))

