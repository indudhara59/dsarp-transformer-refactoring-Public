#!/usr/bin/env python3
"""Inspect configured Arcan exports."""
from pathlib import Path

from dsarp.pipeline.stage1_pipeline import inspect_data

if __name__ == "__main__":
    inspect_data(Path("data"))

