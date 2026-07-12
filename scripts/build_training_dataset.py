#!/usr/bin/env python3
from pathlib import Path
from dsarp.datasets.dataset_builder import build_dataset
if __name__=="__main__": build_dataset(Path("configs/dataset.yaml"),Path("configs/transformer.yaml"))
