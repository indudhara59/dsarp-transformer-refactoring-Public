#!/usr/bin/env python3
from pathlib import Path
from dsarp.training.hyperparameters import generate_manifests
if __name__=="__main__": generate_manifests(Path("outputs/evaluations/experiment_manifests"))
