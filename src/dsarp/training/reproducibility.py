"""Seed and run provenance capture."""
import hashlib
import importlib.metadata
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError: pass


def file_hash(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def run_metadata(configs: list[Path], inputs: list[Path], model_name: str) -> dict[str, Any]:
    try: commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError): commit=None
    packages={name: importlib.metadata.version(name) for name in ["torch","transformers","pandas","numpy","scikit-learn"] if _installed(name)}
    gpu=None
    try:
        import torch
        if torch.cuda.is_available(): gpu=torch.cuda.get_device_name(0)
    except ImportError: pass
    return {"git_commit":commit,"config_hashes":{str(p):file_hash(p) for p in configs},"input_hashes":{str(p):file_hash(p) for p in inputs if p.exists()},"model_name":model_name,"packages":packages,"slurm_job_id":os.getenv("SLURM_JOB_ID"),"gpu":gpu}


def _installed(name: str) -> bool:
    try: importlib.metadata.version(name); return True
    except importlib.metadata.PackageNotFoundError: return False

