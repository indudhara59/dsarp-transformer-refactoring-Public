"""Generate ablation manifests without running experiments."""
import itertools
import json
from pathlib import Path
from typing import Any

from dsarp.models.registry import ABLATIONS


def generate_manifests(output_dir: Path, learning_rates: list[float] | None = None) -> list[Path]:
    output_dir.mkdir(parents=True,exist_ok=True); paths=[]
    for ablation, learning_rate in itertools.product(ABLATIONS,learning_rates or [2e-5]):
        path=output_dir/f"{ablation}_lr{learning_rate:g}.json"; path.write_text(json.dumps({"ablation":ablation,"branches":ABLATIONS[ablation],"learning_rate":learning_rate},indent=2)); paths.append(path)
    return paths

