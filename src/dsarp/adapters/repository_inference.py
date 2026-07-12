"""Repository/version adapter that terminates in the shared CSV pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path

from dsarp.arcan.runner import ArcanRunner
from dsarp.constants import INPUT_FILENAMES
from dsarp.mining.repository_manager import RepositoryManager
from dsarp.pipeline.csv_inference_pipeline import run_csv_inference


def analyze_repository(repository: Path, version: str, output_dir: Path, arcan: ArcanRunner, **inference_options):
    manager = RepositoryManager(repository); commit = manager.resolve(version)
    with tempfile.TemporaryDirectory(prefix="dsarp-repository-") as temporary:
        with manager.worktree(commit, Path(temporary) / "checkout") as checkout:
            files = arcan.run(checkout, output_dir / "arcan")
    return run_csv_inference(files["smells"], files["affects"], files["metrics"], output_dir, **inference_options)
