"""Configurable cached Arcan execution with a test double."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from dsarp.constants import INPUT_FILENAMES


def validate_outputs(output: Path) -> dict[str, Path]:
    files = {role: output / name for role, name in INPUT_FILENAMES.items()}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing: raise FileNotFoundError(f"Arcan outputs missing: {missing}")
    return files


class ArcanRunner(Protocol):
    def run(self, source: Path, output: Path, force: bool = False) -> dict[str, Path]: ...


@dataclass
class SubprocessArcanRunner:
    command: list[str]
    timeout: int = 1800
    version: str = "unknown"

    def run(self, source: Path, output: Path, force: bool = False) -> dict[str, Path]:
        if output.exists() and not force:
            try: return validate_outputs(output)
            except FileNotFoundError: pass
        output.mkdir(parents=True, exist_ok=True)
        args = [part.format(source=source, output=output) for part in self.command]
        try: result = subprocess.run(args, text=True, capture_output=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as exc: raise RuntimeError(f"Arcan timed out after {self.timeout}s") from exc
        metadata = {"command": args, "version": self.version, "finished_at": datetime.now(UTC).isoformat(), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        (output / "arcan-run.json").write_text(json.dumps(metadata, indent=2))
        if result.returncode: raise RuntimeError(f"Arcan failed: {result.stderr.strip()}")
        return validate_outputs(output)


@dataclass
class MockArcanRunner:
    fixture_dir: Path
    def run(self, source: Path, output: Path, force: bool = False) -> dict[str, Path]:
        output.mkdir(parents=True, exist_ok=True)
        for name in INPUT_FILENAMES.values(): shutil.copy2(self.fixture_dir / name, output / name)
        return validate_outputs(output)
