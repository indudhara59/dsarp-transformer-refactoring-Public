"""Cached subprocess and mock adapters for RefactoringMiner."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class RefactoringMiner(Protocol):
    def mine(self, repository: Path, commit: str, output: Path, force: bool = False) -> dict: ...


@dataclass
class SubprocessRefactoringMiner:
    command: list[str]
    timeout: int = 600
    retries: int = 1

    def mine(self, repository: Path, commit: str, output: Path, force: bool = False) -> dict:
        if output.exists() and not force: return json.loads(output.read_text())
        output.parent.mkdir(parents=True, exist_ok=True)
        args = [part.format(repository=repository, commit=commit, output=output) for part in self.command]
        last = ""
        for _ in range(self.retries + 1):
            try:
                result = subprocess.run(args, text=True, capture_output=True, timeout=self.timeout, check=False)
            except subprocess.TimeoutExpired as exc:
                last = f"timeout after {self.timeout}s"; continue
            if result.returncode == 0 and output.exists(): return json.loads(output.read_text())
            last = result.stderr.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"RefactoringMiner failed: {last}")


@dataclass
class MockRefactoringMiner:
    payload: dict
    def mine(self, repository: Path, commit: str, output: Path, force: bool = False) -> dict:
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(self.payload, indent=2)); return self.payload
