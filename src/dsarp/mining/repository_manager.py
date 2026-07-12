"""Failure-bounded, cross-platform Git repository operations."""
from __future__ import annotations

import contextlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class GitError(RuntimeError):
    """A Git operation failed or timed out."""


@dataclass(frozen=True)
class RepositoryState:
    path: Path
    head: str
    branch: str | None
    dirty: bool


class RepositoryManager:
    def __init__(self, path: Path, timeout: int = 120) -> None:
        self.path, self.timeout = path.resolve(), timeout

    @classmethod
    def clone(cls, url: str, destination: Path, timeout: int = 600) -> "RepositoryManager":
        cls._run_at(destination.parent, ["git", "clone", "--", url, str(destination)], timeout)
        return cls(destination, timeout)

    @staticmethod
    def _run_at(cwd: Path, args: list[str], timeout: int) -> str:
        try:
            result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise GitError(f"Timed out after {timeout}s: {' '.join(args)}") from exc
        if result.returncode:
            raise GitError(result.stderr.strip() or f"Git exited {result.returncode}")
        return result.stdout.strip()

    def run(self, *args: str) -> str:
        return self._run_at(self.path, ["git", *args], self.timeout)

    def state(self) -> RepositoryState:
        head = self.run("rev-parse", "HEAD")
        branch = self.run("branch", "--show-current") or None
        return RepositoryState(self.path, head, branch, bool(self.run("status", "--porcelain")))

    def resolve(self, revision: str) -> str:
        return self.run("rev-parse", "--verify", f"{revision}^{{commit}}")

    def parents(self, commit: str) -> list[str]:
        parts = self.run("show", "-s", "--format=%P", commit).split()
        return parts

    def first_parent(self, commit: str) -> str | None:
        values = self.parents(commit)
        return values[0] if values else None

    def fetch(self) -> None:
        self.run("fetch", "--all", "--tags", "--prune")

    @contextlib.contextmanager
    def worktree(self, commit: str, destination: Path) -> Iterator[Path]:
        resolved = self.resolve(commit)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.run("worktree", "add", "--detach", str(destination), resolved)
        try:
            yield destination
        finally:
            self.run("worktree", "remove", "--force", str(destination))
            self.run("worktree", "prune")
