"""Deterministic candidate commit selection; messages are evidence of candidacy only."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime

from dsarp.mining.repository_manager import RepositoryManager

DEFAULT_KEYWORDS = ("refactor", "extract", "move", "rename", "clean", "modular", "architecture", "dependency", "decouple", "split", "interface", "restructure")


@dataclass(frozen=True)
class CommitSelectionConfig:
    keywords: tuple[str, ...] = DEFAULT_KEYWORDS
    since: str | None = None
    until: str | None = None
    max_commits: int | None = None
    sample_size: int | None = None
    seed: int = 42
    all_non_merge: bool = False
    java_only: bool = True
    known_commits: frozenset[str] = field(default_factory=frozenset)


def select_commits(repo: RepositoryManager, config: CommitSelectionConfig) -> list[dict[str, str]]:
    args = ["log", "--format=%H%x1f%P%x1f%aI%x1f%s"]
    if config.since: args.append(f"--since={config.since}")
    if config.until: args.append(f"--until={config.until}")
    if config.max_commits: args.append(f"--max-count={config.max_commits}")
    rows: list[dict[str, str]] = []
    for line in repo.run(*args).splitlines():
        commit, parents, date, subject = line.split("\x1f", 3)
        parent_list = parents.split()
        if not parent_list or len(parent_list) > 1: continue
        lowered = subject.casefold()
        matched = [word for word in config.keywords if word in lowered]
        known = commit in config.known_commits
        java = bool(repo.run("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()) and any(p.endswith(".java") for p in repo.run("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines())
        if config.java_only and not java: continue
        if not (config.all_non_merge or matched or known): continue
        reason = "known_refactoring" if known else ("keyword:" + ",".join(matched) if matched else "all_non_merge")
        rows.append({"commit": commit, "parent_commit": parent_list[0], "date": date, "subject": subject, "selection_strategy": "configured", "selection_reason": reason, "processing_status": "pending"})
    rows.sort(key=lambda row: (datetime.fromisoformat(row["date"]), row["commit"]))
    if config.sample_size is not None and len(rows) > config.sample_size:
        rows = random.Random(config.seed).sample(rows, config.sample_size)
        rows.sort(key=lambda row: row["commit"])
    return rows
