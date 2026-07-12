"""Train-partition-only smoothed historical priors."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class HistoricalFrequencyModel:
    alpha: float = 1.0
    counts: dict[tuple[str, str], int] = field(default_factory=dict)
    smell_totals: dict[str, int] = field(default_factory=dict)
    labels: set[str] = field(default_factory=set)
    fitted_partition: str | None = None

    def fit(self, frame: pd.DataFrame, partition: str = "train") -> "HistoricalFrequencyModel":
        if "split" in frame and set(frame["split"].dropna()) - {partition}: raise ValueError("Frequency fitting accepts one training partition only")
        self.fitted_partition = partition
        for row in frame.itertuples():
            key = (str(row.smell_type), str(row.recommendation_id)); self.counts[key] = self.counts.get(key, 0) + int(bool(row.overall_improved)); self.smell_totals[key[0]] = self.smell_totals.get(key[0], 0) + int(bool(row.overall_improved)); self.labels.add(key[1])
        return self

    def probability(self, smell: str, recommendation: str) -> float:
        if self.fitted_partition is None: raise RuntimeError("Frequency model is not fitted")
        return (self.counts.get((smell, recommendation), 0) + self.alpha) / (self.smell_totals.get(smell, 0) + self.alpha * max(1, len(self.labels)))
