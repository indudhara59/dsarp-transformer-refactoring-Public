"""Minimal early stopping callback."""
from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int; best: float = float("-inf"); bad_epochs: int = 0
    def update(self, value: float) -> bool:
        if value > self.best: self.best=value; self.bad_epochs=0
        else: self.bad_epochs+=1
        return self.bad_epochs >= self.patience

