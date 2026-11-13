"""Budget-enforcing cross-encoder oracle.

Selectors see CE scores only through `Oracle.score`. Each call is one *round* (one
batched CE inference). Each document scored counts against the budget. Re-scoring an
already scored document costs nothing and returns the stored value. Two backends exist:
* CachedOracle: serves precomputed scores (quality simulation, exact & cheap)
* LiveOracle:   runs the CE model and times every call (latency experiments)
"""
from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np


class BudgetExceeded(RuntimeError):
    pass


class Oracle:
    def __init__(self, budget: int):
        self.budget = budget
        self.observed: dict[int, float] = {}   # pool index -> CE score
        self.rounds = 0
        self.round_sizes: list[int] = []
        self.ce_seconds = 0.0

    @property
    def calls(self) -> int:
        return len(self.observed)

    @property
    def remaining(self) -> int:
        return self.budget - self.calls

    def _compute(self, idx: list[int]) -> np.ndarray:
        raise NotImplementedError

    def score(self, idx: Sequence[int]) -> np.ndarray:
        idx = [int(i) for i in idx]
        new = list(dict.fromkeys(i for i in idx if i not in self.observed))
        if len(new) > self.remaining:
            raise BudgetExceeded(f"requested {len(new)} new scores, {self.remaining} left")
        if new:
            t0 = time.perf_counter()
            vals = self._compute(new)
            self.ce_seconds += time.perf_counter() - t0
            self.observed.update(zip(new, map(float, vals)))
            self.rounds += 1
            self.round_sizes.append(len(new))
        return np.array([self.observed[i] for i in idx], dtype=np.float64)


class CachedOracle(Oracle):
    def __init__(self, budget: int, all_scores: np.ndarray):
        super().__init__(budget)
        self._all = np.asarray(all_scores, dtype=np.float64)

    def _compute(self, idx):
        return self._all[idx]


class LiveOracle(Oracle):
    def __init__(self, budget: int, score_fn: Callable[[list[int]], np.ndarray]):
        super().__init__(budget)
        self._fn = score_fn

    def _compute(self, idx):
        return np.asarray(self._fn(idx))
