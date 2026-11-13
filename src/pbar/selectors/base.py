from __future__ import annotations

import numpy as np

from ..oracle import Oracle
from ..retrieval import QueryContext


class Selector:
    """Selectors return a full ordering of pool positions. The final set is the first k.

    Every selector is called with an oracle whose budget is B. It may spend at most B
    CE calls, in any number of rounds (oracle.score calls).
    """
    name = "base"
    uses_budget = True

    def select(self, ctx: QueryContext, oracle: Oracle, k: int) -> np.ndarray:
        raise NotImplementedError


def scored_then_mips(n: int, observed: dict[int, float]) -> np.ndarray:
    """Scored docs ordered by CE desc (ties -> MIPS position), then unscored docs in MIPS order."""
    scored = sorted(observed, key=lambda i: (-observed[i], i))
    rest = [i for i in range(n) if i not in observed]
    return np.array(scored + rest, dtype=np.int64)
