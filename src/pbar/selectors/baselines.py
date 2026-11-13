from __future__ import annotations

import numpy as np

from .base import Selector, scored_then_mips


class MIPSOnly(Selector):
    """First-stage dense ranking. No CE calls."""
    name = "mips"
    uses_budget = False

    def select(self, ctx, oracle, k):
        return np.arange(ctx.n)


class SingleStage(Selector):
    """The paper's SS baseline: CE-score the MIPS top-B, rerank them, append the MIPS tail.

    `batch_size` only changes how the B calls are grouped into rounds (relevant for latency).
    The selected set does not depend on it.
    """
    name = "single_stage"

    def __init__(self, batch_size: int | None = None):
        self.batch_size = batch_size

    def select(self, ctx, oracle, k):
        top = list(range(min(oracle.budget, ctx.n)))
        b = self.batch_size or len(top) or 1
        for s in range(0, len(top), b):
            oracle.score(top[s:s + b])
        return scored_then_mips(ctx.n, oracle.observed)


class FullCE(Selector):
    """Upper bound: CE-score the entire pool (the oracle must have budget >= N)."""
    name = "full_ce"

    def select(self, ctx, oracle, k):
        oracle.score(range(ctx.n))
        return scored_then_mips(ctx.n, oracle.observed)
