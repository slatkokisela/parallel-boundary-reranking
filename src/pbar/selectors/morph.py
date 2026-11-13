"""Sequential BoundaryMORPH (paper Alg. 2), structured so batch variants can subclass it.

Per round:
  eff      = CE score for scored docs, posterior mean mu otherwise (the paper's final-ranking
             rule, applied consistently to the incumbent set as well)
  S_t      = top-k of the pool by eff (ties -> MIPS position)
  (j*, i*) = argmin_{j in S_t, i notin S_t} (eff_j - eff_i) / (sigma_j + sigma_i)       (Eq. 4)
  query whichever of j*, i* has the larger sigma
Choices the paper leaves open (documented, configurable where it matters):
  * pairs with both members already scored are excluded (nothing to buy), and the
    already-scored member of a pair is never re-queried;
  * the denominator is floored at 1e-12.
Subclasses override `acquire(...)` to return up to `b` pool positions per round.
"""
from __future__ import annotations

import time

import numpy as np

from ..gp import ResidualGP
from .base import Selector

DENOM_EPS = 1e-12


def effective_scores(mu: np.ndarray, observed: dict[int, float]) -> np.ndarray:
    eff = mu.copy()
    for i, v in observed.items():
        eff[i] = v
    return eff


def topk_mask(eff: np.ndarray, k: int) -> np.ndarray:
    order = np.lexsort((np.arange(len(eff)), -eff))   # eff desc, then MIPS position asc
    mask = np.zeros(len(eff), dtype=bool)
    mask[order[:k]] = True
    return mask


def boundary_gaps(eff, sigma, k, scored: np.ndarray):
    """Standardised gap matrix G[j, i] for incumbents j and challengers i, with index arrays.
    Pairs where both members are scored get +inf."""
    inc_mask = topk_mask(eff, k)
    inc, ch = np.flatnonzero(inc_mask), np.flatnonzero(~inc_mask)
    G = (eff[inc][:, None] - eff[ch][None, :]) / np.maximum(sigma[inc][:, None] + sigma[ch][None, :], DENOM_EPS)
    G[np.ix_(scored[inc], scored[ch])] = np.inf
    return G, inc, ch


class BoundaryMORPH(Selector):
    name = "morph"

    def __init__(self, batch_size: int = 1, mu0_query: float = 1.0, noise: float = 1e-4,
                 use_prior: bool = True, acquisition: str = "boundary", seed: int = 0,
                 prior: str = "percentile"):
        if prior not in ("percentile", "calibrated"):
            raise ValueError(f"unknown prior {prior!r}")
        self.prior = prior                      # "percentile" (paper) | "calibrated" (extension)
        self.b = batch_size
        self.mu0_query, self.noise, self.use_prior = mu0_query, noise, use_prior
        self.acquisition = acquisition          # "boundary" (paper) | "max_sigma" (ablation)
        self.seed = seed
        self.gp_seconds = 0.0                   # algorithm overhead (excludes CE time)

    # ---- acquisition -----------------------------------------------------------------
    def pick_one(self, eff, sigma, k, scored) -> int | None:
        if scored.all():
            return None
        if self.acquisition == "max_sigma":
            s = np.where(scored, -np.inf, sigma)
            return int(np.argmax(s))
        if not 0 < k < len(eff):                # boundary undefined
            return int(np.argmax(np.where(scored, -np.inf, sigma)))
        G, inc, ch = boundary_gaps(eff, sigma, k, scored)
        r, c = np.unravel_index(np.argmin(G), G.shape)
        j, i = int(inc[r]), int(ch[c])
        if scored[j]:
            return i
        if scored[i]:
            return j
        return j if sigma[j] >= sigma[i] else i    # larger sigma; ties -> incumbent

    def acquire(self, gp, A, y, mu, sigma, observed, k, b) -> list[int]:
        """Return up to b pool positions to score this round. The sequential version uses b = 1."""
        scored = np.zeros(gp.n, dtype=bool)
        scored[list(observed)] = True
        d = self.pick_one(effective_scores(mu, observed), sigma, k, scored)
        return [] if d is None else [d]

    # ---- main loop -----------------------------------------------------------------
    def select(self, ctx, oracle, k):
        self.gp_seconds = 0.0                   # per-query overhead
        t0 = time.perf_counter()
        curve = None
        if self.prior == "calibrated":
            if ctx.rank_prior is None:
                raise ValueError("prior='calibrated' requires ctx.rank_prior (set by the experiment)")
            curve = ctx.rank_prior
        gp = ResidualGP(ctx.doc_emb, ctx.q_emb, self.mu0_query, self.noise, self.use_prior,
                        prior_curve=curve)
        A, y = [gp.query_index], [1.0]
        mu, var = gp.posterior(A, y)
        while oracle.remaining > 0:
            picks = self.acquire(gp, A, y, mu, np.sqrt(var), oracle.observed, k,
                                 min(self.b, oracle.remaining))
            if not picks:
                break
            self.gp_seconds += time.perf_counter() - t0
            vals = oracle.score(picks)                 # one round = one batched CE call
            t0 = time.perf_counter()
            A += list(picks)
            y += vals.tolist()
            mu, var = gp.posterior(A, y)
        eff = effective_scores(mu, oracle.observed)
        ranking = np.lexsort((np.arange(gp.n), -eff))
        self.gp_seconds += time.perf_counter() - t0
        return ranking
