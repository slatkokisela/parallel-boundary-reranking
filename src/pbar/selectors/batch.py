"""Batch extensions of BoundaryMORPH: pick b documents per round, so a round is one batched CE call.

All variants subclass the sequential `BoundaryMORPH` and override only `acquire`. The
outer loop (in morph.py) then makes ceil(B/b) rounds. Each `acquire` call returns
exactly min(b, #unscored) distinct unscored pool positions.

Variants
--------
morph_topb             naive. Walk boundary pairs in ascending standardised gap (Eq. 4)
                       and take the higher-sigma unscored member of each pair, skipping
                       docs already picked this round. No diversity term: neighbouring
                       near-duplicate docs can all be picked in one batch.
morph_kb               kriging believer / kriging liar. Greedy within a round: pick with
                       the sequential rule, append the pick to (A, y) with a *fantasy*
                       value, recompute the posterior, and repeat. The GP variance does
                       not depend on y, so the sigma shrinkage around earlier picks is
                       exact and supplies diversity through the kernel.
                       fantasy = "mean" (y = mu), "pessimistic" (mu - sigma),
                       "optimistic" (mu + sigma).
                       Caveat: the residuals are z-standardised over A (Alg. 1), so even a
                       "mean" fantasy shifts mean(e) and std(e). The fantasised posterior
                       mean is therefore not exactly the pre-fantasy mean. Fantasies live
                       only inside one acquire call. Real observations are never
                       overwritten, and the outer loop conditions only on real CE scores.
morph_random_boundary  control. Sample b unscored docs uniformly from a band around the
                       top-k cut (ranked by the effective score). Default band: the 2k
                       docs nearest the cut, centred on it, i.e. ranks [max(0, k-k), 2k).
                       The band widens outward if it holds fewer than b unscored docs.
                       RNG: default_rng([seed, crc32(qid)]). This is deterministic per query
                       and independent of grid order.

With b = 1, morph_topb and morph_kb(fantasy="mean") reduce exactly to sequential MORPH.
"""
from __future__ import annotations

import zlib

import numpy as np

from .morph import BoundaryMORPH, boundary_gaps, effective_scores


def _scored_mask(n: int, idx) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[list(idx)] = True
    return m


def _fill_by_sigma(picks: list[int], sigma, scored, b) -> list[int]:
    """Top up `picks` with the highest-sigma unscored, unpicked docs (ties -> MIPS position)."""
    if len(picks) >= b:
        return picks
    taken = scored.copy()
    taken[picks] = True
    order = np.lexsort((np.arange(len(sigma)), -sigma))
    for d in order:
        if len(picks) >= b:
            break
        if not taken[d]:
            picks.append(int(d))
            taken[d] = True
    return picks


class MorphTopB(BoundaryMORPH):
    name = "morph_topb"

    def acquire(self, gp, A, y, mu, sigma, observed, k, b):
        scored = _scored_mask(gp.n, observed)
        if scored.all():
            return []
        eff = effective_scores(mu, observed)
        picks: list[int] = []
        if 0 < k < gp.n:
            G, inc, ch = boundary_gaps(eff, sigma, k, scored)
            flat = G.ravel()
            # stable sort: the first element equals np.argmin(G), giving b = 1 equivalence
            for p in np.argsort(flat, kind="stable"):
                if len(picks) >= b or not np.isfinite(flat[p]):
                    break
                r, c = divmod(int(p), G.shape[1])
                j, i = int(inc[r]), int(ch[c])
                if scored[j]:
                    d = i
                elif scored[i]:
                    d = j
                else:
                    d = j if sigma[j] >= sigma[i] else i
                if d not in picks:
                    picks.append(d)
        return _fill_by_sigma(picks, sigma, scored, b)


class MorphKrigingBeliever(BoundaryMORPH):
    name = "morph_kb"

    def __init__(self, fantasy: str = "mean", **kw):
        super().__init__(**kw)
        if fantasy not in ("mean", "pessimistic", "optimistic"):
            raise ValueError(f"unknown fantasy {fantasy!r}")
        self.fantasy = fantasy

    def acquire(self, gp, A, y, mu, sigma, observed, k, b):
        A_f, y_f = list(A), list(y)
        mu_f, sigma_f = mu, sigma
        pseudo = _scored_mask(gp.n, observed)          # real + fantasised this round
        picks: list[int] = []
        for _ in range(b):
            eff = effective_scores(mu_f, observed)     # real CE scores are never overwritten
            d = self.pick_one(eff, sigma_f, k, pseudo)
            if d is None:
                break
            picks.append(d)
            pseudo[d] = True
            if len(picks) == b:
                break                                  # no posterior needed after the last pick
            val = mu_f[d] + {"mean": 0.0, "pessimistic": -1.0, "optimistic": 1.0}[self.fantasy] * sigma_f[d]
            A_f.append(d)
            y_f.append(float(val))
            mu_f, var_f = gp.posterior(A_f, y_f)
            sigma_f = np.sqrt(var_f)
        return picks


class MorphRandomBoundary(BoundaryMORPH):
    name = "morph_random_boundary"

    def __init__(self, band_width: int | None = None, **kw):
        super().__init__(**kw)
        self.band_width = band_width                   # None -> 2k
        self._rng = np.random.default_rng([self.seed, 0])

    def select(self, ctx, oracle, k):
        self._rng = np.random.default_rng([self.seed, zlib.crc32(str(ctx.qid).encode())])
        return super().select(ctx, oracle, k)

    def acquire(self, gp, A, y, mu, sigma, observed, k, b):
        scored = _scored_mask(gp.n, observed)
        n_unscored = int((~scored).sum())
        if n_unscored == 0:
            return []
        b = min(b, n_unscored)
        eff = effective_scores(mu, observed)
        order = np.lexsort((np.arange(gp.n), -eff))    # rank -> pool position
        half = (self.band_width if self.band_width is not None else 2 * k) // 2
        while True:
            lo, hi = max(0, k - half), min(gp.n, k + half)
            band = [int(d) for d in order[lo:hi] if not scored[d]]
            if len(band) >= b or (lo == 0 and hi == gp.n):
                break
            half = max(1, half * 2)
        return [int(d) for d in self._rng.choice(band, size=b, replace=False)]


BATCH_SELECTORS = {c.name: c for c in (MorphTopB, MorphKrigingBeliever, MorphRandomBoundary)}
