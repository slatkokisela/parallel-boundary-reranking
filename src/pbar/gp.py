"""Residual GP posterior of BoundaryMORPH (paper Alg. 1, App. B.3).

Augmented index set: pool positions 0..N-1, plus the query at index N.
    K(d,d')  = exp(kappa * (cos(d,d') - 1))                    vMF kernel
    kappa    = log 2 / (1 - median_{i<j} cos(d_i,d_j)), 1 if degenerate
    mu0(d)   = (N-1-pos)/(N-1)                                  MIPS percentile prior
    e        = y - mu0(A);  z = (e - mean e)/std e              residual standardisation
    mu~,s~^2 = zero-mean GP posterior of z (noise 1e-4 on observed block; 1e-2; pinv)
    mu       = mu0 + mean e + std e * mu~;   sigma^2 = std(e)^2 * s~^2
Resolutions of points the paper leaves unspecified (see docs/PLAN.md §6):
    mu0(query) = `mu0_query` (default 1.0); std(e) uses ddof=0 and is replaced by 1.0 when
    |A| < 2 or std < 1e-8 (with only the query observed, z would otherwise be 0/0).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

STD_EPS = 1e-8


def median_kappa(emb: np.ndarray, max_points: int = 2000, seed: int = 0) -> float:
    x = emb
    if len(x) > max_points:  # median over a subsample (exact for N <= max_points)
        x = x[np.random.default_rng(seed).choice(len(x), max_points, replace=False)]
    cos = x @ x.T
    iu = np.triu_indices(len(x), k=1)
    denom = 1.0 - float(np.median(cos[iu])) if len(iu[0]) else 0.0
    return float(np.log(2) / denom) if denom > 1e-8 else 1.0


def vmf_kernel(emb: np.ndarray, kappa: float) -> np.ndarray:
    cos = np.clip(emb @ emb.T, -1.0, 1.0)
    return np.exp(kappa * (cos - 1.0))


def mips_prior(n: int) -> np.ndarray:
    return (n - 1 - np.arange(n)) / (n - 1) if n > 1 else np.ones(1)


class ResidualGP:
    def __init__(self, doc_emb: np.ndarray, q_emb: np.ndarray, mu0_query: float = 1.0,
                 noise: float = 1e-4, use_prior: bool = True, kappa: float | None = None,
                 prior_curve: np.ndarray | None = None):
        """prior_curve: optional calibrated prior by MIPS rank (see calibration.py). Default
        (None) is the paper's percentile prior."""
        self.n = len(doc_emb)
        aug = np.vstack([doc_emb, q_emb[None, :]])
        self.kappa = median_kappa(doc_emb) if kappa is None else kappa
        self.K = vmf_kernel(aug, self.kappa)
        if not use_prior:
            prior = np.full(self.n, 0.5)
        elif prior_curve is not None:
            prior = np.asarray(prior_curve, dtype=np.float64)[: self.n]
            assert len(prior) == self.n, "prior curve shorter than pool"
        else:
            prior = mips_prior(self.n)
        self.mu0 = np.append(prior, mu0_query)
        self.noise = noise
        self.query_index = self.n

    def _solve_factor(self, Kaa: np.ndarray):
        for jit in (self.noise, 1e-2):
            try:
                return ("chol", cho_factor(Kaa + jit * np.eye(len(Kaa)), lower=True))
            except np.linalg.LinAlgError:
                continue
        return ("pinv", np.linalg.pinv(Kaa + 1e-2 * np.eye(len(Kaa))))

    def posterior(self, A: list[int], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and variance over the N pool documents."""
        A = list(A)
        y = np.asarray(y, dtype=np.float64)
        e = y - self.mu0[A]
        m = float(e.mean())
        s = float(e.std()) if len(e) >= 2 else 0.0
        if s < STD_EPS:
            s = 1.0
        z = (e - m) / s
        Kaa = self.K[np.ix_(A, A)]
        Kca = self.K[: self.n, A]                       # (N, |A|)
        kind, F = self._solve_factor(Kaa)
        if kind == "chol":
            alpha = cho_solve(F, z)
            V = solve_triangular(F[0], Kca.T, lower=True)  # (|A|, N)
            var_t = 1.0 - np.einsum("ij,ij->j", V, V)      # vMF diag = 1
        else:
            alpha = F @ z
            var_t = 1.0 - np.einsum("ij,jk,ik->i", Kca, F, Kca)
        mu = self.mu0[: self.n] + m + s * (Kca @ alpha)
        var = (s ** 2) * np.clip(var_t, 0.0, None)
        return mu, var
