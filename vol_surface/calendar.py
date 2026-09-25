"""Calendar (cross-expiry) no-arbitrage check for a set of per-expiry SVI fits.

Static arbitrage requires total variance to be non-decreasing in time at
fixed log-moneyness: w(k, T1) <= w(k, T2) for T1 < T2 (Gatheral & Jacquier,
2013). This checks that condition on a shared k-grid per underlying, using
only the overlap of each pair of slices' in-sample domains (+/-3 sigma
around each slice's fitted center) -- it does not extrapolate past where a
slice was actually fit.
"""
from __future__ import annotations

import numpy as np

K_POINTS = 41
TOL = 1e-8


def check_calendar_arbitrage(fits: dict, t_by_key: dict,
                              k_points: int = K_POINTS, tol: float = TOL) -> list[dict]:
    """fits: {(underlying, expiry): SVIFit}. t_by_key: {(underlying, expiry): t}.

    Returns one violation record per (underlying, k) where total variance
    decreases going from the nearer to the farther expiry in an adjacent
    pair (sorted by t). An empty list means the check passes everywhere it
    was evaluated.
    """
    violations = []
    by_underlying: dict = {}
    for (underlying, expiry), fit in fits.items():
        by_underlying.setdefault(underlying, []).append((t_by_key[(underlying, expiry)], expiry, fit))

    for underlying, slices in by_underlying.items():
        slices.sort(key=lambda row: row[0])
        for (t1, exp1, fit1), (t2, exp2, fit2) in zip(slices, slices[1:]):
            lo = max(fit1.m - 3 * fit1.sigma, fit2.m - 3 * fit2.sigma)
            hi = min(fit1.m + 3 * fit1.sigma, fit2.m + 3 * fit2.sigma)
            if lo >= hi:
                continue
            k_grid = np.linspace(lo, hi, k_points)
            w_near, w_far = fit1.w(k_grid), fit2.w(k_grid)
            bad = w_far < w_near - tol
            for k, w1, w2 in zip(k_grid[bad], w_near[bad], w_far[bad]):
                violations.append({
                    "underlying": underlying, "k": float(k),
                    "expiry_near": exp1, "w_near": float(w1),
                    "expiry_far": exp2, "w_far": float(w2),
                })
    return violations
