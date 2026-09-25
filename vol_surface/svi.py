"""Per-expiry SVI (raw parametrization) fit via bounded nonlinear least squares.

    w(k) = a + b*(rho*(k - m) + sqrt((k - m)^2 + sigma^2))

Box bounds enforce b>=0 and |rho|<1 so the slice is convex in the wings.
Two static no-arbitrage conditions from Gatheral & Jacquier (2013) are
enforced as soft penalty residuals, since scipy.optimize.least_squares only
supports box bounds directly:
  1. min_k w(k) = a + b*sigma*sqrt(1-rho^2) >= 0   (no negative total variance)
  2. b*(1+|rho|) <= 4                              (Lee's moment/wing bound)
This does not check full butterfly (density) or calendar-spread arbitrage
across expiries -- only these two single-slice conditions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ("a", "b", "rho", "m", "sigma")
MIN_POINTS = 5
ARB_PENALTY_WEIGHT = 50.0


@dataclass
class SVIFit:
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    n_points: int
    cost: float

    def w(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def iv(self, k, t):
        return np.sqrt(np.clip(self.w(k), 0.0, None) / t)


def _svi_w(params, k):
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def _arb_violations(params):
    """Non-positive when both no-arbitrage conditions hold; positive (penalized)
    when violated."""
    a, b, rho, m, sigma = params
    min_variance = a + b * sigma * np.sqrt(1 - rho ** 2)
    wing_slack = 4.0 - b * (1 + abs(rho))
    return min(0.0, min_variance), min(0.0, wing_slack)


def _residuals(params, k, w, arb_weight):
    resid = _svi_w(params, k) - w
    min_var_violation, wing_violation = _arb_violations(params)
    penalties = arb_weight * np.array([min_var_violation, wing_violation])
    return np.concatenate([resid, penalties])


def fit_svi_slice(k, w, arb_weight: float = ARB_PENALTY_WEIGHT) -> SVIFit:
    """Fit one SVI slice to (k, w) points. Raises ValueError if there are
    fewer than MIN_POINTS quotes."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(k) < MIN_POINTS:
        raise ValueError(f"need >={MIN_POINTS} quotes to fit an SVI slice, got {len(k)}")

    a0 = max(w.min(), 1e-6)
    b0 = 0.1
    rho0 = 0.0
    m0 = k[np.argmin(w)]
    sigma0 = max(float(k.std()), 0.1)
    x0 = [a0, b0, rho0, m0, sigma0]

    lower = [0.0, 0.0, -0.999, k.min() - 1.0, 1e-4]
    upper = [w.max() * 4 + 1e-6, 10.0, 0.999, k.max() + 1.0, 5.0]

    result = least_squares(
        _residuals, x0, args=(k, w, arb_weight), bounds=(lower, upper), method="trf",
    )
    a, b, rho, m, sigma = result.x
    return SVIFit(a=a, b=b, rho=rho, m=m, sigma=sigma, n_points=len(k), cost=float(result.cost))


def fit_all_expiries(surface_df) -> dict:
    """Fit one SVI slice per (underlying, expiry) group. Returns
    {(underlying, expiry): SVIFit}; groups with too few points are skipped."""
    fits = {}
    for key, grp in surface_df.groupby(["underlying", "expiry"]):
        try:
            fits[key] = fit_svi_slice(grp["k"].to_numpy(), grp["w"].to_numpy())
        except ValueError:
            continue
    return fits
