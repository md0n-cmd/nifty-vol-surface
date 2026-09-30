"""Calibrate Heston (v0, kappa, theta, xi, rho) to a cleaned OTM market
slice via weighted least squares in implied-vol space.

Weighting is inverse relative bid-ask spread, not vega -- ties the fit
weight to the same liquidity signal already used to clean the chain,
rather than requiring a separate Black-76 vega computation.

Bounds keep v0, kappa, theta, xi > 0 and |rho| < 1; the Feller condition
(2*kappa*theta > xi^2) is not enforced, only checked at the optimum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from vol_surface.implied_vol import implied_vol

from .pricing import heston_price_batch

PARAM_NAMES = ("v0", "kappa", "theta", "xi", "rho")
MIN_POINTS = 5

#            v0     kappa   theta   xi     rho
LOWER_BOUNDS = [1e-4,  0.1,   1e-4,  0.01, -0.999]
UPPER_BOUNDS = [1.0,   10.0,  1.0,   2.5,   0.999]


@dataclass
class HestonFit:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    n_points: int
    cost: float

    @property
    def feller_violated(self) -> bool:
        return 2 * self.kappa * self.theta < self.xi ** 2

    def price(self, forward, strikes, t, discount, option_types):
        return heston_price_batch(forward, strikes, t, self.v0, self.kappa,
                                   self.theta, self.xi, self.rho, discount, option_types)

    def implied_vol_curve(self, forward, strikes, t, discount, option_types):
        prices = self.price(forward, strikes, t, discount, option_types)
        return np.array([
            implied_vol(p, forward, k, t, discount, ty)
            for p, k, ty in zip(prices, strikes, option_types)
        ])


def _residuals(params, strikes, types, forward, t, discount, market_iv, weights):
    v0, kappa, theta, xi, rho = params
    prices = heston_price_batch(forward, strikes, t, v0, kappa, theta, xi, rho, discount, types)
    model_iv = np.array([
        implied_vol(p, forward, k, t, discount, ty)
        for p, k, ty in zip(prices, strikes, types)
    ])
    resid = np.where(np.isnan(model_iv), 0.0, model_iv - market_iv)
    return weights * resid


def _slice_inputs(otm_slice_df, forward, discount, t):
    """Extract the (strikes, types, forward, t, discount, market_iv, weights)
    tuple _residuals/_joint_residuals need from one cleaned OTM slice."""
    clean = otm_slice_df[otm_slice_df["iv"].notna()]
    if len(clean) < MIN_POINTS:
        raise ValueError(f"need >={MIN_POINTS} quotes to calibrate Heston, got {len(clean)}")

    strikes = clean["strike"].to_numpy(dtype=float)
    types = clean["type"].to_numpy()
    market_iv = clean["iv"].to_numpy(dtype=float)

    spread_pct = (clean["ask"] - clean["bid"]) / clean["mid"]
    weights = 1.0 / np.maximum(spread_pct.to_numpy(dtype=float), 1e-4)
    weights = weights / weights.mean()

    return strikes, types, forward, t, discount, market_iv, weights


def calibrate_heston(otm_slice_df, forward: float, discount: float, t: float,
                      x0=None) -> HestonFit:
    """otm_slice_df needs strike, type, iv, bid, ask, mid columns for a
    single (underlying, expiry) OTM-only slice (e.g. vol_surface.otm's
    output, already merged with implied vol)."""
    inputs = _slice_inputs(otm_slice_df, forward, discount, t)
    market_iv = inputs[5]

    if x0 is None:
        v_atm = float(np.median(market_iv) ** 2)
        x0 = [v_atm, 2.0, v_atm, 0.5, -0.3]

    result = least_squares(
        _residuals, x0, args=inputs, bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        method="trf", x_scale="jac", max_nfev=300,
    )
    v0, kappa, theta, xi, rho = result.x
    return HestonFit(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
                      n_points=len(inputs[0]), cost=float(result.cost))


def _joint_residuals(params, slices):
    return np.concatenate([_residuals(params, *inputs) for inputs in slices])


def calibrate_heston_joint(otm_slices: dict, forwards_df, x0=None) -> HestonFit:
    """Fit one shared (v0, kappa, theta, xi, rho) against several expiries at
    once, minimizing the concatenation of every slice's weighted IV
    residuals in a single least-squares objective.

    otm_slices: {(underlying, expiry): otm_slice_df}. forwards_df: the
    forward.forwards_by_expiry() table (needs underlying, expiry, forward,
    discount, t columns) to look up each slice's own forward/t/discount.
    """
    slices = []
    for (underlying, expiry), df in otm_slices.items():
        row = forwards_df[(forwards_df["underlying"] == underlying)
                           & (forwards_df["expiry"] == expiry)].iloc[0]
        slices.append(_slice_inputs(df, row["forward"], row["discount"], row["t"]))

    if x0 is None:
        all_iv = np.concatenate([s[5] for s in slices])
        v_atm = float(np.median(all_iv) ** 2)
        x0 = [v_atm, 2.0, v_atm, 0.5, -0.3]

    result = least_squares(
        _joint_residuals, x0, args=(slices,), bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        method="trf", x_scale="jac", max_nfev=300,
    )
    v0, kappa, theta, xi, rho = result.x
    total_n = sum(len(s[0]) for s in slices)
    return HestonFit(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
                      n_points=total_n, cost=float(result.cost))
