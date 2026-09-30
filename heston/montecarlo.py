"""Crude Euler-scheme Monte Carlo Heston pricer, forward-measure form --
used only to cross-check the semi-analytic pricer, not for production use.

Full-truncation Euler (Lord, Koekkoek & Van Dijk, 2010): variance is
floored at zero inside the drift/diffusion terms of both the variance and
log-forward updates, which is the standard, simple fix for Euler
discretization going negative.
"""
from __future__ import annotations

import numpy as np


def heston_mc_price(forward, strike, t, v0, kappa, theta, xi, rho, discount,
                     option_type, n_paths=200_000, n_steps=50, seed=0):
    """Returns (price, standard_error), both already multiplied by discount."""
    rng = np.random.default_rng(seed)
    dt = t / n_steps

    x = np.zeros(n_paths)
    v = np.full(n_paths, v0)

    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        zind = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho * rho) * zind

        v_pos = np.maximum(v, 0.0)
        sqrt_v_dt = np.sqrt(v_pos * dt)

        x += -0.5 * v_pos * dt + sqrt_v_dt * z1
        v += kappa * (theta - v_pos) * dt + xi * sqrt_v_dt * z2

    f_t = forward * np.exp(x)
    if option_type == "CE":
        payoff = np.maximum(f_t - strike, 0.0)
    else:
        payoff = np.maximum(strike - f_t, 0.0)

    price = discount * payoff.mean()
    stderr = discount * payoff.std(ddof=1) / np.sqrt(n_paths)
    return price, stderr
