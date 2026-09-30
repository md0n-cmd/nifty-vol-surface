"""Heston characteristic function, forward-measure form.

X_T = ln(F_T / F_0) under the T-forward measure:
    dX_t = -0.5*v_t dt + sqrt(v_t) dW1_t
    dv_t = kappa*(theta - v_t) dt + xi*sqrt(v_t) dW2_t
    dW1 dW2 = rho dt

F_t is a driftless martingale by construction (no separate r/q term needed --
the forward already prices in carry, same as forward.py's parity fit), so
E[F_T] = F_0 exactly, i.e. phi(-1j) == 1 for any valid parameter set. That
identity is a cheap, strong sanity check on the formula below.

Uses the "Little Heston Trap" formulation (Albrecher, Mayer, Schoutens &
Tistaert, 2007): g here is the reciprocal of the g in Heston's original 1993
paper, which avoids branch-cut discontinuities in the complex logarithm for
longer maturities / certain parameter regimes.
"""
from __future__ import annotations

import numpy as np


def heston_char_fn(u, t, v0, kappa, theta, xi, rho):
    """E[exp(i*u*X_T)] for X_T = ln(F_T/F_0). u may be a real or complex
    numpy array; returns a complex numpy array of the same shape."""
    u = np.asarray(u, dtype=complex)
    xi2 = xi * xi

    b = kappa - rho * xi * 1j * u
    d = np.sqrt(b * b + xi2 * (1j * u + u * u))
    g = (b - d) / (b + d)

    exp_dt = np.exp(-d * t)
    log_term = np.log((1 - g * exp_dt) / (1 - g))

    C = (kappa * theta / xi2) * ((b - d) * t - 2 * log_term)
    D = ((b - d) / xi2) * ((1 - exp_dt) / (1 - g * exp_dt))

    return np.exp(C + D * v0)
