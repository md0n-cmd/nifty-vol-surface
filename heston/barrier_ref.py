"""Closed-form reference for a down-and-out call under zero-drift Black-76
(forward F a martingale, constant vol sigma), for checking the Monte Carlo barrier
price in the Black-Scholes limit.

Continuous monitoring, H <= K, undiscounted:
    C_do = C(F, K) - C_di
    C_di = H*N(y) - K*(F/H)*N(y - sigma*sqrt(T)),  y = ln(H^2/(F*K))/(sigma*sqrt(T)) + sigma*sqrt(T)/2
(the r = q = 0 case of the standard reflection-principle formula). Discrete
monitoring on m equally spaced dates is approximated by shifting the barrier away
from the spot by exp(-beta*sigma*sqrt(T/m)), beta = 0.5826 (Broadie-Glasserman-Kou,
1997).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

BGK_BETA = 0.5826


def bs_call(forward, strike, t, sigma, discount):
    s = sigma * np.sqrt(t)
    d1 = (np.log(forward / strike) + 0.5 * s * s) / s
    return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d1 - s))


def down_out_call_continuous(forward, strike, barrier, t, sigma, discount):
    """Down-and-out call, continuous monitoring, barrier <= strike."""
    if barrier > strike:
        raise ValueError("formula needs barrier <= strike")
    s = sigma * np.sqrt(t)
    y = np.log(barrier ** 2 / (forward * strike)) / s + 0.5 * s
    c_di = barrier * norm.cdf(y) - strike * (forward / barrier) * norm.cdf(y - s)
    return bs_call(forward, strike, t, sigma, discount) - discount * c_di


def bgk_barrier(barrier, sigma, t, n_steps, beta=BGK_BETA):
    """Continuous-equivalent barrier for discrete monitoring (down barrier)."""
    return barrier * np.exp(-beta * sigma * np.sqrt(t / n_steps))


def down_out_call_discrete_bgk(forward, strike, barrier, t, sigma, discount, n_steps):
    return down_out_call_continuous(forward, strike, bgk_barrier(barrier, sigma, t, n_steps),
                                    t, sigma, discount)
