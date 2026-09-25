"""Black-76 pricing and implied-vol inversion on the forward measure.

Prices are discounted by the parity-implied discount factor D, so this is
just Black-76 with an explicit D rather than exp(-r*T):
    C = D * (F*N(d1) - K*N(d2))
    P = D * (K*N(-d2) - F*N(-d1))
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

LO_VOL, HI_VOL = 1e-4, 5.0


def black76_price(forward: float, strike: float, t: float, sigma: float,
                   discount: float, option_type: str) -> float:
    is_call = option_type == "CE"
    if sigma <= 0 or t <= 0:
        intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
        return discount * intrinsic

    vol_sqrt_t = sigma * np.sqrt(t)
    d1 = (np.log(forward / strike) + 0.5 * sigma ** 2 * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if is_call:
        return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def implied_vol(price: float, forward: float, strike: float, t: float,
                 discount: float, option_type: str,
                 lo: float = LO_VOL, hi: float = HI_VOL) -> float:
    """Invert black76_price for sigma via Brent's method. Returns NaN if unsolvable
    (price below intrinsic, or no sign change over [lo, hi])."""
    intrinsic = black76_price(forward, strike, t, 0.0, discount, option_type)
    if price <= intrinsic:
        return np.nan

    def resid(sigma):
        return black76_price(forward, strike, t, sigma, discount, option_type) - price

    try:
        if resid(lo) > 0 or resid(hi) < 0:
            return np.nan
        return brentq(resid, lo, hi, xtol=1e-8)
    except ValueError:
        return np.nan


def attach_iv_surface(cleaned: pd.DataFrame, forwards: pd.DataFrame) -> pd.DataFrame:
    """Merge cleaned quotes with their expiry's forward/discount/t, then add
    iv, log-moneyness k = ln(K/F), and total variance w = iv^2 * t."""
    merged = cleaned.merge(forwards, on=["underlying", "expiry"], how="inner")
    merged["iv"] = merged.apply(
        lambda r: implied_vol(r["mid"], r["forward"], r["strike"], r["t"], r["discount"], r["type"]),
        axis=1,
    )
    merged = merged[merged["iv"].notna() & (merged["iv"] > 0)].copy()
    merged["k"] = np.log(merged["strike"] / merged["forward"])
    merged["w"] = merged["iv"] ** 2 * merged["t"]
    return merged.reset_index(drop=True)
