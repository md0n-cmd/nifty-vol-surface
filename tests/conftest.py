"""Synthetic option-chain fixtures matching the collector's parquet schema
(see collector.py: flatten()). Data/ is never committed and isn't synced to
dev machines, so tests build internally-consistent chains from known
forward/discount/SVI inputs rather than reading real snapshots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_surface.implied_vol import black76_price
from vol_surface.svi import SVIFit

TS = pd.Timestamp("2026-09-01 09:20:00", tz="Asia/Kolkata")
EXPIRY = "2026-09-25"
UNDERLYING = "NIFTY"


def make_chain(forward=25000.0, discount=0.9995, svi=None, n_strikes=21,
               strike_step=100.0, spread_frac=0.02, oi=1000, ts=TS, expiry=EXPIRY,
               underlying=UNDERLYING, t=None):
    """Build a clean synthetic CE/PE chain priced off a known forward, discount
    and SVI slice, with tight, uniform spreads and positive OI on every leg."""
    if svi is None:
        svi = SVIFit(a=0.01, b=0.15, rho=-0.3, m=0.0, sigma=0.2, n_points=0, cost=0.0)
    if t is None:
        from vol_surface.forward import time_to_expiry
        t = time_to_expiry(ts, expiry)

    center = round(forward / strike_step) * strike_step
    strikes = center + strike_step * np.arange(-n_strikes // 2, n_strikes // 2 + 1)

    rows = []
    for strike in strikes:
        k = np.log(strike / forward)
        w = max(svi.w(k), 1e-8)
        sigma = np.sqrt(w / t)
        for opt_type in ("CE", "PE"):
            price = black76_price(forward, strike, t, sigma, discount, opt_type)
            half_spread = max(price * spread_frac / 2, 0.01)
            rows.append({
                "ts": ts, "underlying": underlying, "expiry": expiry, "spot": forward * discount,
                "strike": float(strike), "type": opt_type,
                "ltp": price,
                "bid": price - half_spread, "bid_qty": 50,
                "ask": price + half_spread, "ask_qty": 50,
                "oi": oi, "volume": 500,
                "iv": sigma,
                "delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def known_svi():
    return SVIFit(a=0.01, b=0.15, rho=-0.3, m=0.0, sigma=0.2, n_points=0, cost=0.0)


@pytest.fixture
def clean_synthetic_chain(known_svi):
    return make_chain(svi=known_svi)
