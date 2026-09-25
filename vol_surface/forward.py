"""Derive the forward price and discount factor per expiry via put-call parity.

For strikes quoted on both sides, C - P = D*(F - K) = D*F - D*K. Regressing
(C - P) mid-price differences against K by OLS recovers the discount factor
D (= -slope) and forward F (= intercept / D) without needing an external
risk-free rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PAIRED_STRIKES = 3
CLOSE_HOUR, CLOSE_MINUTE = 15, 30


def time_to_expiry(ts: pd.Timestamp, expiry: str) -> float:
    """Year fraction (ACT/365) from the snapshot time to expiry close (15:30 IST)."""
    exp_close = pd.Timestamp(f"{expiry} {CLOSE_HOUR:02d}:{CLOSE_MINUTE:02d}", tz=ts.tz)
    seconds = max((exp_close - ts).total_seconds(), 1.0)
    return seconds / (365.0 * 86400.0)


def parity_forward(group: pd.DataFrame) -> dict:
    """Fit the parity regression for one (underlying, expiry) slice of cleaned quotes.

    Raises ValueError if fewer than MIN_PAIRED_STRIKES strikes have both a
    call and a put quote, or if the fit implies a non-positive discount factor.
    """
    calls = group.loc[group["type"] == "CE"].set_index("strike")["mid"]
    puts = group.loc[group["type"] == "PE"].set_index("strike")["mid"]
    strikes = calls.index.intersection(puts.index)
    if len(strikes) < MIN_PAIRED_STRIKES:
        raise ValueError("need >=3 strikes with both call and put quotes for a parity fit")

    K = strikes.to_numpy(dtype=float)
    diff = (calls.loc[strikes] - puts.loc[strikes]).to_numpy(dtype=float)

    slope, intercept = np.polyfit(K, diff, 1)
    discount = -slope
    if discount <= 0:
        raise ValueError("parity fit implied a non-positive discount factor")
    forward = intercept / discount

    ts = group["ts"].iloc[0]
    expiry = group["expiry"].iloc[0]
    return {
        "underlying": group["underlying"].iloc[0],
        "expiry": expiry,
        "forward": forward,
        "discount": discount,
        "t": time_to_expiry(ts, expiry),
        "n_strikes": len(strikes),
    }


def forwards_by_expiry(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Fit parity_forward for every (underlying, expiry) group in a cleaned chain."""
    rows = []
    for _, grp in cleaned.groupby(["underlying", "expiry"]):
        try:
            rows.append(parity_forward(grp))
        except ValueError:
            continue
    if not rows:
        raise ValueError("could not fit a parity forward for any expiry")
    return pd.DataFrame(rows)
