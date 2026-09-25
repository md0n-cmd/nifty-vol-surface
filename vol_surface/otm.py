"""Keep only the out-of-the-money leg at each strike, using the parity forward.

A deep in-the-money quote prices almost entirely off intrinsic value, so a
normal bid/ask quoting error maps to a large implied-vol error on Black-76
inversion (this is what flipped NIFTY 2026-09-29's fitted rho sign when its
13 ITM puts were included). OTM legs price mostly off time value and are
the standard convention for building a vol surface -- this drops the
redundant, noisier ITM side at every strike.
"""
from __future__ import annotations

import pandas as pd


def select_otm(cleaned: pd.DataFrame, forwards: pd.DataFrame) -> pd.DataFrame:
    """For each (underlying, expiry, strike), keep the call if strike >=
    forward and the put if strike <= forward. An exact-ATM strike (strike
    == forward) keeps both sides; every other strike keeps exactly one.
    """
    merged = cleaned.merge(
        forwards[["underlying", "expiry", "forward"]],
        on=["underlying", "expiry"], how="inner",
    )
    is_otm_call = (merged["type"] == "CE") & (merged["strike"] >= merged["forward"])
    is_otm_put = (merged["type"] == "PE") & (merged["strike"] <= merged["forward"])
    out = merged[is_otm_call | is_otm_put].drop(columns="forward")
    return out.reset_index(drop=True)
