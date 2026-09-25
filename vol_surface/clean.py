"""Drop illiquid / stale option legs before surface fitting."""
from __future__ import annotations

import pandas as pd

REQUIRED_COLS = ("bid", "ask", "oi")


def _latest_snapshot_per_series(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent ts within each (underlying, expiry) group.

    A date partition can hold several snapshot times per expiry; mixing them
    would fit a smile against quotes taken minutes apart.
    """
    latest_ts = df.groupby(["underlying", "expiry"])["ts"].transform("max")
    return df[df["ts"] == latest_ts].copy()


def clean_chain(df: pd.DataFrame, max_spread_pct: float = 0.15,
                 min_mid: float = 0.5, min_oi: int = 0) -> pd.DataFrame:
    """Return legs with a valid two-sided quote, live OI, a tight relative
    spread, a non-trivial premium, and enough open interest.

    Drops legs with a missing bid/ask/OI, a non-positive or crossed quote, a
    (ask-bid)/mid spread wider than max_spread_pct, a mid price below min_mid
    (guards against deep-wing quotes where a tiny absolute spread still
    clears the relative-spread filter), or OI below min_oi (a thin-book
    signal that price/spread alone can miss). Adds a `mid` column.
    """
    out = _latest_snapshot_per_series(df)

    for col in REQUIRED_COLS:
        out = out[out[col].notna()]

    out = out[(out["bid"] > 0) & (out["ask"] > 0) & (out["oi"] > 0) & (out["ask"] >= out["bid"])]
    out = out.copy()
    out["mid"] = (out["bid"] + out["ask"]) / 2.0

    spread_pct = (out["ask"] - out["bid"]) / out["mid"]
    out = out[(spread_pct <= max_spread_pct) & (out["mid"] >= min_mid) & (out["oi"] >= min_oi)]

    return out.reset_index(drop=True)
