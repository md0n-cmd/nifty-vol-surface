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


def clean_chain(df: pd.DataFrame, max_spread_pct: float = 0.15) -> pd.DataFrame:
    """Return legs with a valid two-sided quote, live OI, and a tight spread.

    Drops legs with a missing bid/ask/OI, a non-positive or crossed quote, or
    a (ask-bid)/mid spread wider than max_spread_pct. Adds a `mid` column.
    """
    out = _latest_snapshot_per_series(df)

    for col in REQUIRED_COLS:
        out = out[out[col].notna()]

    out = out[(out["bid"] > 0) & (out["ask"] > 0) & (out["oi"] > 0) & (out["ask"] >= out["bid"])]
    out = out.copy()
    out["mid"] = (out["bid"] + out["ask"]) / 2.0

    spread_pct = (out["ask"] - out["bid"]) / out["mid"]
    out = out[spread_pct <= max_spread_pct]

    return out.reset_index(drop=True)
