import numpy as np
import pandas as pd

from vol_surface.clean import clean_chain
from tests.conftest import make_chain


def test_clean_chain_keeps_valid_legs(clean_synthetic_chain):
    out = clean_chain(clean_synthetic_chain)
    assert len(out) == len(clean_synthetic_chain)
    assert "mid" in out.columns
    assert (out["mid"] > 0).all()


def test_drops_legs_missing_bid_or_oi(clean_synthetic_chain):
    df = clean_synthetic_chain.copy()
    df.loc[df.index[0], "bid"] = np.nan
    df.loc[df.index[1], "oi"] = 0

    out = clean_chain(df)

    assert len(out) == len(df) - 2


def test_drops_wide_spread_legs(clean_synthetic_chain):
    df = clean_synthetic_chain.copy()
    wide_idx = df.index[5]
    mid = (df.loc[wide_idx, "bid"] + df.loc[wide_idx, "ask"]) / 2
    df.loc[wide_idx, "bid"] = mid * 0.5
    df.loc[wide_idx, "ask"] = mid * 1.5

    out = clean_chain(df, max_spread_pct=0.15)

    kept_strikes_types = set(zip(out["strike"], out["type"]))
    assert (df.loc[wide_idx, "strike"], df.loc[wide_idx, "type"]) not in kept_strikes_types
    assert len(out) == len(df) - 1


def test_keeps_only_latest_snapshot_per_expiry(clean_synthetic_chain):
    stale = clean_synthetic_chain.copy()
    stale["ts"] = stale["ts"] - pd.Timedelta(minutes=30)
    combined = pd.concat([stale, clean_synthetic_chain], ignore_index=True)

    out = clean_chain(combined)

    assert len(out) == len(clean_synthetic_chain)
    assert out["ts"].nunique() == 1
    assert out["ts"].iloc[0] == clean_synthetic_chain["ts"].iloc[0]
