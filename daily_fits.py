"""Per-day SVI + single-start Heston fits at the snapshot nearest 14:30 IST.

Usage:
    python daily_fits.py [--data-dir data] [--out results/daily_fits.csv]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from heston.calibration import LOWER_BOUNDS, PARAM_NAMES as H_PARAMS, UPPER_BOUNDS, calibrate_heston
from vol_surface.clean import clean_chain
from vol_surface.forward import forwards_by_expiry
from vol_surface.implied_vol import attach_iv_surface
from vol_surface.otm import select_otm
from vol_surface.svi import PARAM_NAMES as S_PARAMS, fit_svi_slice, svi_bounds

STALE_DATES = {"2026-10-02", "2026-10-20"}
TARGET = pd.Timedelta(hours=14, minutes=30)
MAX_GAP = pd.Timedelta(minutes=15)
MIN_DTE = 2
BUCKETS = ("near", "next")
BOUND_TOL = 1e-6
FILE_RE = re.compile(r"^(?P<u>[A-Z]+)_(?P<exp>\d{4}-\d{2}-\d{2})_(?P<hms>\d{6})\.parquet$")


def index_files(data_dir: Path) -> pd.DataFrame:
    """One row per snapshot file: date, underlying, expiry, ts (naive IST), path."""
    rows = []
    for d in sorted(Path(data_dir).glob("date=*")):
        date = d.name[5:]
        for f in d.glob("*.parquet"):
            m = FILE_RE.match(f.name)
            if m:
                ts = pd.Timestamp(f"{date} {m['hms'][:2]}:{m['hms'][2:4]}:{m['hms'][4:]}")
                rows.append((date, m["u"], m["exp"], ts, f))
    return pd.DataFrame(rows, columns=["date", "underlying", "expiry", "ts", "path"])


def pick_snapshots(idx: pd.DataFrame):
    """Pick, per date, the file nearest 14:30 for each (underlying, expiry).

    Returns (picked, skipped_dates, dropped) where picked has columns
    date/underlying/expiry/ts/path/dte/bucket. A date is skipped when no
    snapshot at all is within MAX_GAP of 14:30; individual slices further
    than MAX_GAP are dropped. Buckets are assigned (near/next by expiry
    order per underlying) before the DTE filter.
    """
    idx = idx[~idx["date"].isin(STALE_DATES)].copy()
    idx["gap"] = (idx["ts"] - (idx["ts"].dt.normalize() + TARGET)).abs()
    skipped, dropped, picked = [], [], []
    for date, day in idx.groupby("date"):
        if day["gap"].min() > MAX_GAP:
            skipped.append((date, f"no snapshot within 15 min of 14:30 (nearest gap {day['gap'].min()})"))
            continue
        best = day.loc[day.groupby(["underlying", "expiry"])["gap"].idxmin()]
        for u, grp in best.groupby("underlying"):
            grp = grp.sort_values("expiry")
            for rank, row in enumerate(grp.itertuples()):
                dte = (pd.Timestamp(row.expiry) - pd.Timestamp(date)).days
                bucket = BUCKETS[rank] if rank < len(BUCKETS) else "far"
                reason = None
                if row.gap > MAX_GAP:
                    reason = f"nearest snapshot {row.gap} from 14:30"
                elif bucket == "far":
                    reason = "not near/next bucket"
                elif dte < MIN_DTE:
                    reason = f"DTE {dte} < {MIN_DTE}"
                rec = dict(date=date, underlying=u, expiry=row.expiry, ts=row.ts,
                           path=row.path, dte=dte, bucket=bucket)
                (dropped if reason else picked).append({**rec, "reason": reason} if reason else rec)
    return pd.DataFrame(picked), skipped, pd.DataFrame(dropped)


def _bound_hits(names, values, lower, upper):
    return ",".join(n for n, v, lo, hi in zip(names, values, lower, upper)
                    if abs(v - lo) <= BOUND_TOL * max(1, abs(lo)) or abs(v - hi) <= BOUND_TOL * max(1, abs(hi)))


def _rmse(model_iv, market_iv):
    ok = ~np.isnan(model_iv)
    return float(np.sqrt(np.mean((model_iv[ok] - market_iv[ok]) ** 2))) if ok.any() else np.nan


def load_surface(path):
    """Clean, OTM-filter and attach IV for one snapshot file.
    Returns (surface, forward-row)."""
    raw = pd.read_parquet(path)
    raw["ts"] = pd.to_datetime(raw["ts"])
    raw["expiry"] = pd.to_datetime(raw["expiry"]).dt.date.astype(str)
    cleaned = clean_chain(raw)
    fwd = forwards_by_expiry(cleaned)
    return attach_iv_surface(select_otm(cleaned, fwd), fwd), fwd.iloc[0]


def fit_slice(row) -> dict:
    """Clean, OTM-filter, SVI-fit and Heston-calibrate one snapshot file."""
    surface, f = load_surface(row.path)
    out = {"n": len(surface)}

    k, w = surface["k"].to_numpy(), surface["w"].to_numpy()
    svi = fit_svi_slice(k, w)
    lo, hi = svi_bounds(k, w)
    svi_vals = [getattr(svi, p) for p in S_PARAMS]
    out.update(atm_iv=float(svi.iv(0.0, f["t"])), svi_cost=svi.cost,
               svi_rmse=_rmse(svi.iv(k, f["t"]), surface["iv"].to_numpy()),
               svi_bound_hit=_bound_hits(S_PARAMS, svi_vals, lo, hi),
               **{f"svi_{p}": v for p, v in zip(S_PARAMS, svi_vals)})
    try:
        h = calibrate_heston(surface, f["forward"], f["discount"], f["t"])
        h_vals = [getattr(h, p) for p in H_PARAMS]
        model_iv = h.implied_vol_curve(f["forward"], surface["strike"].to_numpy(), f["t"],
                                       f["discount"], surface["type"].to_numpy())
        out.update(heston_cost=h.cost, heston_rmse=_rmse(model_iv, surface["iv"].to_numpy()),
                   heston_bound_hit=_bound_hits(H_PARAMS, h_vals, LOWER_BOUNDS, UPPER_BOUNDS),
                   **{f"heston_{p}": v for p, v in zip(H_PARAMS, h_vals)})
    except Exception as e:  # keep the SVI result; record the Heston failure
        out["error"] = f"heston: {type(e).__name__}: {e}"
    return out


COLUMNS = (["date", "underlying", "bucket", "expiry", "dte", "n", "atm_iv"]
           + [f"svi_{p}" for p in S_PARAMS] + ["svi_cost"]
           + [f"heston_{p}" for p in H_PARAMS] + ["heston_cost",
              "svi_bound_hit", "heston_bound_hit", "svi_rmse", "heston_rmse", "error"])


def run(data_dir: Path, out_path: Path):
    picked, skipped, dropped = pick_snapshots(index_files(data_dir))
    rows = []
    for row in picked.itertuples():
        base = dict(date=row.date, underlying=row.underlying, bucket=row.bucket,
                    expiry=row.expiry, dte=row.dte)
        try:
            rows.append({**base, **fit_slice(row)})
        except Exception as e:
            rows.append({**base, "error": f"{type(e).__name__}: {e}"})
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df, skipped, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/daily_fits.csv"))
    args = ap.parse_args()
    df, skipped, dropped = run(args.data_dir, args.out)
    print(f"wrote {args.out} ({len(df)} rows)")
    for d, why in skipped:
        print(f"skipped date {d}: {why}")
    if len(dropped):
        print(dropped[["date", "underlying", "expiry", "dte", "bucket", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
