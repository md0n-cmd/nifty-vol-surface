"""SVI-only refit of the daily_fits slices with a's lower bound loosened.

Usage:
    python svi_refit.py [--data-dir data] [--out results/svi_refit.csv] [--a-lower -0.05]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from daily_fits import _bound_hits, _rmse, index_files, load_surface, pick_snapshots
from vol_surface.svi import PARAM_NAMES, fit_svi_slice, svi_bounds

A_LOWER_NEW = -0.05
COMMITTED_0925 = Path("results/date=2026-09-25_svi_params.json")


def refit(path, a_lower: float) -> dict:
    surface, f = load_surface(path)
    k, w = surface["k"].to_numpy(), surface["w"].to_numpy()
    fit = fit_svi_slice(k, w, a_lower=a_lower)
    lo, hi = svi_bounds(k, w, a_lower)
    vals = [getattr(fit, p) for p in PARAM_NAMES]
    return dict(n=len(k), rmse=_rmse(fit.iv(k, f["t"]), surface["iv"].to_numpy()),
                cost=fit.cost, bound_hit=_bound_hits(PARAM_NAMES, vals, lo, hi),
                min_var=fit.a + fit.b * fit.sigma * np.sqrt(1 - fit.rho ** 2),
                **dict(zip(PARAM_NAMES, vals)))


def run(data_dir: Path, daily_csv: Path, out_path: Path, a_lower: float):
    old = pd.read_csv(daily_csv).set_index(["date", "underlying", "bucket"])
    picked, _, _ = pick_snapshots(index_files(data_dir))
    rows = []
    for r in picked.itertuples():
        o = old.loc[(r.date, r.underlying, r.bucket)]
        new = refit(r.path, a_lower)
        rows.append(dict(date=r.date, underlying=r.underlying, bucket=r.bucket, expiry=r.expiry,
                         dte=r.dte, n=new["n"],
                         a_old=o["svi_a"], a_new=new["a"], rho_old=o["svi_rho"], rho_new=new["rho"],
                         b_old=o["svi_b"], b_new=new["b"],
                         m_old=o["svi_m"], m_new=new["m"],
                         sigma_old=o["svi_sigma"], sigma_new=new["sigma"],
                         rmse_old=o["svi_rmse"], rmse_new=new["rmse"],
                         min_var_new=new["min_var"], bound_hit_old=o["svi_bound_hit"],
                         bound_hit_new=new["bound_hit"]))
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def check_0925(data_dir: Path, a_lower: float) -> pd.DataFrame:
    """Refit 2026-09-25 at the latest snapshot per series (what the pipeline
    used for the committed JSON) with both a bounds; compare rho."""
    committed = json.loads(COMMITTED_0925.read_text())
    idx = index_files(data_dir)
    day = idx[idx["date"] == "2026-09-25"]
    latest = day.loc[day.groupby(["underlying", "expiry"])["ts"].idxmax()]
    rows = []
    for r in latest.itertuples():
        c = committed[f"{r.underlying}_{r.expiry}"]
        base, loose = refit(r.path, 0.0), refit(r.path, a_lower)
        rows.append(dict(slice=f"{r.underlying} {r.expiry}", ts=r.ts.strftime("%H:%M:%S"),
                         rho_committed=c["rho"], rho_a0=base["rho"], rho_loose=loose["rho"],
                         a_a0=base["a"], a_loose=loose["a"], b_committed=c["b"], b_a0=base["b"],
                         b_loose=loose["b"], rmse_a0=base["rmse"], rmse_loose=loose["rmse"],
                         hit_loose=loose["bound_hit"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--daily-csv", type=Path, default=Path("results/daily_fits.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/svi_refit.csv"))
    ap.add_argument("--a-lower", type=float, default=A_LOWER_NEW)
    args = ap.parse_args()
    df = run(args.data_dir, args.daily_csv, args.out, args.a_lower)
    print(f"wrote {args.out} ({len(df)} rows)")
    print(f"slices pinning any bound: old={df['bound_hit_old'].notna().sum()} new={df['bound_hit_new'].notna().sum()}")
    print(df["bound_hit_new"].value_counts().to_string())
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(check_0925(args.data_dir, args.a_lower).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
