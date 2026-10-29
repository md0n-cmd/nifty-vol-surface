"""Compare SVI ATM skew (old a>=0 and new a>=-0.05 fits from svi_refit.csv)
with a model-free skew: OLS of market IV on k over near-ATM OTM quotes, with
a bootstrap CI.

Skew is dIV/dk at k=0, k = ln(K/F). ATM IV (and so the quote window
|k| <= WINDOW_FRAC * ATM_IV * sqrt(T)) uses the old SVI fit, same as
daily_fits.csv's atm_iv.

Usage:
    python skew_check.py [--data-dir data] [--out results/skew_check.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from daily_fits import index_files, load_surface, pick_snapshots
from vol_surface.svi import SVIFit

WINDOW_FRAC = 0.5
N_BOOT = 500
MIN_QUOTES = 8
SEED = 0


def svi_atm(a, b, rho, m, sigma, t):
    """(ATM IV, dIV/dk) at k=0 for a raw-SVI slice."""
    fit = SVIFit(a=a, b=b, rho=rho, m=m, sigma=sigma, n_points=0, cost=0.0)
    iv = float(fit.iv(0.0, t))
    dw = b * (rho + (0.0 - m) / np.sqrt(m ** 2 + sigma ** 2))
    return iv, dw / (2.0 * iv * t)


def market_skew(k, iv, n_boot=N_BOOT, seed=SEED):
    """OLS slope of iv on k with a percentile bootstrap 95% CI (resampling
    quotes). Returns (slope, lo, hi); NaN if there are <3 quotes."""
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    if len(k) < 3 or np.ptp(k) == 0:
        return np.nan, np.nan, np.nan
    slope = np.polyfit(k, iv, 1)[0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(k), len(k))
        if np.ptp(k[i]) > 0:
            draws.append(np.polyfit(k[i], iv[i], 1)[0])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(slope), float(lo), float(hi)


def run(data_dir: Path, refit_csv: Path, out_path: Path) -> pd.DataFrame:
    refit = pd.read_csv(refit_csv).set_index(["date", "underlying", "bucket"])
    picked, _, _ = pick_snapshots(index_files(data_dir))
    rows = []
    for r in picked.itertuples():
        p = refit.loc[(r.date, r.underlying, r.bucket)]
        surface, f = load_surface(r.path)
        t = f["t"]
        atm, skew_old = svi_atm(p.a_old, p.b_old, p.rho_old, p.m_old, p.sigma_old, t)
        _, skew_new = svi_atm(p.a_new, p.b_new, p.rho_new, p.m_new, p.sigma_new, t)
        win = surface[surface["k"].abs() <= WINDOW_FRAC * atm * np.sqrt(t)]
        slope, lo, hi = market_skew(win["k"], win["iv"])
        rows.append(dict(date=r.date, underlying=r.underlying, bucket=r.bucket, dte=r.dte,
                         atm_iv=atm, skew_svi_old=skew_old, skew_svi_new=skew_new,
                         skew_market=slope, ci_lo=lo, ci_hi=hi, n_quotes_in_window=len(win)))
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def summarize(df: pd.DataFrame) -> str:
    ok = df[df["skew_market"].notna()]
    inside = lambda c: ((ok[c] >= ok["ci_lo"]) & (ok[c] <= ok["ci_hi"]))
    both = inside("skew_svi_old") & inside("skew_svi_new")
    lines = [f"slices with a market skew estimate: {len(ok)} of {len(df)}",
             f"old SVI skew inside market CI: {inside('skew_svi_old').sum()}/{len(ok)}",
             f"new SVI skew inside market CI: {inside('skew_svi_new').sum()}/{len(ok)}",
             f"both inside: {both.sum()}/{len(ok)}",
             f"market CI entirely below zero: {(ok['ci_hi'] < 0).sum()}/{len(ok)}"]
    few = df[df["n_quotes_in_window"] < MIN_QUOTES]
    lines.append(f"slices with < {MIN_QUOTES} quotes in window: {len(few)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--refit-csv", type=Path, default=Path("results/svi_refit.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/skew_check.csv"))
    args = ap.parse_args()
    df = run(args.data_dir, args.refit_csv, args.out)
    print(f"wrote {args.out} ({len(df)} rows)")
    print(summarize(df))


if __name__ == "__main__":
    main()
