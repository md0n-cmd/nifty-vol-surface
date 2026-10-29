"""Which RMSE does each figure in the earlier post reproduce?

On the 2026-09-25 11:06:02 IST snapshot, for BANKNIFTY 2026-09-29 and 2026-10-27, on the
same cleaned OTM quotes: single-start Heston calibration and the a >= 0 SVI fit. Reports
three RMSEs in vol points (IV x 100): Heston IV vs market IV, Heston IV vs the SVI curve,
and SVI vs market IV, at the quote strikes (Heston points whose price cannot be inverted
are excluded from the two Heston RMSEs and counted), plus the same two Heston-vs-SVI/
market comparisons on a dense k-grid over the quote range for reference.

In: data/date=2026-09-25/. Out: docs/results/post_rmse_check.csv.
Run from the repo root: python analysis/post_rmse_check.py
"""
import sys; sys.path.insert(0, ".")
from pathlib import Path

import numpy as np
import pandas as pd

from daily_fits import load_surface
from heston.calibration import PARAM_NAMES, calibrate_heston
from vol_surface.svi import fit_svi_slice

DAY, TICK = Path("data/date=2026-09-25"), "110602"
SLICES = [("BANKNIFTY", "2026-09-29"), ("BANKNIFTY", "2026-10-27")]
OUT = Path("docs/results/post_rmse_check.csv")


def rmse_pts(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)) * 100), int(ok.sum())


def main():
    rows = []
    for u, exp in SLICES:
        surface, f = load_surface(DAY / f"{u}_{exp}_{TICK}.parquet")
        k, mkt = surface["k"].to_numpy(), surface["iv"].to_numpy()
        strikes, types = surface["strike"].to_numpy(), surface["type"].to_numpy()
        svi = fit_svi_slice(k, surface["w"].to_numpy())
        h = calibrate_heston(surface, f["forward"], f["discount"], f["t"])
        hiv = h.implied_vol_curve(f["forward"], strikes, f["t"], f["discount"], types)
        siv = svi.iv(k, f["t"])
        hm, n_hm = rmse_pts(hiv, mkt); hs, n_hs = rmse_pts(hiv, siv); sm, n_sm = rmse_pts(siv, mkt)
        # dense grid: Heston and SVI on OTM-style calls/puts at k over the quote range
        kg = np.linspace(k.min(), k.max(), 200); Kg = f["forward"] * np.exp(kg)
        tg = np.where(kg < 0, "PE", "CE")
        hg = h.implied_vol_curve(f["forward"], Kg, f["t"], f["discount"], tg)
        hs_grid, n_grid = rmse_pts(hg, svi.iv(kg, f["t"]))
        rows.append(dict(underlying=u, expiry=exp, tick=TICK, n_quotes=len(surface),
                         rmse_heston_vs_market_volpts=hm, n_points_heston_vs_market=n_hm,
                         rmse_heston_vs_svi_volpts=hs, n_points_heston_vs_svi=n_hs,
                         rmse_svi_vs_market_volpts=sm, n_points_svi_vs_market=n_sm,
                         mean_market_minus_heston_volpts=float(np.nanmean(mkt - hiv) * 100),
                         rmse_heston_vs_svi_dense_grid_volpts=hs_grid, n_points_dense_grid=n_grid,
                         **{p: getattr(h, p) for p in PARAM_NAMES}, heston_cost=h.cost))
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(df.round(4).T.to_string())


if __name__ == "__main__":
    main()
