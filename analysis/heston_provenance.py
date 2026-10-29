"""Provenance and fragility check for the stored BANKNIFTY 2026-10-27 Heston set.

For every snapshot on 2026-09-25 from 11:00 to 11:27 IST (inclusive of the 11:27
tick): the parity forward, discount and T of BANKNIFTY 10-27, which tick matches
the stored (F, discount, T), a single-start Heston calibration on that tick, and
the four a>=0 SVI fits (BANKNIFTY 09-29 and 10-27, NIFTY 09-29 and 10-06) compared
with the SVI JSONs still on disk in results/.

In: data/date=2026-09-25/, results/date=2026-09-25_svi_params.json.
Out: docs/results/heston_fragility.csv, docs/results/provenance_parity_svi.csv.
Run from the repo root: python analysis/heston_provenance.py
"""
import sys; sys.path.insert(0, ".")
import json
from pathlib import Path

import numpy as np
import pandas as pd

from daily_fits import _bound_hits, _rmse, load_surface
from heston.calibration import LOWER_BOUNDS, PARAM_NAMES, UPPER_BOUNDS, calibrate_heston
from heston.pricing import _expected_total_variance
from vol_surface.svi import fit_svi_slice

DAY = Path("data/date=2026-09-25")
STORED = dict(v0=0.00010, kappa=3.247, theta=0.15522, xi=0.946, rho=-0.499)
STORED_F, STORED_D, STORED_T = 56022.49113, 0.987123, 0.088173
SLICES = [("BANKNIFTY", "2026-09-29"), ("BANKNIFTY", "2026-10-27"),
          ("NIFTY", "2026-09-29"), ("NIFTY", "2026-10-06")]
JSON_PATH = Path("results/date=2026-09-25_svi_params.json")
FIT_OUT = Path("docs/results/heston_fragility.csv")
SVI_OUT = Path("docs/results/provenance_parity_svi.csv")
LO, HI = "110000", "112759"


def ticks():
    files = sorted(DAY.glob("BANKNIFTY_2026-10-27_*.parquet"))
    return [f.stem.rsplit("_", 1)[1] for f in files if LO <= f.stem.rsplit("_", 1)[1] <= HI]


def main():
    stored_json = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else {}
    fit_rows, svi_rows = [], []
    for tick in ticks():
        for u, exp in SLICES:
            path = DAY / f"{u}_{exp}_{tick}.parquet"
            surface, f = load_surface(path)
            k, w = surface["k"].to_numpy(), surface["w"].to_numpy()
            svi = fit_svi_slice(k, w)
            js = stored_json.get(f"{u}_{exp}", {})
            svi_rows.append(dict(tick=tick, underlying=u, expiry=exp, forward=f["forward"],
                                 discount=f["discount"], t=f["t"], n=len(surface), svi_a=svi.a,
                                 svi_rho=svi.rho, json_rho=js.get("rho"), json_n=js.get("n_points"),
                                 d_rho=svi.rho - js["rho"] if js else np.nan))
            if (u, exp) != ("BANKNIFTY", "2026-10-27"):
                continue
            match = (abs(f["forward"] - STORED_F) < 0.01 and abs(f["discount"] - STORED_D) < 5e-7
                     and abs(f["t"] - STORED_T) < 5e-7)
            h = calibrate_heston(surface, f["forward"], f["discount"], f["t"])
            vals = [getattr(h, p) for p in PARAM_NAMES]
            model_iv = h.implied_vol_curve(f["forward"], surface["strike"].to_numpy(), f["t"],
                                           f["discount"], surface["type"].to_numpy())
            avg_var = _expected_total_variance(f["t"], h.v0, h.kappa, h.theta) / f["t"]
            fit_rows.append(dict(date="2026-09-25", tick=tick, forward=f["forward"], discount=f["discount"],
                                 t=f["t"], n=len(surface), **dict(zip(PARAM_NAMES, vals)), cost=h.cost,
                                 heston_rmse=_rmse(model_iv, surface["iv"].to_numpy()),
                                 bound_hit=_bound_hits(PARAM_NAMES, vals, LOWER_BOUNDS, UPPER_BOUNDS),
                                 sqrt_v0=np.sqrt(h.v0), sqrt_theta=np.sqrt(h.theta),
                                 avg_vol_to_expiry=np.sqrt(avg_var), matches_stored_F_D_T=match,
                                 **{f"d_{p}": v - STORED[p] for p, v in zip(PARAM_NAMES, vals)}))
    fit, svi = pd.DataFrame(fit_rows), pd.DataFrame(svi_rows)
    FIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fit.to_csv(FIT_OUT, index=False); svi.to_csv(SVI_OUT, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(fit.drop(columns=[c for c in fit if c.startswith("d_")]).round(6).to_string(index=False))
    print(svi.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
