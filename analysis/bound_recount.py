"""Recount Heston bound hits in docs/results/daily_fits.csv at two tolerances: the
absolute 1e-6 used by daily_fits.py and a 1% relative tolerance (|x - bound| <= 1% of
the bound). Heston parameters only (the SVI bounds depend on each slice's data).

In: docs/results/daily_fits.csv. Out: docs/results/bound_recount.csv (and stdout).
Run from the repo root: python analysis/bound_recount.py
"""
import sys; sys.path.insert(0, ".")
import pandas as pd

from heston.calibration import LOWER_BOUNDS, PARAM_NAMES, UPPER_BOUNDS

d = pd.read_csv("docs/results/daily_fits.csv")
rows = []
for label, tol in [("absolute 1e-6 (daily_fits.py)", lambda b: 1e-6 * max(1, abs(b))),
                   ("1% relative", lambda b: 0.01 * abs(b))]:
    hit = pd.DataFrame({p: (d[f"heston_{p}"] - lo).abs().le(tol(lo)) | (d[f"heston_{p}"] - hi).abs().le(tol(hi))
                        for p, lo, hi in zip(PARAM_NAMES, LOWER_BOUNDS, UPPER_BOUNDS)})
    rows.append(dict(tolerance=label, slices=len(d), any_bound_hit=int(hit.any(axis=1).sum()),
                     **{p: int(hit[p].sum()) for p in PARAM_NAMES}))
out = pd.DataFrame(rows)
out.to_csv("docs/results/bound_recount.csv", index=False)
print(out.to_string(index=False))
