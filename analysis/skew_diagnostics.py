"""Skew-vs-DTE exponent (date-cluster bootstrap), residual drift tests, spot/ATM IV levels. In: results/skew_check.csv, data/. Out: results/skew_diagnostics.csv.

Run from the repo root: python analysis/skew_diagnostics.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy import stats
from daily_fits import index_files, pick_snapshots

rng = np.random.default_rng(0)
d = pd.read_csv("results/skew_check.csv")
d["expiry"] = (pd.to_datetime(d.date) + pd.to_timedelta(d.dte, unit="D")).dt.strftime("%Y-%m-%d")
dates = sorted(d.date.unique()); d["di"] = d.date.map({x: i for i, x in enumerate(dates)})
d["y"] = np.log(-d.skew_market); d["x"] = np.log(d.dte)
assert (d.skew_market < 0).all()
U = ["BANKNIFTY", "NIFTY"]

def fit(df):
    """interaction OLS: y = alpha_u + beta_u * x. Returns {u: (alpha, beta)}."""
    return {u: np.polyfit(df.x[df.underlying == u], df.y[df.underlying == u], 1)[::-1] for u in U}

out = []
f = fit(d)
for u in U:
    out.append(dict(section="exponent", underlying=u, name="beta_log_dte", value=f[u][1]))
# date-cluster bootstrap
B = 5000; draws = {u: [] for u in U}
for _ in range(B):
    ds = rng.choice(dates, len(dates), replace=True)
    s = pd.concat([d[d.date == x] for x in ds])
    if any(s[s.underlying == u].x.nunique() < 2 for u in U): continue
    g = fit(s)
    for u in U: draws[u].append(g[u][1])
for o in out:
    lo, hi = np.percentile(draws[o["underlying"]], [2.5, 97.5])
    o.update(ci_lo=lo, ci_hi=hi, n=len(draws[o["underlying"]]), note="date-cluster bootstrap (resample 12 dates), percentile 95%")
# slice-level (naive) CI for contrast
for u in U:
    s = d[d.underlying == u]; r = stats.linregress(s.x, s.y); t = stats.t.ppf(0.975, len(s) - 2)
    out.append(dict(section="exponent_naive", underlying=u, name="beta_log_dte", value=r.slope,
                    ci_lo=r.slope - t * r.stderr, ci_hi=r.slope + t * r.stderr, n=len(s), note="slice-level OLS CI (ignores date clustering)"))
# residual drift tests
d["resid"] = [r for r in np.nan_to_num(np.zeros(len(d)))]
for u in U:
    a, b = f[u]; m = d.underlying == u
    d.loc[m, "resid"] = d.y[m] - (a + b * d.x[m])
P = 5000
for (u, bk), s in d.groupby(["underlying", "bucket"]):
    s = s.sort_values("di"); rho, p_sp = stats.spearmanr(s.di, s.resid)
    perm = np.array([stats.spearmanr(rng.permutation(s.di.values), s.resid.values).statistic for _ in range(P)])
    p_perm = (1 + (np.abs(perm) >= abs(rho) - 1e-12).sum()) / (P + 1)
    out.append(dict(section="resid_drift", underlying=u, bucket=bk, name="spearman_rho", value=rho, p_spearman=p_sp,
                    p_perm=p_perm, n=len(s), note=f"resid sd={s.resid.std():.3f}; perm=5000 shuffles of date order, two-sided |rho|"))
# levels: atm_iv + spot
picked, _, _ = pick_snapshots(index_files(__import__("pathlib").Path("data")))
picked["spot"] = [pq.read_table(p, columns=["spot"]).column(0)[0].as_py() for p in picked.path]
lv = d.merge(picked[["date", "underlying", "bucket", "ts", "spot"]], on=["date", "underlying", "bucket"])
for r in lv.itertuples():
    out.append(dict(section="level", date=r.date, underlying=r.underlying, bucket=r.bucket, name="atm_iv_spot",
                    value=r.atm_iv, extra=r.spot, note=f"expiry={r.expiry} dte={r.dte} ts={r.ts:%H:%M:%S}"))
res = pd.DataFrame(out)
res.to_csv("results/skew_diagnostics.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 200)
print(res[res.section != "level"].drop(columns=["date", "extra"]).round(4).to_string(index=False))
w = lv[lv.date.isin(["2026-10-16", "2026-10-21"])].sort_values(["underlying", "date", "bucket"])
print(w[["date", "underlying", "bucket", "expiry", "dte", "atm_iv", "spot", "ts"]].round(4).to_string(index=False))
sp = lv.drop_duplicates(["date", "underlying"]).pivot(index="date", columns="underlying", values="spot")
print(sp.round(1)); 
print(lv.pivot_table(index="date", columns=["underlying", "bucket"], values="atm_iv").round(4))
