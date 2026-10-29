"""Raw-ATM-IV skew with three window definitions, NIFTY-next drift tests, expiry types, raw vs SVI ATM IV. In: results/skew_check.csv, data/. Out: results/skew_robustness.csv.

Run from the repo root: python analysis/skew_robustness.py
"""
import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from daily_fits import index_files, load_surface, pick_snapshots
from skew_check import market_skew

sk = pd.read_csv("results/skew_check.csv")
picked, _, _ = pick_snapshots(index_files(Path("data")))
rows = []
for r in picked.itertuples():
    s, f = load_surface(r.path); t = f["t"]
    h = s[s.k.abs() <= 0.01]
    if len(h) < 3: h = s[s.k.abs() <= 0.02]
    raw = float(h.iv.median()) if len(h) >= 3 else np.nan
    row = dict(date=r.date, underlying=r.underlying, bucket=r.bucket, expiry=r.expiry, dte=r.dte, raw_atm_iv=raw, n_atm=len(h))
    for name, hw in [("rawwin", 0.5 * raw * np.sqrt(t)), ("k02", 0.02), ("k04", 0.04)]:
        w = s[s.k.abs() <= hw]; sl, lo, hi = market_skew(w.k, w.iv)
        row.update({f"skew_{name}": sl, f"ci_lo_{name}": lo, f"ci_hi_{name}": hi, f"n_{name}": len(w), f"hw_{name}": hw})
    rows.append(row)
d = pd.DataFrame(rows).merge(sk[["date", "underlying", "bucket", "atm_iv"]], on=["date", "underlying", "bucket"]).rename(columns={"atm_iv": "svi_atm_iv"})
dates = sorted(d.date.unique()); d["di"] = d.date.map({x: i for i, x in enumerate(dates)})
chk = np.abs(d.skew_rawwin - sk.set_index(["date","underlying","bucket"]).loc[list(zip(d.date,d.underlying,d.bucket)),"skew_market"].values)
print("raw-window skew vs previous skew_market max abs diff:", chk.max().round(3), "| n_window match prev:", (d.n_rawwin.values == sk.set_index(["date","underlying","bucket"]).loc[list(zip(d.date,d.underlying,d.bucket)),"n_quotes_in_window"].values).mean())

rng = np.random.default_rng(0); P = 5000
out = []
N = d[d.underlying == "NIFTY"].copy(); N["x"] = np.log(N.dte)
ntests = 0
for name in ["rawwin", "k02", "k04"]:
    for excl in [False, True]:
        s = N[(N[f"skew_{name}"] < 0)]
        if excl: s = s[s[f"n_{name}"] >= 8]
        y = np.log(-s[f"skew_{name}"]); b, a = np.polyfit(s.x, y, 1)
        s = s.assign(res=y - (a + b * s.x))
        nx = s[s.bucket == "next"].sort_values("di")
        rho, p_sp = stats.spearmanr(nx.di, nx.res)
        perm = np.array([stats.spearmanr(rng.permutation(nx.di.values), nx.res.values).statistic for _ in range(P)])
        p_perm = (1 + (np.abs(perm) >= abs(rho) - 1e-12).sum()) / (P + 1)
        ntests += 2
        out.append(dict(section="nifty_next_drift", name=name, excl_lt8=excl, exponent=b, n_fit=len(s), n_next=len(nx),
                        rho=rho, p_spearman=p_sp, p_perm=p_perm))
drift = pd.DataFrame(out)

# expiries
def monthly(e):
    t = pd.Timestamp(e); nxt = t + pd.Timedelta(days=7)
    return "monthly" if (t.weekday() == 1 and nxt.month != t.month) else "weekly"
nn = N[N.bucket == "next"][["date", "expiry", "dte"]].copy(); nn["type"] = nn.expiry.map(monthly)
nn["weekday"] = pd.to_datetime(nn.expiry).dt.day_name()
iv = N[["date", "bucket", "expiry", "dte", "raw_atm_iv", "svi_atm_iv"]].copy()
iv["diff_pts"] = (iv.svi_atm_iv - iv.raw_atm_iv) * 100; iv["gt2"] = iv.diff_pts.abs() > 2

pd.set_option("display.width", 250, "display.max_rows", 200)
print(d[d.underlying == "NIFTY"].sort_values(["date","bucket"])[["date","bucket","dte","raw_atm_iv","n_atm","skew_rawwin","n_rawwin","skew_k02","n_k02","skew_k04","n_k04"]].round(3).to_string(index=False))
print(drift.round(4).to_string(index=False)); print("p-values:", ntests)
print(nn.to_string(index=False))
print(iv.sort_values(["date","bucket"]).round(3).to_string(index=False))
print("n slices NIFTY", len(iv), ">2pts:", iv.gt2.sum())
print(d[d.underlying=="BANKNIFTY"][["raw_atm_iv","svi_atm_iv"]].assign(df=lambda x:(x.svi_atm_iv-x.raw_atm_iv)*100).df.abs().describe().round(2))

parts = [d.assign(section="slice"), drift, nn.assign(section="nifty_next_expiry"), iv.assign(section="nifty_iv_compare")]
pd.concat(parts, ignore_index=True).to_csv("results/skew_robustness.csv", index=False)
