"""Intraday (12:00-15:27) NIFTY-next skew on 10-14, 10-15, 10-21 plus stale-quote check. In: data/. Out: results/skew_1021_check.csv.

Run from the repo root: python analysis/skew_1021_check.py
"""
import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from daily_fits import index_files, load_surface

idx = index_files(Path("data"))
CASES = {"2026-10-21": "2026-11-03", "2026-10-14": "2026-10-27", "2026-10-15": "2026-10-27"}
HW = 0.02
snap, stale = [], []
for date, exp in CASES.items():
    d = idx[(idx.date == date) & (idx.underlying == "NIFTY") & (idx.expiry == exp)].sort_values("ts")
    d = d[(d.ts >= pd.Timestamp(f"{date} 12:00")) & (d.ts <= pd.Timestamp(f"{date} 15:27:59"))]
    prev = None
    for r in d.itertuples():
        raw = pd.read_parquet(r.path); spot = float(raw.spot.iloc[0])
        try:
            s, f = load_surface(r.path); w = s[s.k.abs() <= HW]
            sk = np.polyfit(w.k, w.iv, 1)[0] if len(w) >= 3 and np.ptp(w.k) > 0 else np.nan
            n = len(w); fwd = f["forward"]
        except Exception as e:
            sk, n, fwd, w = np.nan, 0, np.nan, s.iloc[:0] if 's' in dir() else None
        snap.append(dict(section="snapshot", date=date, expiry=exp, ts=r.ts.strftime("%H:%M:%S"), spot=spot, forward=fwd, skew=sk, n_quotes=n))
        live = raw[(raw.bid > 0) & (raw.ask > 0)].set_index(["strike", "type"])[["bid", "ask", "ltp"]]
        if prev is not None:
            j = live.join(prev, how="inner", lsuffix="", rsuffix="_p")
            same = ((j.bid == j.bid_p) & (j.ask == j.ask_p) & (j.ltp == j.ltp_p))
            k_atm = np.abs(np.log(j.index.get_level_values(0) / fwd)) <= 0.02 if np.isfinite(fwd) else np.zeros(len(j), bool)
            stale.append(dict(section="stale", date=date, expiry=exp, ts=snap[-1]["ts"], prev_ts=prev_ts,
                              spot_same=float(spot == prev_spot), n_legs=len(j), share_identical=same.mean(),
                              n_legs_atm=int(k_atm.sum()), share_identical_atm=same[k_atm].mean() if k_atm.any() else np.nan,
                              share_bidask_identical=((j.bid == j.bid_p) & (j.ask == j.ask_p)).mean()))
        prev, prev_ts, prev_spot = live.rename(columns=lambda c: c + "_p"), snap[-1]["ts"], spot
S = pd.DataFrame(snap); T = pd.DataFrame(stale)
summ = S.groupby("date").agg(n_snap=("skew", "size"), skew_min=("skew", "min"), skew_median=("skew", "median"), skew_max=("skew", "max"),
                             skew_sd=("skew", "std"), n_quotes_min=("n_quotes", "min"), n_quotes_max=("n_quotes", "max"),
                             spot_min=("spot", "min"), spot_max=("spot", "max")).reset_index().assign(section="summary")
tsum = T.groupby("date").agg(share_identical_mean=("share_identical", "mean"), share_identical_atm_mean=("share_identical_atm", "mean"),
                              share_identical_max=("share_identical", "max"), spot_same_share=("spot_same", "mean")).reset_index().assign(section="stale_summary")
pd.concat([S, summ, T, tsum], ignore_index=True).to_csv("results/skew_1021_check.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 300)
print(summ.drop(columns="section").round(3).to_string(index=False)); print(tsum.drop(columns="section").round(3).to_string(index=False))
for date in CASES:
    s = S[S.date == date]
    print(date, len(s)); print(s[["ts", "spot", "skew", "n_quotes"]].round(3).to_string(index=False, header=False))
