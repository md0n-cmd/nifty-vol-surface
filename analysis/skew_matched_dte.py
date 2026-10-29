"""NIFTY-next slices at DTE 12/13: early vs late skew difference (bootstrap CI, exact permutation). In: data/. Out: results/skew_matched_dte.csv.

Run from the repo root: python analysis/skew_matched_dte.py
"""
import sys; sys.path.insert(0, ".")
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from daily_fits import index_files, load_surface, pick_snapshots
from skew_check import market_skew

picked, _, _ = pick_snapshots(index_files(Path("data")))
nn = picked[(picked.underlying == "NIFTY") & (picked.bucket == "next")]
print("all NIFTY next dte:", dict(zip(nn.date, nn.dte)))
nn = nn[nn.dte.isin([12, 13])].sort_values("date")
def typ(e):
    t = pd.Timestamp(e); return "monthly" if t.weekday() == 1 and (t + pd.Timedelta(days=7)).month != t.month else "weekly"
EARLY = {"2026-09-24", "2026-09-30", "2026-10-01"}; LATE = {"2026-10-14", "2026-10-15", "2026-10-21"}
surf = {r.date: load_surface(r.path)[0] for r in nn.itertuples()}
rng = np.random.default_rng(0); B = 5000
rows, summ = [], []
for hw in [0.02, 0.04]:
    data = {}
    for r in nn.itertuples():
        s = surf[r.date]; w = s[s.k.abs() <= hw]
        data[r.date] = (w.k.to_numpy(), w.iv.to_numpy())
        sl, lo, hi = market_skew(*data[r.date])
        rows.append(dict(section="slice", window=hw, date=r.date, expiry=r.expiry, type=typ(r.expiry), dte=r.dte,
                         group="early" if r.date in EARLY else "late" if r.date in LATE else "other",
                         skew=sl, ci_lo=lo, ci_hi=hi, n_quotes=len(w)))
    sl = {x: np.polyfit(*data[x], 1)[0] for x in data}
    E, L = sorted(EARLY & set(data)), sorted(LATE & set(data))
    diff = np.mean([sl[x] for x in L]) - np.mean([sl[x] for x in E])
    draws = []
    for _ in range(B):
        bs = {}
        for x, (k, iv) in data.items():
            i = rng.integers(0, len(k), len(k))
            bs[x] = np.polyfit(k[i], iv[i], 1)[0] if np.ptp(k[i]) > 0 else np.nan
        draws.append(np.mean([bs[x] for x in L]) - np.mean([bs[x] for x in E]))
    draws = np.array(draws); draws = draws[~np.isnan(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    allx = sorted(data); splits = list(combinations(allx, len(L)))
    stat = [np.mean([sl[x] for x in c]) - np.mean([sl[x] for x in allx if x not in c]) for c in splits]
    p_two = np.mean(np.abs(stat) >= abs(diff) - 1e-12); p_one = np.mean(np.array(stat) >= diff - 1e-12)
    # separate bound: smallest achievable
    summ.append(dict(section="summary", window=hw, group="late-early", mean_early=np.mean([sl[x] for x in E]),
                     mean_late=np.mean([sl[x] for x in L]), skew=diff, ci_lo=lo, ci_hi=hi, n_quotes=len(draws),
                     p_perm_two_sided=p_two, p_perm_one_sided_late_flatter=p_one, n_splits=len(splits)))
out = pd.DataFrame(rows + summ); out.to_csv("results/skew_matched_dte.csv", index=False)
pd.set_option("display.width", 250)
print(out[out.section == "slice"].drop(columns="section").round(3).to_string(index=False))
print(out[out.section == "summary"].dropna(axis=1, how="all").round(4).to_string(index=False))
