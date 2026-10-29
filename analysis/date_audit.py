"""Audit each data/date=*/ folder: file count, distinct snapshot timestamps,
first/last timestamp, distinct spot values per underlying, and flags.

Flags: EMPTY (no parquet files), LOW_TS (< 120 distinct timestamps; a full
day is 125), STALE (spot identical across every snapshot of every
underlying, i.e. a closed-market chain).

In: data/. Out: results/date_audit.csv.
Run from the repo root: python analysis/date_audit.py
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA, OUT = Path("data"), Path("results/date_audit.csv")
LOW_TS = 120
UNDERLYINGS = ("NIFTY", "BANKNIFTY")


def audit_day(day: Path) -> dict:
    rows = []
    for f in day.glob("*.parquet"):
        t = pq.read_table(f, columns=["ts", "underlying", "spot"]).slice(0, 1).to_pylist()
        if t:
            rows.append(t[0])
    ts = sorted({r["ts"] for r in rows})
    out = {"date": day.name[5:], "files": len(rows), "distinct_ts": len(ts),
           "first_ts": ts[0] if ts else "", "last_ts": ts[-1] if ts else ""}
    for u in UNDERLYINGS:
        out[f"distinct_spot_{u}"] = len({r["spot"] for r in rows if r["underlying"] == u})
    flags = []
    if not rows:
        flags.append("EMPTY")
    if len(ts) < LOW_TS:
        flags.append("LOW_TS")
    if rows and all(out[f"distinct_spot_{u}"] <= 1 for u in UNDERLYINGS):
        flags.append("STALE")
    out["flags"] = ",".join(flags)
    return out


def main():
    df = pd.DataFrame([audit_day(d) for d in sorted(DATA.glob("date=*")) if d.is_dir()])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
