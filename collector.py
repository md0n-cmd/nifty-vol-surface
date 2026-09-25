"""Snapshot Nifty/BankNifty option chains from DhanHQ v2 and store as Parquet.
Run once per cron tick; exits outside market hours."""
import os, time, logging, datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
BASE = "https://api.dhan.co/v2"
HEADERS = {
    "access-token": os.environ["DHAN_ACCESS_TOKEN"],
    "client-id": os.environ["DHAN_CLIENT_ID"],
    "Content-Type": "application/json",
}
UNDERLYINGS = {"NIFTY": 13, "BANKNIFTY": 25}   # security IDs, index segment
SEG = "IDX_I"
N_EXPIRIES = 2                                  # nearest expiries to record
RATE_SLEEP = 3.1                                # option chain limit ~1 req / 3 s
DATA_DIR = Path(os.environ.get("DATA_DIR", "~/optdata")).expanduser()

logging.basicConfig(
    filename=DATA_DIR / "collector.log" if DATA_DIR.exists() else None,
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)


def post(path, body, retries=3):
    for i in range(retries):
        try:
            r = requests.post(f"{BASE}{path}", json=body, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                return r.json()
            logging.warning("%s %s %s", path, r.status_code, r.text[:200])
        except requests.RequestException as e:
            logging.warning("%s error %s", path, e)
        time.sleep(RATE_SLEEP * (i + 1))
    raise RuntimeError(f"failed {path} {body}")


def market_open(now):
    return now.weekday() < 5 and dt.time(9, 15) <= now.time() <= dt.time(15, 30)


def flatten(chain, ts, name, expiry):
    spot = chain["last_price"]
    rows = []
    for strike, legs in chain["oc"].items():
        for side in ("ce", "pe"):
            o = legs.get(side)
            if not o:
                continue
            g = o.get("greeks") or {}
            rows.append({
                "ts": ts, "underlying": name, "expiry": expiry, "spot": spot,
                "strike": float(strike), "type": side.upper(),
                "ltp": o.get("last_price"),
                "bid": o.get("top_bid_price"), "bid_qty": o.get("top_bid_quantity"),
                "ask": o.get("top_ask_price"), "ask_qty": o.get("top_ask_quantity"),
                "oi": o.get("oi"), "volume": o.get("volume"),
                "iv": o.get("implied_volatility"),
                "delta": g.get("delta"), "gamma": g.get("gamma"),
                "theta": g.get("theta"), "vega": g.get("vega"),
            })
    return rows


def main():
    now = dt.datetime.now(IST)
    if not market_open(now):
        return
    ts = now.replace(microsecond=0)
    day_dir = DATA_DIR / f"date={ts.date()}"
    day_dir.mkdir(parents=True, exist_ok=True)

    for name, sid in UNDERLYINGS.items():
        base = {"UnderlyingScrip": sid, "UnderlyingSeg": SEG}
        expiries = post("/optionchain/expirylist", base)["data"][:N_EXPIRIES]
        time.sleep(RATE_SLEEP)
        for exp in expiries:
            chain = post("/optionchain", {**base, "Expiry": exp})["data"]
            rows = flatten(chain, ts, name, exp)
            out = day_dir / f"{name}_{exp}_{ts:%H%M%S}.parquet"
            pd.DataFrame(rows).to_parquet(out, index=False)
            logging.info("saved %s rows=%d", out.name, len(rows))
            time.sleep(RATE_SLEEP)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("snapshot failed")
        raise
