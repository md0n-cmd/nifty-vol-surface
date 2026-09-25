"""Load raw option-chain snapshots from the ./data/date=YYYY-MM-DD/ tree."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATE_DIR_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")


def latest_date_dir(data_dir: Path) -> Path:
    """Return the most recent date=YYYY-MM-DD partition under data_dir."""
    data_dir = Path(data_dir)
    candidates = [p for p in data_dir.iterdir() if p.is_dir() and DATE_DIR_RE.match(p.name)]
    if not candidates:
        raise FileNotFoundError(f"no date= partitions found under {data_dir}")
    return max(candidates, key=lambda p: DATE_DIR_RE.match(p.name).group(1))


def load_date(date_dir: Path) -> pd.DataFrame:
    """Concatenate every *.parquet snapshot under a single date partition."""
    date_dir = Path(date_dir)
    files = sorted(date_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {date_dir}")

    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"])
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date.astype(str)
    return df


def load_latest(data_dir: Path) -> tuple[Path, pd.DataFrame]:
    """Load every snapshot for the latest available date. Returns (date_dir, df)."""
    date_dir = latest_date_dir(data_dir)
    return date_dir, load_date(date_dir)
