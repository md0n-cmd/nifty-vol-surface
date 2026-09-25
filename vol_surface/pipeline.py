"""End-to-end pipeline: load the latest snapshot date, clean, fit SVI per
expiry, and plot fitted vs market IV.

Usage:
    python -m vol_surface.pipeline [--data-dir DATA_DIR] [--out-dir OUT_DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import io as vio
from .clean import clean_chain
from .forward import forwards_by_expiry
from .implied_vol import attach_iv_surface
from .plotting import plot_surface
from .svi import fit_all_expiries


def run(data_dir: Path, out_dir: Path, max_spread_pct: float = 0.15) -> dict:
    date_dir = vio.latest_date_dir(data_dir)
    raw = vio.load_date(date_dir)

    cleaned = clean_chain(raw, max_spread_pct=max_spread_pct)
    forwards = forwards_by_expiry(cleaned)
    surface = attach_iv_surface(cleaned, forwards)
    fits = fit_all_expiries(surface)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    surface.to_parquet(out_dir / f"{date_dir.name}_surface.parquet", index=False)

    params = {
        f"{underlying}_{expiry}": {
            "a": fit.a, "b": fit.b, "rho": fit.rho, "m": fit.m, "sigma": fit.sigma,
            "n_points": fit.n_points, "cost": fit.cost,
        }
        for (underlying, expiry), fit in fits.items()
    }
    (out_dir / f"{date_dir.name}_svi_params.json").write_text(json.dumps(params, indent=2))

    plots = plot_surface(surface, fits, out_dir / "plots")

    return {"date_dir": date_dir, "cleaned": cleaned, "forwards": forwards,
            "surface": surface, "fits": fits, "plots": plots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("vol_surface_out"))
    parser.add_argument("--max-spread-pct", type=float, default=0.15)
    args = parser.parse_args()

    result = run(args.data_dir, args.out_dir, args.max_spread_pct)
    print(f"date: {result['date_dir'].name}")
    print(f"cleaned legs: {len(result['cleaned'])}")
    print(f"fitted expiries: {len(result['fits'])}")
    for (underlying, expiry), fit in result["fits"].items():
        print(f"  {underlying} {expiry}: a={fit.a:.5f} b={fit.b:.4f} "
              f"rho={fit.rho:.3f} m={fit.m:.4f} sigma={fit.sigma:.4f} "
              f"n={fit.n_points} cost={fit.cost:.3e}")
    for path in result["plots"]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
