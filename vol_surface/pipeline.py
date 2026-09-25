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
from .calendar import check_calendar_arbitrage
from .clean import clean_chain
from .forward import forwards_by_expiry
from .implied_vol import attach_iv_surface
from .otm import select_otm
from .plotting import plot_surface
from .svi import fit_all_expiries


def run(data_dir: Path, out_dir: Path, max_spread_pct: float = 0.15,
        min_mid: float = 0.5, min_oi: int = 0, robust_svi: bool = False,
        otm_only: bool = True) -> dict:
    date_dir = vio.latest_date_dir(data_dir)
    raw = vio.load_date(date_dir)

    cleaned = clean_chain(raw, max_spread_pct=max_spread_pct, min_mid=min_mid, min_oi=min_oi)
    forwards = forwards_by_expiry(cleaned)
    # forward extraction needs both sides of the chain (parity uses C-P at
    # every strike), so OTM selection happens after, using the forward it
    # just produced -- not before.
    fit_input = select_otm(cleaned, forwards) if otm_only else cleaned
    surface = attach_iv_surface(fit_input, forwards)
    fits = fit_all_expiries(surface, robust=robust_svi)

    t_by_key = {(row.underlying, row.expiry): row.t for row in forwards.itertuples()}
    calendar_violations = check_calendar_arbitrage(fits, t_by_key)

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

    return {"date_dir": date_dir, "raw_legs": len(raw), "cleaned": cleaned,
            "forwards": forwards, "surface": surface, "fits": fits, "plots": plots,
            "calendar_violations": calendar_violations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("vol_surface_out"))
    parser.add_argument("--max-spread-pct", type=float, default=0.15)
    parser.add_argument("--min-mid", type=float, default=0.5)
    parser.add_argument("--min-oi", type=int, default=0)
    parser.add_argument("--robust-svi", action="store_true",
                         help="two-pass SVI fit: drop >3 MAD outliers in w-space, refit")
    parser.add_argument("--no-otm-filter", action="store_false", dest="otm_only",
                         default=True,
                         help="fit on both ITM and OTM legs instead of OTM-only (default: OTM-only)")
    args = parser.parse_args()

    result = run(args.data_dir, args.out_dir, args.max_spread_pct, args.min_mid,
                 args.min_oi, args.robust_svi, args.otm_only)
    dropped = result["raw_legs"] - len(result["cleaned"])
    print(f"date: {result['date_dir'].name}")
    print(f"raw legs: {result['raw_legs']}, cleaned: {len(result['cleaned'])}, dropped: {dropped}")
    print(f"fitted expiries: {len(result['fits'])}")
    for (underlying, expiry), fit in result["fits"].items():
        print(f"  {underlying} {expiry}: a={fit.a:.5f} b={fit.b:.4f} "
              f"rho={fit.rho:.3f} m={fit.m:.4f} sigma={fit.sigma:.4f} "
              f"n={fit.n_points} cost={fit.cost:.3e}")
    violations = result["calendar_violations"]
    if violations:
        print(f"calendar no-arbitrage check: FAILED ({len(violations)} violation points)")
        for v in violations[:5]:
            print(f"  {v['underlying']} k={v['k']:.4f}: w({v['expiry_near']})={v['w_near']:.5f} "
                  f"> w({v['expiry_far']})={v['w_far']:.5f}")
    else:
        print("calendar no-arbitrage check: passed (total variance non-decreasing across expiries)")
    for path in result["plots"]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
