"""Static per-expiry plots of fitted SVI smiles vs market IV.

Colors are slots 1 (blue) and 2 (orange) of the validated categorical
palette -- a colorblind-safe pair (CVD Delta E 9.1, normal-vision 19.6) --
paired with direct legend labels so identity never rides on hue alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
MARKET_COLOR = "#2a78d6"   # categorical slot 1 (blue)
FIT_COLOR = "#eb6834"      # categorical slot 2 (orange)
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"


def plot_surface(surface_df, fits: dict, out_dir: Path) -> list[Path]:
    """Write one PNG per (underlying, expiry): market IV scatter + SVI fit line.
    Returns the list of paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for (underlying, expiry), grp in surface_df.groupby(["underlying", "expiry"]):
        fit = fits.get((underlying, expiry))
        if fit is None:
            continue
        t = grp["t"].iloc[0]

        fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)

        ax.scatter(grp["k"], grp["iv"] * 100, s=28, color=MARKET_COLOR,
                   label="market IV", zorder=3, edgecolors="none")

        k_grid = np.linspace(grp["k"].min(), grp["k"].max(), 200)
        ax.plot(k_grid, fit.iv(k_grid, t) * 100, color=FIT_COLOR, lw=2,
                label="SVI fit", zorder=2)

        ax.set_title(f"{underlying} {expiry} — SVI fit vs market IV",
                     color=PRIMARY_INK, fontsize=12)
        ax.set_xlabel("log-moneyness  k = ln(K/F)", color=MUTED_INK)
        ax.set_ylabel("implied vol (%)", color=MUTED_INK)
        ax.tick_params(colors=MUTED_INK)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, labelcolor=PRIMARY_INK)

        out_path = out_dir / f"{underlying}_{expiry}_svi_fit.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)

    return written
