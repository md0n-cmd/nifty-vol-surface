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


def plot_svi_params(fits: dict, out_path: Path) -> Path:
    """Small multiples comparing a, b, rho, m, sigma across every fitted
    (underlying, expiry) slice, colored by underlying identity -- a
    calibration sanity check, e.g. to spot one expiry's rho sign or
    magnitude diverging from the rest.
    """
    from .svi import PARAM_NAMES

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    expiries = sorted({expiry for _, expiry in fits})
    underlyings = sorted({underlying for underlying, _ in fits})
    colors = {underlyings[0]: MARKET_COLOR}
    if len(underlyings) > 1:
        colors[underlyings[1]] = FIT_COLOR

    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(4 * len(PARAM_NAMES), 4.5),
                              facecolor=SURFACE)
    x_pos = {expiry: i for i, expiry in enumerate(expiries)}

    for ax, param in zip(axes, PARAM_NAMES):
        ax.set_facecolor(SURFACE)
        for underlying in underlyings:
            xs, ys = [], []
            for expiry in expiries:
                fit = fits.get((underlying, expiry))
                if fit is None:
                    continue
                xs.append(x_pos[expiry])
                ys.append(getattr(fit, param))
            ax.plot(xs, ys, marker="o", ms=7, lw=1.5, color=colors[underlying],
                    label=underlying, zorder=3)

        if param == "rho":
            ax.axhline(0.0, color=AXIS_COLOR, lw=1, linestyle="--", zorder=1)

        ax.set_title(param, color=PRIMARY_INK, fontsize=12)
        ax.set_xticks(range(len(expiries)))
        ax.set_xticklabels(expiries, rotation=30, ha="right", color=MUTED_INK, fontsize=8)
        ax.tick_params(colors=MUTED_INK)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS_COLOR)
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(underlyings),
               frameon=False, labelcolor=PRIMARY_INK, bbox_to_anchor=(0.5, 1.06))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
