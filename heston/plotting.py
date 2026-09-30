"""Static charts comparing raw market IV, an SVI fit, and a calibrated
Heston-implied IV curve. Reuses vol_surface.plotting's palette/style tokens
for visual consistency across the project.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
MARKET_COLOR = "#2a78d6"   # categorical slot 1 (blue)
SVI_COLOR = "#eb6834"      # categorical slot 2 (orange)
HESTON_COLOR = "#1baf7a"   # categorical slot 3 (aqua)
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"


def _draw_panel(ax, surface_df, svi_fit, heston_fit, forward, t, discount):
    k = surface_df["k"].to_numpy()
    strikes = surface_df["strike"].to_numpy()
    types = surface_df["type"].to_numpy()

    ax.set_facecolor(SURFACE)
    ax.scatter(k, surface_df["iv"].to_numpy() * 100, s=22, color=MARKET_COLOR,
               label="market IV", zorder=3, edgecolors="none")

    if svi_fit is not None:
        k_grid = np.linspace(k.min(), k.max(), 200)
        ax.plot(k_grid, svi_fit.iv(k_grid, t) * 100, color=SVI_COLOR, lw=2,
                label="SVI fit", zorder=2)

    if heston_fit is not None:
        order = np.argsort(k)
        heston_iv = heston_fit.implied_vol_curve(forward, strikes, t, discount, types)
        ax.plot(k[order], heston_iv[order] * 100, color=HESTON_COLOR, lw=2,
                label="Heston", zorder=2, marker="o", ms=2.5)

    ax.tick_params(colors=MUTED_INK, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_svi_vs_heston(surface_df, svi_fit, heston_fit, forward, t, discount,
                        title: str, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    _draw_panel(ax, surface_df, svi_fit, heston_fit, forward, t, discount)
    ax.set_title(title, color=PRIMARY_INK, fontsize=12)
    ax.set_xlabel("log-moneyness  k = ln(K/F)", color=MUTED_INK)
    ax.set_ylabel("implied vol (%)", color=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_svi_vs_heston_grid(slices: list, heston_fit, out_path: Path,
                             suptitle: str = "") -> Path:
    """slices: list of dicts with keys surface_df, svi_fit, forward, t,
    discount, title -- one per (underlying, expiry) panel. heston_fit is
    shared across all panels (e.g. a calibrate_heston_joint result)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(slices)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), facecolor=SURFACE)
    if n == 1:
        axes = [axes]

    for ax, s in zip(axes, slices):
        _draw_panel(ax, s["surface_df"], s["svi_fit"], heston_fit,
                    s["forward"], s["t"], s["discount"])
        ax.set_title(s["title"], color=PRIMARY_INK, fontsize=11)
        ax.set_xlabel("k = ln(K/F)", color=MUTED_INK, fontsize=9)

    axes[0].set_ylabel("implied vol (%)", color=MUTED_INK)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               labelcolor=PRIMARY_INK, bbox_to_anchor=(0.5, 1.08))
    if suptitle:
        fig.suptitle(suptitle, color=PRIMARY_INK, fontsize=13, y=1.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
