"""Market ATM skew over time from docs/results/skew_check.csv only.

skew_market against date with bootstrap 95% CI error bars (ci_lo, ci_hi), one panel per
underlying (same y-range on both), near and next shown as separate markers with no
connecting lines. NIFTY points are labelled with their days to expiry (dte), placed so no
label touches a marker, an error bar or another label; the BANKNIFTY panel is not labelled
per point and gives the dte ranges of near and next in its title. Slices with fewer than 8
quotes in the skew window are hollow markers; 2026-10-21 is marked on the NIFTY panel.

In: docs/results/skew_check.csv. Out: docs/results/skew_over_time.png.
Run from the repo root: python analysis/skew_plot.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC, OUT = Path("docs/results/skew_check.csv"), Path("docs/results/skew_over_time.png")
MIN_QUOTES, MARK_DATE = 8, "2026-10-21"
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLOR = {"near": "#2a78d6", "next": "#eb6834"}
OFFSET = {"near": -0.16, "next": 0.16}
FS_TICK, FS_AXIS, FS_LEGEND, FS_CAPTION, FS_POINT = 12, 13, 12, 12, 9
CAPTION = ("Error bars are bootstrap 95% CIs and capture quote scatter only. Skew window: |k| ≤ 0.5·ATM_IV·√T. "
           "Points are labelled with days to expiry; the points mix different expiries and maturities, "
           "so they are not a clean time series. 12 dates, equally spaced. "
           "BANKNIFTY points are not individually labelled; their DTE ranges are in the panel title.")
# label offsets to try, in points: (dx, dy, ha, va)
CANDIDATES = [(dx, dy, ha, va) for r in (11, 13, 15, 18, 21, 25, 30) for dx, dy, ha, va in
              [(r, 0, "left", "center"), (-r, 0, "right", "center"), (r, r, "left", "bottom"), (-r, r, "right", "bottom"),
               (r, -r, "left", "top"), (-r, -r, "right", "top"), (0, r, "center", "bottom"), (0, -r, "center", "top"),
               (r, r // 2, "left", "center"), (-r, r // 2, "right", "center"),
               (r, -(r // 2), "left", "center"), (-r, -(r // 2), "right", "center")]]


def _leader(dx, dy, color):
    """Thin leader line from label to marker when the label sits farther than ~18 pt away."""
    if (dx * dx + dy * dy) ** 0.5 <= 18:
        return None
    return dict(arrowstyle="-", color=color, lw=0.7, shrinkA=1, shrinkB=6)


def place_labels(fig, ax, points, extra_obstacles):
    """points: list of (x, y, ylo, yhi, text, color) in data coordinates. Put each label at the first
    candidate offset whose box overlaps no marker, error bar, cap, earlier label or other obstacle;
    if none is free, keep the candidate with the smallest overlap area."""
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    obstacles = list(extra_obstacles)
    for x, y, lo, hi, _, _ in points:
        X, Y = ax.transData.transform((x, y)); _, Ylo = ax.transData.transform((x, lo)); _, Yhi = ax.transData.transform((x, hi))
        r = 6.0 * px / 1.8 * 1.0 + 3
        obstacles.append((X - r, Y - r, X + r, Y + r))                                  # marker
        obstacles.append((X - 2, min(Ylo, Yhi) - 2, X + 2, max(Ylo, Yhi) + 2))           # error bar
        for Yc in (Ylo, Yhi):
            obstacles.append((X - 3.5 * px - 1, Yc - 2, X + 3.5 * px + 1, Yc + 2))      # caps
    for x, y, _, _, text, color in points:
        best, best_area = None, None
        for dx, dy, ha, va in CANDIDATES:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=FS_POINT, color=color, zorder=5, arrowprops=_leader(dx, dy, color))
            bb = t.get_window_extent(renderer); box = (bb.x0, bb.y0, bb.x1, bb.y1)
            ab = ax.get_window_extent(renderer)
            inside = box[0] >= ab.x0 and box[2] <= ab.x1 and box[1] >= ab.y0 and box[3] <= ab.y1
            area = sum(max(0, min(box[2], o[2] + 1.5) - max(box[0], o[0] - 1.5)) * max(0, min(box[3], o[3] + 1.5) - max(box[1], o[1] - 1.5))
                       for o in obstacles) + (0 if inside else 1e6)
            if area == 0:
                obstacles.append(box); best = None; break
            if best_area is None or area < best_area:
                best_area, best = area, (dx, dy, ha, va)
            t.remove()
        if best is not None:                       # no free spot: keep the least-overlapping candidate
            dx, dy, ha, va = best
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=FS_POINT, color=color, zorder=5, arrowprops=_leader(dx, dy, color))
            bb = t.get_window_extent(renderer); obstacles.append((bb.x0, bb.y0, bb.x1, bb.y1))
            print(f"  note: label {text} at ({x:.2f}, {y:.2f}) overlaps (area {best_area:.0f} px^2)")


def main():
    d = pd.read_csv(SRC)
    dates = sorted(d["date"].unique())
    pos = {x: i for i, x in enumerate(dates)}
    ylo, yhi = d["ci_lo"].min() - 0.08, max(d["ci_hi"].max(), 0.0) + 0.12   # one range for both panels
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.0), facecolor=BG, sharey=True)
    label_points = []
    for ax, u in zip(axes, ["NIFTY", "BANKNIFTY"]):
        ax.set_facecolor(BG)
        for b in ["near", "next"]:
            s = d[(d["underlying"] == u) & (d["bucket"] == b)].sort_values("date")
            x = s["date"].map(pos) + OFFSET[b]
            y = s["skew_market"]
            ax.errorbar(x, y, yerr=[y - s["ci_lo"], s["ci_hi"] - y], fmt="none", ecolor=COLOR[b],
                        elinewidth=1.6, capsize=3.5, zorder=3)
            few = s["n_quotes_in_window"] < MIN_QUOTES
            ax.scatter(x[~few], y[~few], s=70, color=COLOR[b], edgecolor=BG, linewidth=1.5, zorder=4)
            ax.scatter(x[few], y[few], s=70, facecolor=BG, edgecolor=COLOR[b], linewidth=2.0, zorder=4)
            if u == "NIFTY":
                for xi, yi, lo, hi, n in zip(x, y, s["ci_lo"], s["ci_hi"], s["dte"]):
                    label_points.append((xi, yi, lo, hi, f"{int(n)}d", COLOR[b]))
        ax.axhline(0, color=INK2, lw=0.9, zorder=1)
        if u == "NIFTY" and MARK_DATE in pos:
            ax.axvline(pos[MARK_DATE], color=INK2, lw=1, ls="--", zorder=1)
            ax.annotate("10-21", (pos[MARK_DATE], 0.02), xycoords=("data", "axes fraction"), xytext=(-5, 0),
                        textcoords="offset points", ha="right", va="bottom", color=INK2, fontsize=FS_TICK)
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(-0.7, len(dates) + 0.2)
        if u == "BANKNIFTY":
            r = d[d["underlying"] == u].groupby("bucket")["dte"].agg(["min", "max"])
            rng = ", ".join(f"{b}: {int(r.loc[b, 'min'])}–{int(r.loc[b, 'max'])} d" for b in ["near", "next"])
            ax.set_title(f"{u}  ({rng})", loc="left", color=INK, fontsize=15, fontweight="bold")
        else:
            ax.set_title(u, loc="left", color=INK, fontsize=15, fontweight="bold")
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels([x[5:] for x in dates], rotation=45, color=INK2, fontsize=FS_TICK)
        ax.set_xlabel("Snapshot date (2026, MM-DD)", color=INK2, fontsize=FS_AXIS)
        ax.tick_params(axis="y", colors=INK2, labelsize=FS_TICK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(GRID)
    axes[0].set_ylabel("Market skew, dIV/dk at k = 0", color=INK2, fontsize=FS_AXIS)
    handles = [plt.Line2D([], [], color=COLOR[b], marker="o", ms=9, mec=BG, ls="", label=b) for b in COLOR]
    handles.append(plt.Line2D([], [], color=INK2, marker="o", ms=9, mfc=BG, mec=INK2, ls="",
                              label=f"< {MIN_QUOTES} quotes in window"))
    axes[0].legend(handles=handles, frameon=False, labelcolor=INK2, fontsize=FS_LEGEND, loc="lower left")
    fig.suptitle("Market-implied ATM skew, 14:30 snapshot", color=INK, x=0.01, ha="left",
                 fontsize=18, fontweight="bold")
    fig.text(0.01, 0.012, CAPTION.replace(" Points are labelled", "\nPoints are labelled")
             .replace(" BANKNIFTY points", "\nBANKNIFTY points"), color=INK2,
             fontsize=FS_CAPTION, va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.canvas.draw()
    ax0 = axes[0]; renderer = fig.canvas.get_renderer(); extra = []
    lb = ax0.get_legend().get_window_extent(renderer); extra.append((lb.x0, lb.y0, lb.x1, lb.y1))
    for t in ax0.texts:                                        # the 10-21 annotation
        bb = t.get_window_extent(renderer); extra.append((bb.x0, bb.y0, bb.x1, bb.y1))
    zx0, zy = ax0.transData.transform((-0.7, 0)); zx1, _ = ax0.transData.transform((len(dates) + 0.2, 0))
    extra.append((zx0, zy - 1.5, zx1, zy + 1.5))               # zero line
    vx, vy0 = ax0.transData.transform((pos[MARK_DATE], ylo)); _, vy1 = ax0.transData.transform((pos[MARK_DATE], yhi))
    extra.append((vx - 1.5, vy0, vx + 1.5, vy1))               # dashed 10-21 line
    place_labels(fig, ax0, label_points, extra)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, facecolor=BG)
    print(OUT)


if __name__ == "__main__":
    main()
