"""Figures: market skew and skew*sqrt(dte/365) by date, with CIs. In: results/skew_check.csv. Out: results/plots/skew_{market,normalised}_by_date.png (+ stats on stdout).

Run from the repo root: python analysis/skew_plots.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COL = {"near": "#2a78d6", "next": "#eb6834"}
d = pd.read_csv("results/skew_check.csv")
d["norm"] = d.skew_market * np.sqrt(d.dte / 365)
d["n_lo"] = d.ci_lo * np.sqrt(d.dte / 365)
d["n_hi"] = d.ci_hi * np.sqrt(d.dte / 365)
dates = sorted(d.date.unique()); xpos = {x: i for i, x in enumerate(dates)}
d["x"] = d.date.map(xpos)

def fig(ycol, lo, hi, title, ylabel, path):
    f, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False, facecolor=BG)
    for ax, u in zip(axes, ["NIFTY", "BANKNIFTY"]):
        ax.set_facecolor(BG)
        for b in ["near", "next"]:
            s = d[(d.underlying == u) & (d.bucket == b)].sort_values("x")
            off = -0.08 if b == "near" else 0.08
            x = s.x + off
            ax.plot(x, s[ycol], color=COL[b], lw=2, zorder=2)
            ax.errorbar(x, s[ycol], yerr=[s[ycol] - s[lo], s[hi] - s[ycol]], fmt="none",
                        ecolor=COL[b], elinewidth=1.4, capsize=3, zorder=3)
            few = s.n_quotes_in_window < 8
            ax.scatter(x[~few], s[ycol][~few], s=46, color=COL[b], edgecolor=BG, linewidth=1.5, zorder=4)
            ax.scatter(x[few], s[ycol][few], s=46, facecolor=BG, edgecolor=COL[b], linewidth=1.8, zorder=4)
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        ax.set_title(u, loc="left", color=INK, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(dates))); ax.set_xticklabels([x[5:] for x in dates], rotation=45, color=INK2, fontsize=9)
        ax.tick_params(axis="y", colors=INK2); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]: ax.spines[sp].set_color(GRID)
        ax.set_ylabel(ylabel, color=INK2)
    h = [plt.Line2D([], [], color=COL[b], lw=2, marker="o", ms=7, mec=BG, label=b) for b in COL]
    h.append(plt.Line2D([], [], color=INK2, marker="o", ms=7, mfc=BG, mec=INK2, ls="", label="< 8 quotes in window"))
    axes[0].legend(handles=h, frameon=False, labelcolor=INK2, fontsize=9, loc="lower left")
    f.suptitle(title, color=INK, x=0.01, ha="left", fontsize=13, fontweight="bold")
    f.text(0.01, 0.005, "Error bars: bootstrap 95% CI (500 draws). Dates are trading days with a fit (equally spaced).", color=INK2, fontsize=8.5)
    f.tight_layout(rect=(0, 0.03, 1, 0.95)); f.savefig(path, dpi=150, facecolor=BG); plt.close(f)

fig("skew_market", "ci_lo", "ci_hi", "Market ATM skew dIV/dk by date", "dIV/dk", "results/plots/skew_market_by_date.png")
fig("norm", "n_lo", "n_hi", "Normalised market skew (skew × √(DTE/365)) by date", "skew × √(DTE/365)", "results/plots/skew_normalised_by_date.png")

pd.set_option("display.width", 250)
rows = []
for (u, b), s in d.groupby(["underlying", "bucket"]):
    s = s.sort_values("x")
    sl = stats.linregress(s.x, s.norm); rho = stats.spearmanr(s.x, s.norm)
    touch = s[(s.n_lo <= 0) & (s.n_hi >= 0)]
    rows.append(dict(u=u, b=b, n=len(s), min=s.norm.min(), med=s.norm.median(), max=s.norm.max(),
                     slope_per_day=sl.slope, p=sl.pvalue, spearman=rho.statistic, sp_p=rho.pvalue,
                     first3=s.norm.iloc[:3].mean(), last3=s.norm.iloc[-3:].mean(), sd=s.norm.std(),
                     touch=list(touch.date)))
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(d.groupby(["underlying","bucket"]).apply(lambda s: s.sort_values("x").norm.round(3).tolist()).to_string())
