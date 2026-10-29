# analysis/

One-off studies of the daily fits. Run everything from the repo root; all paths
are relative (`data/`, `results/`). `data/` and `results/` are git-ignored.
Bootstraps and permutations use fixed seeds, so reruns reproduce the CSVs and
PNGs byte for byte.

## Run order

| # | Command | Inputs | Outputs |
|---|---|---|---|
| 1 | `python daily_fits.py` (~33 min, Heston) | `data/` | `results/daily_fits.csv` |
| 2 | `python svi_refit.py` | `data/`, `results/daily_fits.csv` | `results/svi_refit.csv` |
| 3 | `python skew_check.py` | `data/`, `results/svi_refit.csv` | `results/skew_check.csv` |
| 4 | `python analysis/skew_plots.py` | `results/skew_check.csv` | `results/plots/skew_{market,normalised}_by_date.png` |
| 5 | `python analysis/skew_diagnostics.py` | `results/skew_check.csv`, `data/` | `results/skew_diagnostics.csv` |
| 6 | `python analysis/skew_robustness.py` | `results/skew_check.csv`, `data/` | `results/skew_robustness.csv` |
| 7 | `python analysis/skew_matched_dte.py` | `data/` | `results/skew_matched_dte.csv` |
| 8 | `python analysis/skew_1021_check.py` | `data/` | `results/skew_1021_check.csv` |

Steps 1-3 live at the repo root and are tested modules (`tests/`); steps 4-8
are scripts. Steps 4-8 are independent of each other once step 3 has run
(7 and 8 read only `data/`). Scripts print their summary tables to stdout.

## Notes

- Skew is the OLS slope of market IV on k = ln(K/F) over OTM quotes.
- Raw-SVI rho/b/a are weakly identified on these narrow strike ranges, so the
  studies use model-free market skew as the reference.
- Stale-spot dates (2026-10-02, 2026-10-20) are excluded in `daily_fits.py`.
