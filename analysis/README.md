# analysis/

One-off studies of the daily fits. Run everything from the repo root; all paths
are relative (`data/`, `results/`). `data/` and `results/` are git-ignored.

**The repo cannot be rerun from a fresh clone.** `data/` is not committed, so
every step needs your own collected option chains (`data/date=YYYY-MM-DD/
{NIFTY,BANKNIFTY}_<expiry>_<time>.parquet`, from the collector). The tests use
synthetic chains and do run on a fresh clone.

Bootstraps and permutations use fixed seeds, so reruns reproduce the CSVs and
PNGs byte for byte.

## Run order

| # | Command | Inputs | Outputs |
|---|---|---|---|
| 0 | `python analysis/date_audit.py` | `data/` | `results/date_audit.csv` (flags EMPTY / LOW_TS / STALE per date) |
| 1 | `python daily_fits.py` (~33 min, Heston) | `data/` | `results/daily_fits.csv` |
| 2 | `python svi_refit.py` | `data/`, `results/daily_fits.csv` | `results/svi_refit.csv` |
| 3 | `python skew_check.py` | `data/`, `results/svi_refit.csv` | `results/skew_check.csv` |
| 4 | `python analysis/skew_plots.py` | `results/skew_check.csv` | `results/plots/skew_{market,normalised}_by_date.png` |
| 5 | `python analysis/skew_diagnostics.py` | `results/skew_check.csv`, `data/` | `results/skew_diagnostics.csv` |
| 6 | `python analysis/skew_robustness.py` | `results/skew_check.csv`, `data/` | `results/skew_robustness.csv` |
| 7 | `python analysis/skew_matched_dte.py` | `data/` | `results/skew_matched_dte.csv` |
| 8 | `python analysis/skew_1021_check.py` | `data/` | `results/skew_1021_check.csv` |
| 9 | `python analysis/mc_validation.py` (fixed seed; recalibration check is skipped without `data/`) | optional `data/date=2026-09-25/` | `docs/results/mc_validation.csv` |
| 10 | `python analysis/heston_provenance.py` (BANKNIFTY 10-27 on 2026-09-25, 11:00 to 11:27) | `data/date=2026-09-25/`, `results/date=2026-09-25_svi_params.json` | `docs/results/heston_fragility.csv`, `docs/results/provenance_parity_svi.csv` |
| 11 | `python analysis/barrier_check.py` (20 seeds, several minutes; run after step 10) | `data/date=2026-09-25/` (the 11:06:02 BANKNIFTY 10-27 file), `docs/results/heston_fragility.csv` | `docs/results/barrier_check.csv`, `docs/results/param_dependence.csv` |
| 12 | `python analysis/bound_recount.py` | `docs/results/daily_fits.csv` (a copy of step 1's `results/daily_fits.csv`) | `docs/results/bound_recount.csv` (and stdout) |
| 13 | `python analysis/post_rmse_check.py` | `data/date=2026-09-25/` (the 11:06:02 BANKNIFTY 09-29 and 10-27 files) | `docs/results/post_rmse_check.csv` |

Steps 1-3 live at the repo root and are tested modules (`tests/`); steps 4-8
are scripts. Steps 4-8 are independent of each other once step 3 has run
(7 and 8 read only `data/`). Scripts print their summary tables to stdout.

Steps 9-13 write to `docs/results/`. Run order for the last three: step 11
(`barrier_check.py`) reads `docs/results/heston_fragility.csv`, so it must come after
step 10. Which of them need `data/`: steps 11 and 13 do (`barrier_check.py` loads the
11:06:02 snapshot for its objective comparison, `post_rmse_check.py` calibrates on the
snapshots); step 12 (`bound_recount.py`) runs from the committed CSV alone. Step 9 runs
without `data/` but skips its recalibration check.

## Notes

- Skew is the OLS slope of market IV on k = ln(K/F) over OTM quotes.
- Raw-SVI rho/b/a are weakly identified on these narrow strike ranges, so the
  studies use model-free market skew as the reference.
- Stale-spot dates (2026-10-02, 2026-10-20) are hard-coded as `STALE_DATES` in
  `daily_fits.py`, taken from step 0's STALE flag (spot frozen across all 125
  snapshots). Re-run step 0 after collecting new data and update that set if
  new STALE days appear.
