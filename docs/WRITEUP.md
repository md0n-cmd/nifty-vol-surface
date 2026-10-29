# NSE index-option vol surfaces: write-up

Source tags in `[brackets]` name the file (or commit) each figure comes from.
Aggregate CSVs and the two figures the text cites are copied to `docs/results/`.
Those copies are snapshots of the git-ignored `results/` outputs, so tags like
`docs/results/skew_check.csv` can be checked without rerunning anything. Figures
from the intraday check (`analysis/skew_1021_check.py`, CSV not copied) and from
`data/collector.log` (local, not in the repo) are given in the text only. "Tests"
means the files under `tests/`; code constants are cited by file.

## 1. Question and data

**Question.** What do NSE NIFTY and BANKNIFTY option chains imply about the
implied-vol surface (raw SVI per expiry), and can a Heston model fitted to the same
slices be used for Monte Carlo pricing of an exotic? This is a one-month research
project, not a trading study.

**Collector.** `collector.py` polls DhanHQ v2 option-chain endpoints for NIFTY
and BANKNIFTY, the two nearest expiries each (`N_EXPIRIES = 2`), and writes one
Parquet file per (underlying, expiry, snapshot) with bid/ask/ltp/oi/volume/iv and
greeks [`collector.py`]. It runs from cron on a VPS during market hours; chains
are rsynced down [`CLAUDE.md`]. A full day has 125 distinct snapshot timestamps,
roughly every 3 minutes from 09:15 to 15:27 IST [`docs/results/date_audit.csv`].

**Coverage.** 26 date folders, 16 with data, 10 empty
[`docs/results/date_audit.csv`]. Of the 16, two are stale (below) and six are
partial days (under 120 timestamps).

**Outages and their causes.** From `data/collector.log`. **The log is in UTC; every
time in the table below is IST (UTC+5:30)**, converted from the log. As a check, the
first 09-24 snapshot is logged at 05:18:25 UTC, which is 10:48:25 IST, against a
filename time of 10:48:21. "Error" means a `snapshot failed` line. Every long gap from
09-25 on is one of two API responses, both HTTP 401:

- **808**, "Authentication Failed - Client ID or Token invalid"
- **806**, "Data APIs not Subscribed"

| Date | Data kept (IST) | First → last error (IST) | Cause |
|---|---|---|---|
| 09-24 | from 10:48 (78 timestamps) | none logged | setup day: first manual run 10:48, cron schedule fixed about 11:39 (errors in `cron_debug.log` on the VPS, not in `collector.log`) |
| 09-25 | 09:15 → 11:30 (46) | 11:30:28 → 15:27:21 | 808 |
| 09-28 | empty | 09:15:21 → 15:27:22 | 808 |
| 09-29 | empty | 09:15:22 → 15:27:21 | 808 |
| 09-30 | from 10:59 (91) | 09:15:22 → 10:57:32 | 808 |
| 10-07 | empty | 09:15:20 → 15:27:21 | 808 |
| 10-08 | empty | 09:15:21 → 15:27:27 | 808 |
| 10-09 | from 14:03 (29) | 09:15:22 → 14:00:23 | 808 |
| 10-13 | empty | 09:15:23 → 15:27:23 | 808 |
| 10-21 | from 11:45 (75) | 09:15:22 → 11:42:22 | 806 |
| 10-22 | 09:15 → 12:30 (66) | 12:30:28 → 15:27:21 | 806 |
| 10-23 | empty | 09:15:22 → 15:27:21 | 806 |
| 10-26 | empty | 09:15:22 → 15:27:22 | 806 |
| 10-27 | empty | 09:15:22 → 15:27:26 | 806 |
| 10-28 | empty | 09:15:21 → 15:27:21 | 806 |
| 10-29 | empty | 09:15:22 → 11:03:22 (log ends 11:06) | 806 |

The data-kept column is from `docs/results/date_audit.csv`. Why the credentials went
invalid, and why the data subscription lapsed, is unknown: the log only records the
API's response. On 10-21 the first good snapshot is 11:45:09 IST, three minutes after
the last error. The raw 10-21 log lines, **in UTC as logged** (03:45 UTC = 09:15 IST,
06:12:22 UTC = 11:42:22 IST, 06:15:09 UTC = 11:45:09 IST): the first lines of the day,
then the last three warnings, the last error with its traceback, and the first saved
snapshot after it.

````
2026-10-21 03:45:03,305 WARNING /optionchain/expirylist 401 {"data":{"806":"Data APIs not Subscribed"},"status":"failed"}
2026-10-21 03:45:06,690 WARNING /optionchain/expirylist 401 {"data":{"806":"Data APIs not Subscribed"},"status":"failed"}
2026-10-21 03:45:13,515 WARNING /optionchain/expirylist 401 {"data":{"806":"Data APIs not Subscribed"},"status":"failed"}
...
2026-10-21 06:12:03,334 WARNING /optionchain/expirylist 401 {"data":{"806":"Data APIs not Subscribed"},"status":"failed"}
2026-10-21 06:12:06,575 WARNING /optionchain/expirylist 401 {"data":{"806":"Data APIs not Subscribed"},"status":"failed"}
2026-10-21 06:12:13,678 WARNING /optionchain/expirylist 401 {"data":{"806":"Data APIs not Subscribed"},"status":"failed"}
2026-10-21 06:12:22,979 ERROR snapshot failed
Traceback (most recent call last):
  File "/root/optquant/collector.py", line 92, in <module>
    main()
  File "/root/optquant/collector.py", line 79, in main
    expiries = post("/optionchain/expirylist", base)["data"][:N_EXPIRIES]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/optquant/collector.py", line 39, in post
    raise RuntimeError(f"failed {path} {body}")
RuntimeError: failed /optionchain/expirylist {'UnderlyingScrip': 13, 'UnderlyingSeg': 'IDX_I'}
2026-10-21 06:15:09,618 INFO saved NIFTY_2026-10-27_114502.parquet rows=522
````

**Stale days.** On 2026-10-02 and 2026-10-20 the spot is identical across all 125
snapshots for both underlyings (1 distinct spot value each), though chains were
collected on schedule [`docs/results/date_audit.csv`, `analysis/date_audit.py`].
They are presumed exchange holidays, not verified against NSE circulars, and
are excluded as stale chains. They are the hard-coded `STALE_DATES` in
`daily_fits.py` (commit 4d97924).

**14:30 snapshot design.** For each (date, underlying, expiry) take the snapshot
nearest 14:30 IST; skip a date if none is within 15 minutes; label the nearest
expiry "near" and the second "next"; drop slices with DTE < 2 [`daily_fits.py`,
commit 1490f37].

- 16 non-empty dates minus 2 stale = 14 usable.
- 09-25 (last snapshot 11:30) and 10-22 (last 12:30) have no snapshot within 15
  minutes, so 12 dates are fitted [`docs/results/date_audit.csv`,
  `docs/results/daily_fits.csv`].
- 12 dates × 4 slices − 4 dropped = 44 slices: 20 NIFTY and 24 BANKNIFTY
  [`docs/results/daily_fits.csv`]. The four dropped slices are NIFTY "near" on
  10-05, 10-06, 10-12 and 10-19 (DTE below 2; absent from
  `docs/results/daily_fits.csv`, rule in `daily_fits.py`).

## 2. Surface fitting

**Cleaning** [`vol_surface/clean.py`]: keep the latest snapshot per
(underlying, expiry); require a positive two-sided quote with ask ≥ bid and live
open interest; require relative spread (ask − bid)/mid ≤ 0.15 and mid ≥ 0.5.

**Forward.** For each expiry, regress C − P on strike over strikes quoted on both
sides: discount D = −slope, forward F = intercept/D, so no external rate is needed.
It needs at least 3 paired strikes. Time to expiry is ACT/365 to 15:30 IST
[`vol_surface/forward.py`]. IVs come from Black-76 with that explicit D, inverted
by Brent's method [`vol_surface/implied_vol.py`].

**Why OTM-only.** Deep in-the-money options price almost entirely off intrinsic
value, so ordinary quote noise becomes a large error in the inverted implied vol.
Commit 2d2a857 gives this as the motivation: "Deep-ITM legs price near intrinsic, so
ordinary quoting noise was blowing up their inverted IV" [commit 2d2a857;
`vol_surface/otm.py`]. So each strike keeps only its OTM side (call at or above the
forward, put at or below), the standard convention. How much including the ITM legs
changes the fitted rho was looked at during development, but that diagnostic's output
was not saved to a file, so no figure from it is used here, and rho changes are not
used as evidence anywhere in this section.

**SVI.** Raw SVI, fitted independently per expiry:
w(k) = a + b(ρ(k−m) + √((k−m)²+σ²)), by bounded least squares with soft penalties
for min w ≥ 0 and Lee's wing bound b(1+|ρ|) ≤ 4 [`vol_surface/svi.py`]. The box
bound on `a` was a hard `a ≥ 0` (`A_LOWER = 0.0`); an `a_lower` option was added in
commit 1490f37 (default unchanged).

**Finding: raw-SVI rho is weakly identified on these slices.**

- With the hard `a ≥ 0` bound, `a` sits at the bound on 43 of 44 slices
  [`docs/results/svi_refit.csv`, `bound_hit_old`].
- Loosening it to −0.05 lowers RMSE on all 44 slices (mean 0.0093 → 0.0023, IV
  units), but moves rho and `b` a lot: mean rho −0.083 → +0.148, mean `b`
  0.021 → 0.089, and on BANKNIFTY "next" rho_new reaches +0.866
  [`docs/results/svi_refit.csv`]. Two slices (BANKNIFTY next, 10-15 and 10-16) still
  pin `a` at −0.05.
- A better RMSE therefore does not show the parameters are right. Raw rho should
  not be read as a skew measure here; model-free market skew is used instead
  (section 3). The commit message records this [commit 1490f37].

## 3. Market skew

Market skew is the OLS slope of market IV on k = ln(K/F) over OTM quotes, with a
bootstrap 95% CI resampling quotes [`skew_check.py`]. The default window is
|k| ≤ 0.5·ATM_IV·√T. Four claims are made, and nothing stronger.

1. **Market-implied ATM skew was negative on all 44 slices.** `skew_market` runs
   from −1.450 to −0.133, and all 44 bootstrap CIs lie below zero
   [`docs/results/skew_check.csv`]. The sign does not depend on the window: all 44
   slices are negative under each of three windows (sized from the raw ATM IV,
   fixed |k| ≤ 0.02, fixed |k| ≤ 0.04) [`docs/results/skew_robustness.csv`, slice
   section]. The bootstrap CI is entirely below zero on 44, 43 and 43 of them
   respectively; the one slice whose CI touches zero under the two wider windows is
   **NIFTY "near" on 2026-10-21 (DTE 6)**: skew −0.083 with CI [−0.248, +0.065] at
   |k| ≤ 0.02, and −0.034 with CI [−0.237, +0.134] at |k| ≤ 0.04 [same file]. CIs
   reflect quote-to-quote scatter only.
2. **BANKNIFTY's far ("next") expiry held a skew of about −0.40 to −0.50** (market
   skew dIV/dk: min −0.501, median −0.459, max −0.400) on all 12 dates, across
   DTE 33–54 [`docs/results/skew_check.csv`]. *Footnote:* scaled as
   skew × √(DTE/365) the same series is −0.176 to −0.137, but that √T scaling
   over-corrects BANKNIFTY (its fitted DTE exponent is −0.07, CI −0.19 to +0.06,
   not −0.5 [`docs/results/skew_diagnostics.csv`]), so the raw range is the one to
   read.
3. **Among the six NIFTY "next" slices at 12–13 DTE, skew on 10-21 was much flatter
   than on the other five, cause unknown.** The |k| ≤ 0.02 skew is −0.153 on 10-21
   against −0.401 to −0.526 on the others, and at |k| ≤ 0.04 it is −0.194 against
   −0.377 to −0.481 [`docs/results/skew_matched_dte.csv`]. On the default window, the NIFTY "next" slices on 10-16 (DTE 11) and
   10-19 (DTE 8; 7 quotes in the window, fewer than 8) are also flatter than the early dates, with market skew −0.238 (CI −0.331 to
   −0.091) and −0.182 (CI −0.207 to −0.144), against −0.480, −0.447 and −0.568 on 09-24, 09-30 and 10-01
   [`docs/results/skew_check.csv`]; but they are outside the 12–13 DTE matched set, they were not tested separately, and no
   drift is claimed. **This claim rests on
   10-21 alone**, a date on which the collector log shows subscription errors (806)
   until 11:42 and the first good snapshot at 11:45 (section 1). The 14:30 snapshot
   is after collection resumed, and I cannot say whether the lapse is related.
   Intraday on 10-21 (12:00 to 15:27, 70 snapshots) the skew ranges −0.331 to
   −0.140, with −0.147, −0.153 and −0.140 at 14:27, 14:30 and 14:33
   [`analysis/skew_1021_check.py` output, figures in text only]. **Single-snapshot
   skew varies by about 0.1 within a day**: standard deviation 0.050 (10-21), 0.053
   (10-14) and 0.107 (10-15) [same output]. 10-21 shows no sign of stale quotes: 0%
   of in-window legs are unchanged between consecutive snapshots, against a 5.2%
   mean across all live legs [same output].
4. **The `a ≥ 0` raw-SVI fit's rho and ATM IV are unreliable on these slices**
   (nothing is claimed about any other SVI fit's rho). Rho: `a` is pinned at 0 on
   43 of 44 slices, and the fitted rho moves a lot when the bound is loosened
   [`docs/results/svi_refit.csv`]; the `a ≥ 0` SVI skew ranges from −2.19 to +3.77
   and is positive on 11 slices, while market skew is negative everywhere, and it
   lies inside the market CI on 6 of 44 slices [`docs/results/skew_check.csv`].
   ATM IV: the `a ≥ 0` fit's ATM IV differs from the raw ATM IV (median market IV of
   OTM quotes with |k| ≤ 0.01) by more than 2 vol points on 8 of 44 slices, 6 of 20
   NIFTY (all between 10-06 and 10-19) and 2 of 24 BANKNIFTY (09-24 and 10-21)
   [`docs/results/skew_robustness.csv`, slice section].

## 4. Heston and Monte Carlo

### 4.1 Pricer and calibration

**Pricer** [`heston/pricing.py`]. Semi-analytic Gil-Pelaez inversion in the
forward-measure form, with Gauss-Legendre quadrature whose upper limit is widened
adaptively until the characteristic function has decayed. It falls back to Black-76
at the expected-variance vol when the integral cannot converge or the average
variance is below 1e-6. Tests: put-call parity to 1e-6, and agreement with
Black-76 at tiny vol-of-vol (`tests/test_heston_pricing.py`).

**The quadrature bug.** The first version used a fixed range (N = 256, cutoff 400),
which passed moderate-parameter tests but gave negative prices once a calibration
pushed kappa and xi toward their bounds at a 4-day expiry. The module notes 27
outright negative prices at T = 4 days with the fixed cutoff
[`heston/pricing.py` docstring]. The regression test is
`test_stress_short_maturities_with_extreme_bounds_params_stay_arbitrage_free`
(`tests/test_heston_pricing.py`). The commit records "maturity-adaptive
quadrature and short-maturity stress test" [commit 4f048d2].

**Calibration limits** [`heston/calibration.py`].

- Per-expiry, single-start, weighted least squares in IV space (weights: inverse
  relative bid-ask spread), starting at v0 = θ = median(IV)², κ = 2, ξ = 0.5,
  ρ = −0.3, at most 300 function evaluations.
- Bounds: v0 ∈ [1e-4, 1], κ ∈ [0.1, 10], θ ∈ [1e-4, 1], ξ ∈ [0.01, 2.5],
  ρ ∈ [−0.999, 0.999]. The Feller condition is checked, not enforced.
- **At least 26 of 44 daily fits hit a Heston bound.** 26 at the 1e-6 absolute
  tolerance the flag uses (kappa 18, xi 8, v0 5, theta 2, rho 0), and 33 at a 1%
  relative tolerance (within 1% of the bound's value; kappa 26, the others
  unchanged) [`docs/results/daily_fits.csv`, `docs/results/bound_recount.csv`]. Heston IV RMSE has median 0.0147 and mean
  0.0307 (max 0.1503, NIFTY near on 10-16, DTE 3), against SVI's mean 0.0093
  [same file]. Points whose Heston price cannot be inverted are excluded from Heston-versus-market RMSEs, so these Heston figures understate the error on the worst-fit slices.
- **The BANKNIFTY 10-27 Heston parameter set used for Monte Carlo has v0 at its
  lower bound (1e-4)** [`docs/results/mc_validation.csv`, `params_used`], and it is
  one of the more fragile fits in the project (next paragraph). Nine slices in
  `docs/results/daily_fits.csv` have v0 of at most 0.000107; the bound-hit flag marks
  only 5 because its tolerance is 1e-6.
- **kappa is not identified from a single expiry.** It sits at the ceiling of 10 on
  both dates for NIFTY 2026-10-06 [`docs/results/heston_rho_kappa_timeseries.png`].
  For the BANKNIFTY 2026-10-27 expiry in `docs/results/daily_fits.csv` it is within
  0.001 of the ceiling on nine consecutive dates (09-30 to 10-16) and 1.93 to 2.73
  on the others (09-24, 10-19, 10-21).

**Provenance and fragility of the stored BANKNIFTY 2026-10-27 set.** The stored set
(v0 0.00010, kappa 3.247, theta 0.15522, xi 0.946, rho −0.499; forward 56,022.49113,
discount 0.987123, T 0.088173) comes from the **11:06:02 IST snapshot on 2026-09-25**,
not the day's last (11:27:02). I computed the parity forward, discount and T of this
slice at every snapshot from 11:00 to 11:27 (ten ticks); only 11:06:02 matches all
three. Rerunning the single-start Heston calibration there reproduces the stored
parameters (v0 1.00e-4, kappa 3.2466, theta 0.155217, xi 0.9456, rho −0.49895), and
the four `a ≥ 0` SVI fits at that tick reproduce three of the four rho and point
counts in the SVI JSONs still on disk to six decimals (BANKNIFTY 09-29: −0.275868,
99 points; BANKNIFTY 10-27: −0.002998, 77; NIFTY 10-06: −0.045396, 87). The fourth,
NIFTY 09-29 (JSON rho 0.082445, 103 points), matches the 11:09:01 tick exactly
instead (at 11:06:02 it gives 0.0879) [`docs/results/provenance_parity_svi.csv`,
`docs/results/heston_fragility.csv`]. So the JSONs were generated from a mixture of
ticks, consistent with each series' "latest snapshot" being a different tick while
the 2026-09-25 partition was still being collected or synced (I cannot verify the
sync state at the time). "Latest snapshot" is therefore not a stable choice while a
partition is still being collected; the 14:30 design in section 1 avoids it.

The same calibration on each of the ten ticks gives very different parameters
[`docs/results/heston_fragility.csv`]:

- kappa is within 0.0005 of the ceiling of 10 on 9 of 10 ticks; only the 11:06:02
  tick (the stored one) has kappa below it (3.25). The bound flag marks 9 of 10
  ticks (kappa on 7, v0 on 2: 11:06:02 and 11:12:01).
- v0 ranges from 1.0e-4 to 1.85e-3 (√v0 from 1.0% to 4.3%).
- On the nine other ticks theta is 0.059 to 0.063 (√θ 24.3% to 25.0%); on the stored
  tick it is 0.155 (√θ 39.4%). xi ranges 0.95 to 1.57 and rho −0.51 to −0.39.
- Heston IV RMSE is 0.0066 to 0.0112 and the objective 2.2e-4 to 7.4e-4. The stored
  tick has RMSE 0.0100 and objective 3.2e-4, neither the lowest. (Points whose Heston price
  cannot be inverted are excluded from these RMSEs, so they understate the error on the
  worst-fit slices.)
- The average vol to expiry implied by the fitted variance path is 14.2% to 14.8%
  on all ten ticks. The level is pinned by the data while its split into v0, theta
  and kappa is not.

On the 11:06:02 data the stored set has weighted objective 3.218e-04 and IV RMSE 0.00999; the 11:27:02 fit (kappa 10, theta 0.060) has 3.268e-04 and 0.00969, and the 11:00:02 fit 9.767e-04 and 0.00741 [`docs/results/param_dependence.csv`]. Priced at the same forward, discount and T with the same seeds, the stored set is lower than the 11:27:02 set by 0.91 ± 0.07 (ATM call), 1.49 ± 0.10 (5% down-and-out) and 0.93 ± 0.07 (10% down-and-out), mean ± SE over 20 seeds: 0.10%, 0.17% and 0.11% of the price (section 4.2). On the 11:06:02 data the stored set and the 11:27:02 fit have weighted objectives within about 1.5% of each other, and they price the ATM call and the 5% and 10% down-and-out calls within 0.2%. The 11:00:02 fit has a weighted objective about 3× worse (9.77e-4 against 3.22e-4) but the lowest unweighted IV RMSE of the three (0.00741 against 0.00999 for the stored set and 0.00969 for the 11:27:02 fit), so the two metrics disagree about it. (Points whose Heston price cannot be inverted are excluded from Heston-versus-market RMSEs, so they understate the error on the worst-fit slices.) Two of the three sets were priced, on three payoffs, at 32 days, and nothing is claimed beyond that. Kappa is 3.2 for the stored set against 10 for the 11:27:02 fit, and theta 0.155 against 0.060. A Monte Carlo run on the stored set says nothing about those parameters being the market's.

**The joint-calibration failure.** `calibrate_heston_joint` fits one shared
parameter set to several expiries, and recovers the parameters on synthetic data
(`tests/test_heston_calibration.py`). On the real 4-expiry chain from step 3 the
joint Heston smile is nearly flat while market IV and the SVI fit show a wide,
strongly curved smile [`docs/results/all_expiries_svi_vs_heston_joint.png`]. So the
joint fit was abandoned in favour of per-expiry calibration; why it fails on market
data is not established.

### 4.2 Monte Carlo

Every number in this subsection is from `analysis/mc_validation.py` (20 seeds, 0 to
19; 100,000 paths and 50 steps per run, `docs/results/mc_validation.csv`),
`analysis/barrier_check.py` (`docs/results/barrier_check.csv`,
`docs/results/param_dependence.csv`) or `heston/barrier_ref.py` and its tests. The
simulation is Andersen's quadratic-exponential (QE) scheme for the variance with a
martingale-corrected log-price step [`heston/qe.py`].

**Parameters used.** BANKNIFTY 2026-10-27: v0 = 0.00010, kappa = 3.247,
theta = 0.15522, xi = 0.946, rho = −0.499, forward 56,022.49113, discount 0.987123,
T = 0.088173. Source: the 2026-09-25 fit on the **11:06:02 IST snapshot**, matched on
forward, discount and T (section 4.1), as stored in `tests/test_heston_qe.py`.
Recalibrating at 11:06:02 reproduces it (the script's flag reads "same"). It is not
the day's last snapshot: calibrating the same slice at 11:27:02 gives v0 = 2.4e-4,
kappa = 10.0 (the ceiling), theta = 0.0604, xi = 1.095, rho = −0.481, forward
55,986.41, discount 0.98859 and T = 0.088134. The parameters are fragile across the
ten ticks from 11:00 to 11:27 (kappa at the ceiling on 9, v0 from 1e-4 to 1.85e-3,
theta from 0.059 to 0.155, while the implied average vol stays 14.2% to 14.8%;
section 4.1). The Monte Carlo below is valid for this parameter set as given.

**European check, 7 strikes (OTM side), QE MC against `heston_price`, over 20
seeds.** z = (MC − semi-analytic)/standard error, per seed. The table gives the mean
z per strike with its standard error across seeds, the largest |z| over seeds, and
the bias detection limit se_plain/√20 (mean per-run standard error over √20, in
price units: the standard error of the 20-seed mean price).

| k = ln(K/F) | Strike | Type | Semi-analytic | Mean z | SE of mean z | Max abs z | Detection limit (se_plain/√20) |
|---|---|---|---|---|---|---|---|
| -0.09 | 51,200.70 | put | 59.780 | -0.039 | 0.252 | 2.84 | 0.306 |
| -0.06 | 52,760.00 | put | 147.981 | -0.020 | 0.233 | 2.24 | 0.484 |
| -0.03 | 54,366.78 | put | 363.284 | +0.046 | 0.213 | 2.04 | 0.749 |
| +0.00 | 56,022.49 | call | 871.942 | -0.152 | 0.243 | 2.10 | 0.843 |
| +0.03 | 57,728.63 | call | 233.810 | -0.107 | 0.229 | 2.22 | 0.464 |
| +0.06 | 59,486.73 | call | 43.259 | +0.107 | 0.208 | 2.39 | 0.207 |
| +0.09 | 61,298.37 | call | 7.194 | +0.219 | 0.209 | 1.71 | 0.087 |

The mean z is within about 1.05 standard errors of zero at every strike. The
largest case is k = +0.09, mean z 0.219 against an SE of 0.209 (1.05 SE); the other
six are 0.08 to 0.62 SE. Every one of the 140 (strike, seed) z-values is within 5
standard errors (largest |z| 2.84). The detection limit (0.087 to 0.843 price
units, about 0.1% of the price at the money) is the size of bias this check could
have seen at roughly 2 standard errors: a QE bias smaller than that is not detectable
here. **This is a check against the semi-analytic pricer at the same parameters. It
validates the simulation, not the calibration.**

**Variance reduction on the ATM call.**

- *Antithetic variates:* the variance ratio plain/antithetic is about 1.9× (min
  1.895, median 1.923, max 1.989 over the 20 seeds).
- *Control variate:* for the European call the control is the call's own payoff, a
  degenerate case, so the standard error collapses to 0.000. It tests the
  machinery, not real variance reduction.

**Barrier price, Black-Scholes-limit check.** The down-and-out call has no
closed form under Heston, so I checked the simulation in the limit where it has one:
v0 = theta = 0.14², xi = 0.01 (kappa and rho at the stored values), strike at the
forward, 50 monitoring steps, T = 0.088173. The reference is the zero-drift Black-76
down-and-out price for continuous monitoring, shifted for discrete monitoring by
Broadie–Glasserman–Kou (barrier multiplied by exp(−0.5826·σ·√(T/50))) [`heston/barrier_ref.py`].
The formula is tested first (three tests, `tests/test_barrier_ref.py`): it returns the
vanilla price as H → 0 (H = 1e-6·F, to relative 1e-9) and 0 when H = F (to 1e-8); and
the price rises with the distance of the barrier, stays below the vanilla, and the
discrete-monitoring (shifted) price exceeds the continuous one. Barriers 3%, 5% and 10%
below the forward, 20 seeds:

| Barrier below forward | MC mean | BGK-shifted closed form | Unshifted continuous | Mean z (SE across seeds) | Max abs z | Detection limit |
|---|---|---|---|---|---|---|
| 3% | 866.42 | 867.35 | 844.22 | -0.22 (0.30) | 2.85 | 0.97 |
| 5% | 913.19 | 914.05 | 911.99 | -0.20 (0.30) | 2.86 | 0.97 |
| 10% | 916.28 | 917.08 | 917.08 | -0.19 (0.30) | 2.81 | 0.97 |

The three rows are one comparison, not three. The same 20 seeds' simulated paths
price all three barriers (the barrier does not change the simulation), so the rows
share one offset: MC minus closed form is −0.92, −0.86 and −0.80, inside the detection
limit below. The 10% row is effectively a vanilla check: its shifted closed-form price
(917.0822966) differs from the Black-76 vanilla price from `bs_call` (917.0823309) by 3.4e-5
(vanilla minus shifted; 3.7e-8 relative). The mean z is −0.22, −0.20
and −0.19 (SE across seeds 0.30). The detection limit is the standard error of the
20-seed mean price: a price bias smaller than about 2 × 0.97 ≈ 1.9 (0.2% of the price)
could not be seen in these single-price comparisons. The unshifted continuous formula
is off by +22.2 at the 3% barrier and +1.2 at 5%, so the discrete-monitoring shift is
needed.

**The real barrier test is the knock-out cost**, the 10% price minus the 3% price,
paired across seeds (the same paths price both barriers, so the shared offset
cancels): 49.86 ± 0.11 (SE across seeds) against 49.74 from the shifted formula
(z = +1.07) and against 72.86 from the unshifted continuous formula (z = −207). This
check covers the Black-Scholes limit only; it does not test the barrier at the stored
Heston parameters (v0 1e-4, xi 0.946).

**Barrier price under the stored Heston set** (20 seeds; semi-analytic vanilla call
871.942):

| Barrier below forward | Steps | DO price (mean of 20 seeds) | Mean per-run SE | DO / vanilla |
|---|---|---|---|---|
| 3% | 50 | 794.508 | 3.695 | 0.9112 |
| 5% | 50 | 854.459 | 3.755 | 0.9799 |
| 10% | 50 | 871.052 | 3.770 | 0.9990 |
| 5% | 100 | 851.786 | 3.750 | 0.9769 |
| 5% | 200 | 850.540 | 3.746 | 0.9755 |

The 10% barrier price (871.052) is within 0.9 of the vanilla, essentially
vanilla: the DO/vanilla ratio is 0.911 at 3%, 0.980 at 5% and 0.999 at 10%. The knock-out cost
(10% minus 3% price, paired across seeds) is 76.5 ± 0.4.

**Monitoring-step caveat.** The barrier is monitored only on the simulation grid, so
every barrier price above is a discretely monitored price, not a continuously monitored
one. At the 5% barrier the price falls from 854.459 (50 steps) to
851.786 (100) and 850.540 (200); the 50-step minus 200-step difference is
3.92 ± 1.06 (SE across seeds). Extrapolating to continuous monitoring assuming the
monitoring error scales as 1/√steps gives 846.6 ± 1.8 from the 50- and 200-step prices and
846.4 ± 1.7 from a fit to all three, so the 50-step price overstates the
continuous-monitoring price by about 7.8 ± 2.1 (0.9%). The quoted errors are
across seeds only; they do not include the error from assuming the √Δt scaling.

**Barrier parity is an identity.** On each path exactly one of "breached" and "not
breached" holds, so the down-and-out and down-and-in payoffs sum to the vanilla payoff
by construction. The largest absolute gap over the 20 seeds is
2.3e-13, floating-point zero. That shows the code applies the complement
correctly; it is not a validation of the barrier price. The validation is the
Black-Scholes-limit check above.

**Barrier control variate.** The vanilla call is the control for the down-and-out
call, under the stored Heston set, 20 seeds, 50 steps, 100,000 paths per run
[`docs/results/barrier_check.csv`, `heston_cv` rows]. The variance ratio is plain over
control variate:

| Barrier level | Variance ratio min | median | max | Breach fraction | Mean paths per run that breach and finish above the strike (of 100,000) |
|---|---|---|---|---|---|
| 3% below forward (H=54341.82) | 7.5 | 8.1 | 8.6 | 36.75% | 6,073.0 |
| 5% below forward (H=53221.37) | 32.5 | 34.9 | 40.3 | 19.50% | 1,363.1 |
| 10% below forward (H=50420.24) | 961.5 | 2,132.0 | 6,244.1 | 3.81% | 27.4 |
| H=50,000 (10.8% below forward) | 1,343.3 | 3,213.6 | 16,283.0 | 2.96% | 16.0 |

The barrier payoff equals the vanilla payoff on every path except those that breach and
finish above the strike, so the control variate removes more variance the fewer such
paths there are. The ratio therefore depends strongly on the barrier: median
8.1× at the 3% barrier, 34.9× at 5%, 2,132× at 10% and 3,214× at H = 50,000,
where an average of 16 paths per run breach and finish above the strike. Across seeds the
ratio at H = 50,000 varies by an order of magnitude.

**Parameter dependence.** Under the stored set and under the 11:27:02 fit (kappa 10,
theta 0.060), both priced at the stored forward, discount and T with the same 20 seeds, the
paired differences (stored minus 11:27:02 set, mean ± SE over seeds) are
-0.91 ± 0.07 for the ATM call, -1.49 ± 0.10 for the 5% down-and-out and
-0.93 ± 0.07 for the 10% down-and-out, against prices of 871, 854 and
871 [`docs/results/param_dependence.csv`]. On the 11:06:02 data the stored set and the 11:27:02 fit have weighted objectives within about 1.5% of each other, and they price the ATM call and the 5% and 10% down-and-out calls within 0.2%. The 11:00:02 fit has a weighted objective about 3× worse (9.77e-4 against 3.22e-4) but the lowest unweighted IV RMSE of the three (0.00741 against 0.00999 for the stored set and 0.00969 for the 11:27:02 fit), so the two metrics disagree about it. (Points whose Heston price cannot be inverted are excluded from Heston-versus-market RMSEs, so they understate the error on the worst-fit slices.) Two of the three sets were priced, on three payoffs, at 32 days, and nothing is claimed beyond that.

How these figures compare with the earlier post's is in section 5.

## 5. Retractions

- **The 2d2a857 wording.** The commit subject is "Step 2: OTM-only filter fixes
  ITM-inversion rho contamination" and the body says restricting each strike to its
  OTM side "drops that contamination and brings both near-dated rhos back near zero /
  mildly negative, with the calendar check still passing." The filter is still
  justified for the ITM-contamination reason, but those rho values are not
  evidence of anything: with `a` loosened rho moves from a mean of −0.083 to +0.148
  on the same quotes [`docs/results/svi_refit.csv`]. Raw rho is weakly identified
  (commit 1490f37).
- **The committed SVI JSONs are superseded.**
  `results/date=2026-09-24_svi_params.json` and
  `results/date=2026-09-25_svi_params.json` were force-added in commit 5dd8003. For
  2026-09-25 I traced their input (section 4.1): three of the four slices were fitted
  on the 11:06:02 IST snapshot (rho to six decimals, point counts exactly) and NIFTY
  09-29 on the 11:09:01 snapshot. So they are `a ≥ 0` fits on partial-day snapshots,
  taken under two hours into a session whose collection stopped at 11:30 (section 1),
  built from a mixture of ticks while the partition was still being collected or
  synced. They describe no single snapshot. Raw rho is also weakly identified on
  slices like these (section 2), so their rho values are not evidence of skew either
  way. They were untracked in commit fb0f54e (files kept on disk), and nothing here
  depends on them. The 2026-09-24 JSON was not traced.
- **The earlier LinkedIn post.** The write-up corrects these claims in it:
  - *"SVI/SSVI surface fitting"*: the project fits raw SVI per expiry only; there is no
    SSVI.
  - *"91 tests"*: the suite had 91 tests at commit 4f048d2 and now collects 101
    (`pytest --collect-only -q`), the additions covering `daily_fits.py`, the `a_lower`
    option, `skew_check.py` and the closed-form barrier reference
    (`tests/test_barrier_ref.py`).
  - *"ITM options … systematically distorting rho"*: rho is weakly identified on these
    slices (section 2), so ITM contamination distorted the fit but rho is not a
    reliable measure of it. Commit 2d2a857's wording is used only as the motivation for
    the OTM-only filter.
  - *"Cleanest calibration: BankNifty's 32-day expiry, RMSE 0.94, Feller condition
    satisfied"*: that fit came from the 11:06:02 mid-collection snapshot of a partial
    day, and calibrations on the ten snapshots from 11:00 to 11:27 gave very different
    parameters (kappa at its ceiling on 9 of 10; section 4.1), so it is not a clean or
    well-identified calibration. The 0.94 is Heston measured against the SVI curve
    (0.9427 vol points, next item), not against market quotes. The Feller condition does hold for
    the stored set (2·kappa·theta = 1.008 against xi² = 0.895).
  - *The post's RMSE figures (0.94, ~9.4 and the session's 8.02)*: the post's figures were
    Heston against the SVI curve, not against market quotes. On the 2026-09-25 11:06:02
    snapshot, on the same cleaned OTM quotes, Heston against the SVI curve is
    0.9427 vol points for BANKNIFTY 2026-10-27 (77 points) and
    9.387 for BANKNIFTY 2026-09-29 (93 points), which reproduce the post's 0.94 and
    ~9.4 (the 4-day expiry); the session table's column was "SVI-vs-Heston RMSE". The
    session's 8.02 reproduces as market minus Heston, 8.024, on 93 of the
    99 quotes. Against market quotes the Heston fits are off by
    0.9992 vol points (2026-10-27, 77 points) and 8.024 (2026-09-29,
    93 of 99 invertible points); the SVI fit itself is 0.4161 and 2.871 from the market
    (77 and 99 points) [`docs/results/post_rmse_check.csv`]. Points whose Heston price cannot be
    inverted are excluded from the Heston-versus-market figures, so they understate the
    error on the worst-fit slices. For comparison, `docs/results/daily_fits.csv` (14:30
    snapshots, same exclusion) shows Heston IV RMSE with median 0.0147 and maximum 0.150
    (NIFTY near, 10-16, DTE 3), and has no BANKNIFTY slice at 4 days (the nearest is
    09-24 at DTE 5).
  - *"~1.95× antithetic"*: about 1.9× (20-seed range 1.89 to 1.99; 1.95 lies inside it)
    [`docs/results/mc_validation.csv`].
  - *"~1355× control variate, driven by 0.9996 correlation"*: that was a single seed at
    a barrier (H = 50,000) where the option is nearly vanilla. A 1355× variance ratio
    corresponds to a correlation of about 0.9996 between the barrier and vanilla
    payoffs (ratio = 1/(1 − ρ²)), but the ratio depends on the barrier and the seed: over
    20 seeds the median is 8.1× at the 3% barrier, 34.9× at 5%, 2,132× at 10% and
    3,214× at H = 50,000 (range 1,343 to 16,283) (section 4.2).
  - *"Barrier pricing validated independently via in-out parity"*: parity holds by
    construction, so it is an identity, not a validation. The validation is the
    Black-Scholes-limit check, above all its knock-out-cost test (section 4.2).
- **Unsourced: the post's "across 19 multi-start draws".** This is not a correction. No
  multi-start output was saved, and the repo's calibration is single-start
  (`heston/calibration.py`), so I can neither source nor correct anything the post said
  that rests on those draws.

## 6. Analyses that didn't hold up

- **The NIFTY drift.** Across the 12 dates, NIFTY "next" residual skew (after a DTE
  fit) correlated with date order at rho = −0.80, p = 0.0019 Spearman / 0.0026
  permutation [`docs/results/skew_diagnostics.csv`]. That doesn't hold up. It
  weakens to p = 0.055 at the |k| ≤ 0.04 window
  [`docs/results/skew_robustness.csv`], and in the matched-DTE comparison (six
  slices, early minus late) the permutation p is 0.20 two-sided and 0.10 one-sided
  at |k| ≤ 0.02, and 0.30 / 0.15 at ±0.04 [`docs/results/skew_matched_dte.csv`]. The
  difference rests on one slice (10-21; 10-14 and 10-15 are as steep as the early
  ones). I do not claim a drift.
- **BANKNIFTY near's drift was a √T artifact.** The normalised skew
  (skew × √(DTE/365)) of BANKNIFTY "near" looked like it drifted flatter. After a
  DTE fit the residuals show nothing (Spearman p = 0.31, permutation p = 0.31), and
  BANKNIFTY's fitted DTE exponent is −0.07 (95% CI −0.19 to +0.06) rather than −0.5,
  so the √T normalisation over-corrects it [`docs/results/skew_diagnostics.csv`].
  This is consistent with the apparent trend coming from DTE falling through the
  expiry cycle; the unadjusted test on the normalised series was printed, not saved
  to a file.
- **The far-dated BANKNIFTY rho.** Raw-SVI rho for BANKNIFTY "next" (DTE 33–54) is
  positive on 10 of 12 dates under `a ≥ 0` (−0.113 to +0.161) and on all 12 once `a`
  is loosened (+0.085 to +0.866, with `b` rising from at most 0.035 to as much as
  0.465), while the market skew is negative on every date
  [`docs/results/svi_refit.csv`, `docs/results/skew_check.csv`]. The fitted rho
  here is a trade-off between `a`, `b` and rho, not a measure of skew.

## 7. Limitations

- **Small sample.** 12 dates (09-24 to 10-21), 44 slices, one calendar month.
  Per-date, per-bucket series have 8 to 12 points, and 9 slices have fewer than 8
  quotes in the skew window [`docs/results/skew_check.csv`].
- **Claim 3 rests on one date.** See section 3.
- **No test of anything predictive.** Nothing here relates skew to returns, vol
  or any future outcome. There is no claim of stability, a regime change, a drift
  or predictive power.
- **"No error" is not "good fit".** All 44 daily fits finished without raising an
  exception [`docs/results/daily_fits.csv`, `error` column empty]. That is a
  statement about the code running, not about fit quality (see the Heston bound
  hits and RMSE in section 4.1).
- **DTE, calendar date and expiry type move together.** The DTE exponent
  (NIFTY −0.72 [−0.99, −0.34]; BANKNIFTY −0.07 [−0.19, +0.06]; date-cluster
  bootstrap [`docs/results/skew_diagnostics.csv`]) is not a clean term-structure
  measure, and the NIFTY exponent changes sign when thin slices are dropped
  (+0.12 [`docs/results/skew_robustness.csv`]).
- **CIs capture quote scatter only.** The bootstrap resamples quotes within a
  window, so it ignores model error, window-width bias, and the 12-slice sample.
  Several tests were run over the same data: 8 on the residual drift
  [`docs/results/skew_diagnostics.csv`] and 12 p-values from 6 runs on NIFTY "next"
  [`docs/results/skew_robustness.csv`]. All p-values quoted are raw.
- **Raw SVI per expiry; rho is weakly identified.** Slices are fitted independently,
  so there is no built-in cross-expiry no-arbitrage guarantee (a calendar check exists
  in the pipeline [`vol_surface/calendar.py`] but is not part of the daily run), and
  rho, `a` and `b` trade off against each other on these narrow strike ranges
  (section 2).
- **Heston calibration is per-expiry and single-start.** At least 26 of 44 fits hit a
  bound (26 at the flag's 1e-6 tolerance, 33 at 1% relative). The stored BANKNIFTY
  10-27 set has v0 at 1e-4 and is one of ten calibrations within 27 minutes, nine of
  which have kappa at its ceiling and theta near 0.06. On the 11:06:02 data the stored
  set and the 11:27:02 fit have weighted objectives within about 1.5% of each other, and
  they price the ATM call and the 5% and 10% down-and-out calls within 0.2%; the 11:00:02
  fit has a weighted objective about 3× worse but the lowest unweighted IV RMSE of the three
  (0.00741 against 0.00999 and 0.00969), so the metrics disagree about it (points whose Heston
  price cannot be inverted are excluded from these RMSEs, so they understate the error on the
  worst-fit slices). Two of
  the three sets were priced, on three payoffs, at 32 days, and nothing is claimed beyond
  that (section 4.2). Kappa is unidentified from a single expiry, and the joint fit
  failed on market data.
- **"Latest snapshot" is unstable during collection.** The cleaning step keeps the
  latest snapshot per series, which changes while a partition is still being collected
  or synced. The 09-25 JSONs were fitted on 11:06 snapshots (11:09 for NIFTY 09-29) of
  a partial day, and my later refit on the 11:27 to 11:30 snapshots differed for that
  reason. The 14:30 design avoids it.
- **The Monte Carlo validates the simulation only.** It matches the semi-analytic
  pricer at the same parameters (section 4.2); it says nothing about how well those
  parameters describe the market. The barrier is monitored on a 50-step grid: at the 5%
  barrier the 50-step price is about 0.9% above a continuous-monitoring extrapolation,
  and the Black-Scholes-limit check cannot see a price bias below about 1.9 price units.
- **Data quality.** Spot is unchanged over the last four snapshot pairs of the day
  (about 15:15 to 15:27) on 10-14, 10-15 and 10-21 [`analysis/skew_1021_check.py`
  output], a feed artifact that doesn't affect the 14:30 snapshot. 10-02 and 10-20
  are excluded on the audit's evidence alone and are only presumed holidays. Two
  dates (09-25, 10-22) are lost to early collector outages.
- **Not reproducible from a fresh clone** without your own `data/` (section 8).

## 8. Reproducing

`data/` and `results/` are git-ignored, so a fresh clone can't rerun the
pipeline: you need your own collected chains under
`data/date=YYYY-MM-DD/{NIFTY,BANKNIFTY}_<expiry>_<time>.parquet`, written by
`collector.py` (it needs `DHAN_ACCESS_TOKEN`, `DHAN_CLIENT_ID` and optionally
`DATA_DIR`). The tests use synthetic chains and run on a fresh clone:

```
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

With data in place, run from the repo root in the order in `analysis/README.md`:
`analysis/date_audit.py`, `daily_fits.py`, `svi_refit.py`, `skew_check.py`, then
the `analysis/skew_*.py` scripts. `analysis/mc_validation.py` runs without data
(it skips the recalibration check). Bootstraps and permutations use fixed seeds; a
rerun reproduced the existing CSVs and PNGs byte for byte (commit fb0f54e), and
two runs of `mc_validation.py` gave identical numbers. `analysis/heston_provenance.py`
produces the provenance and fragility CSVs in section 4.1, `analysis/barrier_check.py` the
barrier and parameter-dependence CSVs, and `analysis/bound_recount.py` the bound counts.
