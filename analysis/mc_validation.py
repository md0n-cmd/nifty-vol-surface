"""Monte Carlo validation for the Heston QE pricer, over 20 fixed seeds (0..19).

Parameter set: BANKNIFTY 2026-10-27, calibrated on 2026-09-25 from the 11:06:02 IST
snapshot (identified by analysis/heston_provenance.py: its parity forward, discount
and T equal the stored ones; it is not the day's last snapshot, 11:27:02). The
constants below are the ones the tests use (tests/test_heston_qe.py). If data/ is
present, the slice is recalibrated now at 11:06:02 and at the latest snapshot, and
the output flags whether each recalibration differs from the stored parameters.

Reports: (1) a 7-strike European check of QE MC against heston_price: per-seed z
and the mean z per strike with its standard error across seeds, (2) the antithetic
variance ratio on the ATM call per seed (min/median/max), plus the call's own
control variate (degenerate, see heston/qe.py), (3) down-and-out / down-and-in
barrier prices (seed 0), the in/out parity gap and the barrier control-variate
variance ratio per seed.

In: (optional) data/date=2026-09-25/. Out: docs/results/mc_validation.csv.
Run from the repo root: python analysis/mc_validation.py
"""
import sys; sys.path.insert(0, ".")
from pathlib import Path

import numpy as np
import pandas as pd

from heston.calibration import PARAM_NAMES, calibrate_heston
from heston.pricing import heston_price
from heston.qe import (heston_qe_down_barrier_call_mc_price, heston_qe_mc_price,
                       simulate_heston_qe)

SEEDS, N_PATHS, N_STEPS = list(range(20)), 100_000, 50
SEED = SEEDS[0]  # single-seed items (barrier prices, degenerate control variate)
USED = dict(v0=0.00010, kappa=3.247, theta=0.15522, xi=0.946, rho=-0.499)
FORWARD, DISCOUNT, T = 56022.49113, 0.987123, 0.088173
BARRIER = 50_000.0
K_GRID = [-0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09]
SOURCE = ("2026-09-25 fit, BANKNIFTY 2026-10-27, 11:06:02 IST snapshot (matched on forward/discount/T by "
          "analysis/heston_provenance.py); constants from tests/test_heston_qe.py")
RTOL, ATOL = 1e-2, 1e-5
SNAP_DIR, EXPIRY, UNDERLYING = Path("data/date=2026-09-25"), "2026-10-27", "BANKNIFTY"
OUT = Path("docs/results/mc_validation.csv")
P = {**USED, "rho": USED["rho"]}


MATCHED_TICK = "110602"


def today_fit(tick=None):
    """Recalibrate the stored slice now, at a given snapshot tick (HHMMSS) or at the
    day's latest. Returns (fit-dict, forward, discount, t, filename) or None."""
    from daily_fits import load_surface
    files = sorted(SNAP_DIR.glob(f"{UNDERLYING}_{EXPIRY}_*.parquet")) if SNAP_DIR.exists() else []
    if tick:
        files = [f for f in files if f.stem.endswith(tick)]
    if not files:
        return None
    surface, f = load_surface(files[-1])
    fit = calibrate_heston(surface, f["forward"], f["discount"], f["t"])
    return {p: getattr(fit, p) for p in PARAM_NAMES}, f["forward"], f["discount"], f["t"], files[-1].name


def main():
    rows = []
    add = lambda **kw: rows.append(kw)

    for p, v in USED.items():
        add(section="params_used", item=p, value=v, note=SOURCE)
    for name, v in [("forward", FORWARD), ("discount", DISCOUNT), ("t_years", T)]:
        add(section="params_used", item=name, value=v, note=SOURCE)
    add(section="settings", item="seeds", value=len(SEEDS), note=f"seeds 0..{SEEDS[-1]}; n_paths={N_PATHS}, n_steps={N_STEPS}, barrier={BARRIER}")

    for label, tick in [("params_today_matched", MATCHED_TICK), ("params_today_latest", None)]:
        now = today_fit(tick)
        if now is None:
            add(section=label, item="fit", note="data/ snapshot not available; no comparison made")
            continue
        fit, fwd, disc, t, fname = now
        differs = []
        for p in PARAM_NAMES:
            diff = fit[p] - USED[p]
            if abs(diff) > ATOL + RTOL * abs(USED[p]):
                differs.append(p)
            add(section=label, item=p, value=fit[p], reference=USED[p], note=f"diff {diff:+.3g}; snapshot {fname}")
        for name, v, ref, tol in [("forward", fwd, FORWARD, 0.01), ("discount", disc, DISCOUNT, 5e-7),
                                  ("t_years", t, T, 5e-7)]:
            if abs(v - ref) > tol:
                differs.append(name)
            add(section=label, item=name, value=v, reference=ref)
        add(section=label, item="flag", note=("DIFFERS: " + ",".join(differs)) if differs
            else "same (params within rtol 1e-2 / atol 1e-5; forward/discount/T to 1e-2 / 5e-7)")

    # (1) 7-strike European check over all seeds, OTM side, vs semi-analytic price
    zs = {k: [] for k in K_GRID}
    ses = {k: [] for k in K_GRID}
    for k in K_GRID:
        K = FORWARD * np.exp(k); ty = "PE" if k < 0 else "CE"
        an = heston_price(FORWARD, K, T, **P, discount=DISCOUNT, option_type=ty)
        for seed in SEEDS:
            mc, se = heston_qe_mc_price(FORWARD, K, T, **P, discount=DISCOUNT, option_type=ty,
                                        n_paths=N_PATHS, n_steps=N_STEPS, seed=seed)
            zs[k].append((mc - an) / se); ses[k].append(se)
            add(section="european_seed", item=f"k={k:+.2f} {ty} K={K:.2f}", seed=seed, value=mc,
                stderr=se, reference=an, z=(mc - an) / se)
        z = np.array(zs[k])
        add(section="european_summary", item=f"k={k:+.2f} {ty} K={K:.2f}", value=z.mean(),
            stderr=z.std(ddof=1) / np.sqrt(len(z)), reference=an, z=np.abs(z).max(),
            note=f"value = mean z over {len(z)} seeds, stderr = its standard error across seeds, "
                 f"z column = max |z| over seeds")
        add(section="european_detection", item=f"k={k:+.2f} {ty} K={K:.2f}", value=np.mean(ses[k]) / np.sqrt(len(ses[k])),
            stderr=np.mean(ses[k]), reference=an,
            note="value = bias detection limit = mean per-run se_plain / sqrt(20) (price units), the standard error of the "
                 "20-seed mean price; stderr column = mean per-run se_plain; reference = semi-analytic price")

    # (2) variance reduction on the ATM call, per seed
    ratios = []
    for seed in SEEDS:
        _, se_p = heston_qe_mc_price(FORWARD, FORWARD, T, **P, discount=DISCOUNT, option_type="CE",
                                     n_paths=N_PATHS, n_steps=N_STEPS, seed=seed)
        _, se_a = heston_qe_mc_price(FORWARD, FORWARD, T, **P, discount=DISCOUNT, option_type="CE",
                                     n_paths=N_PATHS, n_steps=N_STEPS, seed=seed, antithetic=True)
        ratios.append((se_p / se_a) ** 2)
        add(section="atm_antithetic_seed", item="plain/antithetic", seed=seed, stderr=se_p, reference=se_a,
            z=ratios[-1], note="z column = variance ratio (se_plain/se_antithetic)^2")
    r = np.array(ratios)
    for name, v in [("min", r.min()), ("median", np.median(r)), ("max", r.max())]:
        add(section="atm_antithetic_summary", item=f"variance ratio {name}", value=v, note=f"over {len(r)} seeds")
    _, se_cv = heston_qe_mc_price(FORWARD, FORWARD, T, **P, discount=DISCOUNT, option_type="CE",
                                  n_paths=N_PATHS, n_steps=N_STEPS, seed=SEED, control_variate=True)
    add(section="atm_call_self_control", item="control_variate stderr", stderr=se_cv,
        note="control is the call's own payoff, so degenerate; seed 0")

    # (3) barrier, discrete monitoring on N_STEPS: prices at seed 0, ratios/parity per seed
    out, se_do = {}, {}
    for knock in ("out", "in"):
        out[knock], se_do[knock] = heston_qe_down_barrier_call_mc_price(
            FORWARD, FORWARD, BARRIER, T, **P, discount=DISCOUNT, knock=knock,
            n_paths=N_PATHS, n_steps=N_STEPS, seed=SEED)
        add(section="barrier", item=f"down-and-{knock} call", value=out[knock], stderr=se_do[knock],
            note=f"seed {SEED}")
    an_call = heston_price(FORWARD, FORWARD, T, **P, discount=DISCOUNT, option_type="CE")
    gaps, cv_var, breach = [], [], []
    for seed in SEEDS:
        f_paths, _ = simulate_heston_qe(FORWARD, USED["v0"], USED["kappa"], USED["theta"], USED["xi"],
                                        USED["rho"], T, N_STEPS, N_PATHS, seed)
        pay = np.maximum(f_paths[:, -1] - FORWARD, 0.0)
        v_mc = DISCOUNT * pay.mean(); v_se = DISCOUNT * pay.std(ddof=1) / np.sqrt(N_PATHS)
        d_out, s_out = heston_qe_down_barrier_call_mc_price(
            FORWARD, FORWARD, BARRIER, T, **P, discount=DISCOUNT, knock="out", n_paths=N_PATHS,
            n_steps=N_STEPS, seed=seed)
        d_in, _ = heston_qe_down_barrier_call_mc_price(
            FORWARD, FORWARD, BARRIER, T, **P, discount=DISCOUNT, knock="in", n_paths=N_PATHS,
            n_steps=N_STEPS, seed=seed)
        _, s_cv = heston_qe_down_barrier_call_mc_price(
            FORWARD, FORWARD, BARRIER, T, **P, discount=DISCOUNT, knock="out", n_paths=N_PATHS,
            n_steps=N_STEPS, seed=seed, control_variate=True)
        gaps.append(d_out + d_in - v_mc); cv_var.append((s_out / s_cv) ** 2)
        breach.append(float((f_paths.min(axis=1) <= BARRIER).mean()))
        if seed == SEED:
            add(section="barrier", item="vanilla call (MC, same paths)", value=v_mc, stderr=v_se,
                reference=an_call, note=f"seed {SEED}")
            add(section="barrier", item="parity gap vs analytic vanilla", value=d_out + d_in - an_call,
                stderr=v_se, z=(d_out + d_in - an_call) / v_se, note=f"seed {SEED}")
    cv = np.array(cv_var)
    add(section="barrier_summary", item="max |parity gap| (DO+DI - vanilla MC, same paths)",
        value=float(np.abs(gaps).max()), note=f"over {len(gaps)} seeds; floating-point zero expected")
    for name, v in [("min", cv.min()), ("median", np.median(cv)), ("max", cv.max())]:
        add(section="barrier_summary", item=f"DO control-variate variance ratio {name}", value=v,
            note=f"over {len(cv)} seeds; vanilla call is the control")
    add(section="barrier_summary", item="breach fraction mean", value=float(np.mean(breach)),
        note=f"over {len(breach)} seeds")

    df = pd.DataFrame(rows, columns=["section", "item", "seed", "value", "stderr", "reference", "z", "note"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 400)
    print(df[~df.section.isin(["european_seed", "atm_antithetic_seed"])].to_string(index=False))


if __name__ == "__main__":
    main()
