"""Barrier checks for the QE Monte Carlo (20 seeds, 0..19; 100,000 paths).

1. Black-Scholes limit (v0 = theta = 0.14^2, xi = 0.01): MC down-and-out call vs the
   closed-form continuous price shifted by Broadie-Glasserman-Kou (heston/barrier_ref.py),
   barriers 3%, 5%, 10% below the forward, 50 steps; z per barrier across seeds.
2. Stored Heston set: DO price, standard error and DO/vanilla at the same barriers (50
   steps) and at 5% for 50, 100, 200 steps.
3. Parameter dependence: weighted objective and IV RMSE at the 11:06:02 data of the stored
   set, the 11:27:02 fit and the 11:00:02 fit; ATM call, 5% and 10% DO priced under the
   stored set and the 11:27:02 set with the same seeds (paired difference, mean +/- SE).

In: data/date=2026-09-25/ (item 3 objective), docs/results/heston_fragility.csv.
Out: docs/results/barrier_check.csv, docs/results/param_dependence.csv.
Run from the repo root: python analysis/barrier_check.py
"""
import sys; sys.path.insert(0, ".")
from pathlib import Path

import numpy as np
import pandas as pd

from daily_fits import _rmse, load_surface
from heston.barrier_ref import down_out_call_continuous, down_out_call_discrete_bgk
from heston.calibration import _residuals, _slice_inputs
from heston.pricing import heston_price
from heston.qe import (_control_variate_adjust, heston_qe_down_barrier_call_mc_price, heston_qe_mc_price,
                       simulate_heston_qe)

SEEDS, N_PATHS = list(range(20)), 100_000
F, D, T = 56022.49113, 0.987123, 0.088173
STORED = dict(v0=0.00010, kappa=3.247, theta=0.15522, xi=0.946, rho=-0.499)
SIGMA = 0.14
BS = dict(v0=SIGMA ** 2, kappa=STORED["kappa"], theta=SIGMA ** 2, xi=0.01, rho=STORED["rho"])
BARRIERS = [0.03, 0.05, 0.10]
DAY = Path("data/date=2026-09-25")
FRAG = Path("docs/results/heston_fragility.csv")
P = ("v0", "kappa", "theta", "xi", "rho")


def do_mc(params, barrier, steps, seed):
    return heston_qe_down_barrier_call_mc_price(F, F, barrier, T, **params, discount=D, knock="out",
                                                n_paths=N_PATHS, n_steps=steps, seed=seed)


def main():
    rows = []
    add = lambda **kw: rows.append(kw)

    # 1. Black-Scholes limit
    bs_prices, bs_ref, bs_cont, bs_se = {}, {}, {}, {}
    for b in BARRIERS:
        H = F * (1 - b)
        bgk = down_out_call_discrete_bgk(F, F, H, T, SIGMA, D, 50)
        cont = down_out_call_continuous(F, F, H, T, SIGMA, D)
        zs, prices, ses = [], [], []
        for seed in SEEDS:
            mc, se = do_mc(BS, H, 50, seed)
            zs.append((mc - bgk) / se); prices.append(mc); ses.append(se)
        z = np.array(zs)
        bs_prices[b], bs_ref[b], bs_cont[b], bs_se[b] = np.array(prices), bgk, cont, np.mean(ses)
        add(section="bs_limit_detection", item=f"barrier {b:.0%}: bias detection limit", value=np.mean(ses) / np.sqrt(len(SEEDS)),
            note="value = mean per-run MC standard error / sqrt(20): the standard error of the 20-seed mean price, "
                 "so a bias smaller than about 2x this cannot be detected")
        add(section="bs_limit", item=f"barrier {b:.0%} below forward (H={H:.2f})", value=np.mean(prices),
            stderr=np.mean(ses), reference=bgk, z=z.mean(), extra=z.std(ddof=1) / np.sqrt(len(z)),
            note=f"value=mean MC DO price, stderr=mean per-run se, reference=BGK-shifted closed form, "
                 f"z=mean z over 20 seeds, extra=SE of mean z across seeds; continuous unshifted={cont:.4f}; "
                 f"max|z|={np.abs(z).max():.2f}; mean(MC-BGK)={np.mean(prices) - bgk:.4f}; "
                 f"vs continuous: {np.mean(prices) - cont:.4f}")

    # knock-out cost: DO(10% barrier) - DO(3% barrier), paired across seeds (same paths per seed)
    cost_mc = bs_prices[0.10] - bs_prices[0.03]
    n = len(SEEDS); se_c = cost_mc.std(ddof=1) / np.sqrt(n)
    for label, ref in [("BGK-shifted closed form", bs_ref[0.10] - bs_ref[0.03]),
                       ("unshifted continuous closed form", bs_cont[0.10] - bs_cont[0.03])]:
        add(section="bs_limit_cost", item=f"knock-out cost (10% price - 3% price), MC vs {label}", value=cost_mc.mean(),
            stderr=se_c, reference=ref, z=(cost_mc.mean() - ref) / se_c,
            note="value = mean MC cost over 20 seeds, stderr = paired SE across seeds of the per-seed cost "
                 "(same paths per seed for both barriers), reference = closed-form cost, z = (MC - reference)/stderr")

    # 2. stored Heston set
    steps_prices = {}
    van = heston_price(F, F, T, **STORED, discount=D, option_type="CE")
    for b, steps in [(0.03, 50), (0.05, 50), (0.10, 50), (0.05, 100), (0.05, 200)]:
        H = F * (1 - b)
        res = [do_mc(STORED, H, steps, s) for s in SEEDS]
        pr = np.array([r[0] for r in res]); se = np.array([r[1] for r in res])
        steps_prices[(b, steps)] = pr
        add(section="heston_stored", item=f"barrier {b:.0%} below forward, {steps} steps", value=pr.mean(),
            stderr=se.mean(), reference=van, z=pr.mean() / van, extra=pr.std(ddof=1) / np.sqrt(len(pr)),
            note="value=mean DO price over 20 seeds, stderr=mean per-run se, reference=semi-analytic vanilla, "
                 "z column=DO/vanilla, extra=SE of mean price across seeds")

    cost_h = steps_prices[(0.10, 50)] - steps_prices[(0.03, 50)]
    add(section="heston_cost", item="knock-out cost (10% price - 3% price), stored Heston set, 50 steps",
        value=cost_h.mean(), stderr=cost_h.std(ddof=1) / np.sqrt(len(cost_h)),
        note="value = mean cost over 20 seeds, stderr = paired SE across seeds (same paths per seed); no closed form")
    p50, p100, p200 = (steps_prices[(0.05, m)] for m in (50, 100, 200))
    d = p50 - p200
    add(section="heston_steps", item="5% barrier: price(50 steps) - price(200 steps)", value=d.mean(),
        stderr=d.std(ddof=1) / np.sqrt(len(d)),
        note="value = mean difference over 20 seeds, stderr = SE across seeds of the per-seed difference "
             "(different step counts draw different shocks, so the two runs are independent given the seed)")
    ext2 = 2 * p200 - p50  # P(m) = Pinf + b/sqrt(m): the 50 and 200 step points give Pinf = 2*P200 - P50
    x = np.array([1 / np.sqrt(50), 1 / np.sqrt(100), 1 / np.sqrt(200)])
    ext3 = np.array([np.polyfit(x, [a_, b_, c_], 1)[1] for a_, b_, c_ in zip(p50, p100, p200)])
    for label, e in [("two-point (50, 200 steps)", ext2), ("three-point OLS (50, 100, 200 steps)", ext3)]:
        bias = p50 - e
        add(section="heston_steps", item=f"5% barrier: continuous-monitoring extrapolation, {label}, P = Pinf + b/sqrt(steps)",
            value=e.mean(), stderr=e.std(ddof=1) / np.sqrt(len(e)), reference=p50.mean(), z=bias.mean(),
            extra=bias.std(ddof=1) / np.sqrt(len(bias)),
            note="value = mean extrapolated continuous-monitoring DO price, stderr = SE across seeds, "
                 "reference = mean 50-step price, z column = 50-step price minus extrapolation (monitoring bias), "
                 "extra = SE of that bias across seeds; assumes the sqrt(dt) scaling of the monitoring error")

    # 2b. down-and-out control variate (vanilla call as control), stored set, 50 steps, one simulation per seed
    levels = [(f"3% below forward (H={F * 0.97:.2f})", F * 0.97), (f"5% below forward (H={F * 0.95:.2f})", F * 0.95),
              (f"10% below forward (H={F * 0.90:.2f})", F * 0.90),
              (f"H=50,000 ({100 * (1 - 50_000 / F):.1f}% below forward)", 50_000.0)]
    van_undisc = heston_price(F, F, T, **STORED, discount=D, option_type="CE") / D
    cv = {lab: [] for lab, _ in levels}; br = {lab: [] for lab, _ in levels}; br_itm = {lab: [] for lab, _ in levels}
    for seed in SEEDS:
        paths, _ = simulate_heston_qe(F, STORED["v0"], STORED["kappa"], STORED["theta"], STORED["xi"], STORED["rho"],
                                      T, 50, N_PATHS, seed)
        f_t = paths[:, -1]; vanilla = np.maximum(f_t - F, 0.0); pmin = paths.min(axis=1)
        for lab, H in levels:
            breached = pmin <= H
            do = np.where(breached, 0.0, vanilla)
            adj = _control_variate_adjust(do, vanilla, van_undisc)
            cv[lab].append((do.std(ddof=1) / adj.std(ddof=1)) ** 2)
            br[lab].append(breached.mean()); br_itm[lab].append(int((breached & (f_t > F)).sum()))
    for lab, _ in levels:
        r = np.array(cv[lab])
        for nm, v in [("min", r.min()), ("median", np.median(r)), ("max", r.max())]:
            add(section="heston_cv", item=f"barrier {lab}: DO control-variate variance ratio {nm}", value=v,
                note="value = plain/control-variate variance ratio (the vanilla call is the control), over 20 seeds, "
                     "stored Heston set, 50 steps, 100,000 paths")
        add(section="heston_cv", item=f"barrier {lab}: breach fraction (mean over 20 seeds)", value=float(np.mean(br[lab])),
            note="share of paths whose running minimum touches the barrier")
        add(section="heston_cv", item=f"barrier {lab}: mean paths per run that breach and finish above the strike",
            value=float(np.mean(br_itm[lab])),
            note="count out of 100,000 paths: the only paths where the down-and-out payoff differs from the vanilla payoff")

    # 3. parameter dependence
    frag = pd.read_csv(FRAG, dtype={"tick": str}).set_index("tick")
    sets = {"stored": STORED, "fit_112702": {p: float(frag.loc["112702", p]) for p in P},
            "fit_110002": {p: float(frag.loc["110002", p]) for p in P}}
    surface, f = load_surface(DAY / "BANKNIFTY_2026-10-27_110602.parquet")
    inputs = _slice_inputs(surface, f["forward"], f["discount"], f["t"])
    from heston.calibration import HestonFit
    for name, pr in sets.items():
        resid = _residuals([pr[p] for p in P], *inputs)
        fit = HestonFit(**pr, n_points=len(surface), cost=0.0)
        iv = fit.implied_vol_curve(f["forward"], surface["strike"].to_numpy(), f["t"], f["discount"],
                                   surface["type"].to_numpy())
        add(section="objective_at_110602", item=name, value=0.5 * float(np.sum(resid ** 2)),
            stderr=_rmse(iv, surface["iv"].to_numpy()),
            note="value=weighted objective (0.5*sum of squared weighted IV residuals, as least_squares cost); "
                 "stderr column=IV RMSE; evaluated on the 11:06:02 slice with its own F/discount/T; params: "
                 + ", ".join(f"{p}={pr[p]:.6g}" for p in P))
    a, b2 = sets["stored"], sets["fit_112702"]
    prods = {"ATM call": lambda pr, s: heston_qe_mc_price(F, F, T, **pr, discount=D, option_type="CE",
                                                           n_paths=N_PATHS, n_steps=50, seed=s),
             "DO 5% below": lambda pr, s: do_mc(pr, F * 0.95, 50, s),
             "DO 10% below": lambda pr, s: do_mc(pr, F * 0.90, 50, s)}
    for label, fn in prods.items():
        pa = np.array([fn(a, s)[0] for s in SEEDS]); pb = np.array([fn(b2, s)[0] for s in SEEDS])
        sea = np.array([fn(a, s)[1] for s in SEEDS])
        d = pa - pb
        add(section="price_difference", item=f"{label}: stored - 11:27:02 set", value=d.mean(),
            stderr=d.std(ddof=1) / np.sqrt(len(d)), reference=pa.mean(), z=pb.mean(), extra=sea.mean(),
            note="value=mean paired difference over 20 seeds, stderr=SE of that mean, reference=mean price "
                 "under stored set, z column=mean price under 11:27:02 set, extra=mean per-run se (stored); "
                 "both sets priced at the stored F/discount/T with the same seeds")
    df = pd.DataFrame(rows, columns=["section", "item", "value", "stderr", "reference", "z", "extra", "note"])
    out_b = df[df.section != "price_difference"].loc[lambda x: x.section != "objective_at_110602"]
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    out_b.to_csv("docs/results/barrier_check.csv", index=False)
    df[df.section.isin(["objective_at_110602", "price_difference"])].to_csv("docs/results/param_dependence.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 100)
    print(df.drop(columns="note").to_string(index=False))


if __name__ == "__main__":
    main()
