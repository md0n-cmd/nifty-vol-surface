"""Tests for heston.qe: the Andersen QE variance-process discretization.

Uses the BANKNIFTY 2026-10-27 calibrated fit (v0=0.00010, kappa=3.247,
theta=0.15522, xi=0.946) as the primary parameter set, plus the NIFTY
2026-10-06 fit (kappa=10.0, theta=0.10717, xi=2.063), which -- unlike the
BANKNIFTY fit -- is in the noise-to-signal regime that exercises QE's
exponential branch (see test_one_step_sample_moments_match_exact_cir_formula).
"""
import numpy as np
import pytest

from heston.pricing import heston_price
from heston.qe import (DEFAULT_PSI_C, heston_qe_down_barrier_call_mc_price, heston_qe_mc_price,
                        qe_step, simulate_heston_qe, simulate_variance_qe)

BANKNIFTY_2026_10_27 = dict(kappa=3.247, theta=0.15522, xi=0.946)
BANKNIFTY_V0 = 0.00010


@pytest.mark.parametrize("n_steps", [50, 200, 1000])
def test_long_run_mean_converges_to_theta(n_steps):
    """Over many mean-reversion half-lives (kappa*t ~ 16 here), v_t's
    long-run distribution is the CIR stationary law with mean theta --
    independent of dt, so this should hold whether the path is coarsely or
    finely stepped."""
    kappa, theta, xi = (BANKNIFTY_2026_10_27["kappa"], BANKNIFTY_2026_10_27["theta"],
                        BANKNIFTY_2026_10_27["xi"])
    t = 5.0
    n_paths = 20_000

    paths = simulate_variance_qe(BANKNIFTY_V0, kappa, theta, xi, t, n_steps, n_paths, seed=42)
    v_T = paths[:, -1]

    stationary_variance = theta * xi ** 2 / (2 * kappa)  # CIR stationary Var[v_inf]
    se = np.sqrt(stationary_variance / n_paths)
    assert v_T.mean() == pytest.approx(theta, abs=6 * se)


@pytest.mark.parametrize("n_steps", [10, 100])
@pytest.mark.parametrize("kappa,theta,xi,v0", [
    (3.247, 0.15522, 0.946, 0.00010),  # BANKNIFTY 2026-10-27 fit (quadratic branch)
    (10.0, 0.10717, 2.063, 0.00024),   # NIFTY 2026-10-06 fit (exponential branch)
    (0.1, 0.02, 2.5, 1e-4),            # extreme corner: kappa lo, xi hi, v0 near-zero
    (10.0, 1.0, 2.5, 1.0),             # extreme corner: kappa hi, xi hi, large v0
])
def test_variance_paths_stay_non_negative(kappa, theta, xi, v0, n_steps):
    """The whole point of QE over naive Euler: both branches are
    non-negative by construction, so this should hold trivially -- but a
    transform-derivation bug (e.g. dropping the floor at zero in the
    exponential branch, or a sign error under the sqrt in the quadratic
    branch) would violate it, so it's still worth asserting directly."""
    t = 0.1
    paths = simulate_variance_qe(v0, kappa, theta, xi, t, n_steps, n_paths=5_000, seed=1)
    assert paths.min() >= 0.0


def _independent_exact_cir_moments(v, dt, kappa, theta, xi):
    """Re-derivation of the CIR transition mean/variance, written
    independently of heston.qe.conditional_moments (plain exp() instead of
    expm1, no shared helper) so a shared bug in that helper can't make this
    test pass vacuously."""
    exp_kdt = np.exp(-kappa * dt)
    m = theta + (v - theta) * exp_kdt
    s2 = ((v * xi ** 2 / kappa) * exp_kdt * (1 - exp_kdt)
          + (theta * xi ** 2 / (2 * kappa)) * (1 - exp_kdt) ** 2)
    return m, s2


COMBOS = [
    # (v, dt, kappa, theta, xi, expected_branch) -- BANKNIFTY fit stays in the
    # quadratic branch across a short and a long step; the NIFTY fit (much
    # higher xi/kappa noise-to-signal at this tiny v) crosses into the
    # exponential branch at both a short and a long step.
    (0.00010, 1 / 365, 3.247, 0.15522, 0.946, "QUAD"),
    (0.00010, 0.25, 3.247, 0.15522, 0.946, "QUAD"),
    (0.00024, 0.02, 10.0, 0.10717, 2.063, "EXP"),
    (0.00024, 1.0, 10.0, 0.10717, 2.063, "EXP"),
]


@pytest.mark.parametrize("v,dt,kappa,theta,xi,expected_branch", COMBOS)
def test_one_step_sample_moments_match_exact_cir_formula(v, dt, kappa, theta, xi, expected_branch):
    """Draws many i.i.d. one-step samples from a fixed v via qe_step and
    checks their sample mean/variance against the closed-form CIR
    conditional moments -- QE is defined by matching these moments exactly,
    so this catches a wrong quadratic/exponential transform even though
    such a bug would still (by construction) produce non-negative draws."""
    n_samples = 300_000
    m_exact, s2_exact = _independent_exact_cir_moments(v, dt, kappa, theta, xi)
    psi = s2_exact / m_exact ** 2
    actual_branch = "QUAD" if psi <= DEFAULT_PSI_C else "EXP"
    assert actual_branch == expected_branch, f"psi={psi} didn't land in the intended branch"

    rng = np.random.default_rng(7)
    samples = qe_step(np.full(n_samples, v), dt, kappa, theta, xi, rng)

    sample_mean = samples.mean()
    sample_var = samples.var(ddof=1)

    se_mean = np.sqrt(sample_var / n_samples)
    fourth_central = np.mean((samples - sample_mean) ** 4)
    excess_kurtosis = fourth_central / sample_var ** 2 - 3.0
    se_var = sample_var * np.sqrt(max(excess_kurtosis + 2.0, 0.0) / n_samples)

    assert sample_mean == pytest.approx(m_exact, abs=6 * se_mean)
    assert sample_var == pytest.approx(s2_exact, abs=6 * se_var)


def test_asset_price_simulation_is_a_martingale():
    """With discount=1 (isolating the forward-measure driftless property),
    E[F_T] should equal F_0 -- this is exactly what the per-step martingale
    correction in simulate_heston_qe is derived to guarantee, so a wrong K0
    (e.g. reverting to a naive -0.5*v*dt Euler drift) would show up here as
    a bias well beyond Monte Carlo noise, before even pricing an option."""
    forward, t = 56022.49, 32.18 / 365.0
    rho = -0.499
    n_paths = 200_000

    f_paths, v_paths = simulate_heston_qe(forward, BANKNIFTY_V0, **BANKNIFTY_2026_10_27,
                                           rho=rho, t=t, n_steps=50, n_paths=n_paths, seed=3)
    f_T = f_paths[:, -1]

    assert v_paths.min() >= 0.0
    assert np.all(np.isfinite(f_paths))
    se = f_T.std(ddof=1) / np.sqrt(n_paths)
    assert f_T.mean() == pytest.approx(forward, abs=6 * se)


@pytest.mark.parametrize("k,option_type", [(-0.08, "PE"), (0.0, "CE"), (0.08, "CE")])
def test_qe_mc_price_matches_semianalytic_within_noise(k, option_type):
    """QE Monte Carlo (martingale-corrected drift) vs. the semi-analytic
    Fourier-inversion pricer, at the BANKNIFTY 2026-10-27 fit -- a biased
    log-price discretization would show up as a persistent multi-stderr
    gap here, not just noise."""
    forward, discount, t = 56022.49113, 0.987123, 0.088173
    rho = -0.499
    strike = forward * np.exp(k)

    analytic = heston_price(forward, strike, t, v0=BANKNIFTY_V0, rho=rho,
                             discount=discount, option_type=option_type,
                             **BANKNIFTY_2026_10_27)
    mc_price, se = heston_qe_mc_price(forward, strike, t, v0=BANKNIFTY_V0, rho=rho,
                                       discount=discount, option_type=option_type,
                                       n_paths=100_000, n_steps=50, seed=0,
                                       **BANKNIFTY_2026_10_27)

    assert abs(mc_price - analytic) < 5 * se


def test_antithetic_meaningfully_reduces_stderr_at_matched_path_count():
    """At a fixed total path count, antithetic pairing should give a
    materially smaller stderr than plain MC -- Heston's payoff response is
    asymmetric (skewed) in its Brownian innovations, so a path and its
    negation are correlated beyond the trivial "same v0" case, and
    antithetic variates should capture that. A ratio near 1.0 would mean
    the pairing isn't actually reducing variance (e.g. a bug pairing a path
    with something other than its true antithetic partner), which is worth
    failing loudly on rather than silently accepting."""
    forward, discount, t = 56022.49113, 0.987123, 0.088173
    rho = -0.499
    strike = forward  # ATM

    _, se_plain = heston_qe_mc_price(forward, strike, t, v0=BANKNIFTY_V0, rho=rho,
                                      discount=discount, option_type="CE",
                                      n_paths=100_000, n_steps=50, seed=0,
                                      antithetic=False, **BANKNIFTY_2026_10_27)
    _, se_antithetic = heston_qe_mc_price(forward, strike, t, v0=BANKNIFTY_V0, rho=rho,
                                           discount=discount, option_type="CE",
                                           n_paths=100_000, n_steps=50, seed=0,
                                           antithetic=True, **BANKNIFTY_2026_10_27)

    variance_reduction_ratio = (se_plain / se_antithetic) ** 2
    assert se_antithetic < 0.8 * se_plain, (
        f"antithetic stderr ({se_antithetic:.4f}) should be well below plain MC "
        f"stderr ({se_plain:.4f}) at matched total path count -- variance "
        f"reduction ratio was only {variance_reduction_ratio:.3f}"
    )


@pytest.mark.parametrize("antithetic", [False, True])
def test_control_variate_degenerate_case_collapses_stderr(antithetic):
    """Using this option's own simulated payoff as its own control variate
    is a trivial, perfectly-correlated degenerate case: the estimated
    coefficient should come out to (numerically) exactly 1, cancelling the
    target's sampling noise against itself entirely and leaving only the
    known analytic mean. This isolates whether the coefficient-estimation
    and adjustment machinery is implemented correctly -- a broken
    coefficient (e.g. hardcoded to 0, a no-op) would leave stderr at its
    plain/antithetic-only level instead of collapsing, so this is a real
    regression test, not just a smoke test. Parametrized over antithetic to
    also cover the combined case (control variate applied to each
    antithetic pair average, per heston_qe_mc_price's docstring)."""
    forward, discount, t = 56022.49113, 0.987123, 0.088173
    rho = -0.499
    strike = forward  # ATM

    analytic = heston_price(forward, strike, t, v0=BANKNIFTY_V0, rho=rho,
                             discount=discount, option_type="CE", **BANKNIFTY_2026_10_27)
    price, se = heston_qe_mc_price(forward, strike, t, v0=BANKNIFTY_V0, rho=rho,
                                    discount=discount, option_type="CE",
                                    n_paths=100_000, n_steps=50, seed=0,
                                    antithetic=antithetic, control_variate=True,
                                    **BANKNIFTY_2026_10_27)

    assert se < 1e-6, f"degenerate control variate should collapse stderr near zero, got {se}"
    assert price == pytest.approx(analytic, abs=1e-6)


BARRIER = 50_000.0  # ~10.7% below the BANKNIFTY 2026-10-27 forward


def test_barrier_in_out_parity():
    """Down-and-out and down-and-in payoffs are pathwise complementary --
    exactly one of "breached"/"not breached" holds for each path -- so
    they must sum EXACTLY to the vanilla call payoff at that path's
    terminal value, not just approximately. This is a strong, independent
    correctness check on the barrier/breach-tracking logic that doesn't
    depend on antithetic or control-variate machinery being right."""
    forward, discount, t = 56022.49113, 0.987123, 0.088173
    rho = -0.499
    strike = forward

    f_paths, _ = simulate_heston_qe(forward, BANKNIFTY_V0, **BANKNIFTY_2026_10_27, rho=rho,
                                     t=t, n_steps=50, n_paths=100_000, seed=0)
    f_t = f_paths[:, -1]
    breached = f_paths.min(axis=1) <= BARRIER
    vanilla_payoff = np.maximum(f_t - strike, 0.0)
    do_payoff = np.where(breached, 0.0, vanilla_payoff)
    di_payoff = np.where(breached, vanilla_payoff, 0.0)

    assert np.max(np.abs((do_payoff + di_payoff) - vanilla_payoff)) < 1e-9
    assert 0.0 < breached.mean() < 1.0  # barrier should be reachable but not near-certain

    price_do, se_do = heston_qe_down_barrier_call_mc_price(
        forward, strike, BARRIER, t, v0=BANKNIFTY_V0, rho=rho, discount=discount, knock="out",
        n_paths=100_000, n_steps=50, seed=0, **BANKNIFTY_2026_10_27)
    price_di, se_di = heston_qe_down_barrier_call_mc_price(
        forward, strike, BARRIER, t, v0=BANKNIFTY_V0, rho=rho, discount=discount, knock="in",
        n_paths=100_000, n_steps=50, seed=0, **BANKNIFTY_2026_10_27)
    vanilla_mc_price, vanilla_mc_se = heston_qe_mc_price(
        forward, strike, t, v0=BANKNIFTY_V0, rho=rho, discount=discount, option_type="CE",
        n_paths=100_000, n_steps=50, seed=0, **BANKNIFTY_2026_10_27)

    # same paths/seed -> DO+DI must equal the vanilla MC estimate up to floating point
    assert (price_do + price_di) == pytest.approx(vanilla_mc_price, abs=1e-6)

    vanilla_analytic = heston_price(forward, strike, t, v0=BANKNIFTY_V0, rho=rho,
                                     discount=discount, option_type="CE", **BANKNIFTY_2026_10_27)
    assert (price_do + price_di) == pytest.approx(vanilla_analytic, abs=3 * vanilla_mc_se)


def test_barrier_control_variate_gives_real_but_imperfect_variance_reduction():
    """Unlike heston_qe_mc_price's self-control (target==control exactly,
    corr=1), the barrier option's control (the vanilla call) is only
    imperfectly correlated with the down-and-out target -- they differ
    exactly on the ~3% of paths that breach the barrier. Variance
    reduction should still be substantial (most paths are unaffected by
    the barrier) but this guards against both "no reduction" (broken
    coefficient) and silently regressing to the trivial degenerate case."""
    forward, discount, t = 56022.49113, 0.987123, 0.088173
    rho = -0.499
    strike = forward

    _, se_plain = heston_qe_down_barrier_call_mc_price(
        forward, strike, BARRIER, t, v0=BANKNIFTY_V0, rho=rho, discount=discount, knock="out",
        n_paths=100_000, n_steps=50, seed=0, control_variate=False, **BANKNIFTY_2026_10_27)
    _, se_cv = heston_qe_down_barrier_call_mc_price(
        forward, strike, BARRIER, t, v0=BANKNIFTY_V0, rho=rho, discount=discount, knock="out",
        n_paths=100_000, n_steps=50, seed=0, control_variate=True, **BANKNIFTY_2026_10_27)

    assert 0.0 < se_cv < 0.5 * se_plain  # real reduction ...
    assert se_cv > 1e-4                  # ... but not the trivial ~exact-zero degenerate case
