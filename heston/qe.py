"""Andersen (2008) Quadratic-Exponential (QE) scheme for simulating the
Heston variance process, plus the log-forward process alongside it.

At each step, v_{t+dt} | v_t follows a (scaled, non-central chi-square)
distribution whose first two moments are known in closed form. QE matches
those exact moments with one of two branches, chosen by
psi = Var[v_{t+dt}|v_t] / E[v_{t+dt}|v_t]^2:

  - psi <= psi_c (low noise-to-signal): a quadratic transform of a standard
    normal, v_{t+dt} = a*(b + Z)^2, Z ~ N(0,1) -- always >= 0 by construction.
  - psi >  psi_c (high noise-to-signal, near-zero-variance regime): a
    mixture of a point mass at zero and an exponential tail, inverted from
    a uniform draw -- also always >= 0 by construction.

This is what makes QE non-negative unconditionally, unlike naive/Euler
discretization of the same SDE (dv = kappa*(theta-v)dt + xi*sqrt(v)dW),
which can and does go negative.

The log-forward step (X_t = ln(F_t/F_0), dX = -0.5*v*dt + sqrt(v)*dW_1,
dW_1 dW_2 = rho*dt, matching heston.pricing/heston.montecarlo's driftless
forward-measure convention) uses Andersen's discretization (section 4):
correlation is carried entirely through v_{t+dt} (already drawn jointly
with v_t via the same QE branch) via K1*v_t + K2*v_{t+dt}, plus an
*independent* orthogonal normal draw scaled by sqrt(1-rho^2) for the
residual, uncorrelated part of the price innovation -- no separate
correlated bivariate draw is needed. K0 is then chosen per-step, per-path
as the exact martingale-correcting drift (not the naive Euler drift
-0.5*v*dt), i.e. the unique constant making E[F_{t+dt}/F_t | v_t] = 1
exactly given the known distribution of v_{t+dt} in whichever branch it
was drawn from -- this is what keeps the discretization free of the small
persistent bias a naive log-Euler drift would introduce.

Reference: L. Andersen, "Efficient Simulation of the Heston Stochastic
Volatility Model", Journal of Computational Finance, 2008, sections 3.2
(variance) and 4 (asset price, incl. 4.3 martingale correction).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from .pricing import heston_price

DEFAULT_PSI_C = 1.5
DEFAULT_GAMMA1 = 0.5   # "central" weighting of v_t vs v_{t+dt} in the integrated-variance
DEFAULT_GAMMA2 = 0.5   # approximation; gamma1=gamma2=0.5 is Andersen's usual default.


def conditional_moments(v, dt, kappa, theta, xi):
    """Exact E[v_{t+dt}|v_t=v] and Var[v_{t+dt}|v_t=v] for the CIR variance
    process dv = kappa*(theta-v)dt + xi*sqrt(v)dW (Andersen 2008, eq. 17-18).
    """
    v = np.asarray(v, dtype=float)
    exp_kdt = np.exp(-kappa * dt)
    one_minus = -np.expm1(-kappa * dt)  # 1 - exp(-kappa*dt), stable for small kappa*dt
    m = theta + (v - theta) * exp_kdt
    s2 = ((v * xi ** 2 * exp_kdt / kappa) * one_minus
          + (theta * xi ** 2 / (2 * kappa)) * one_minus ** 2)
    return m, s2


def _qe_transform(v, dt, kappa, theta, xi, psi_c, u):
    """Core QE branch transform, given v (current variance, array) and u
    (one matched Uniform(0,1) draw per path). Returns v_{t+dt} plus the
    per-path branch parameters (quad mask, a/b for the quadratic branch,
    p/beta for the exponential branch) needed for the log-price step's
    martingale correction.

    Both branches are driven by the SAME per-path uniform via an
    inverse-transform: the quadratic branch uses ndtri(u) (inverse
    standard-normal CDF) as its Z, the exponential branch uses u directly
    as its own inversion variable. This is what lets antithetic sampling
    (u -> 1-u) work correctly even when a path and its antithetic partner
    land in different branches at a given step -- see simulate_heston_qe.
    """
    v = np.asarray(v, dtype=float)
    u = np.asarray(u, dtype=float)
    m, s2 = conditional_moments(v, dt, kappa, theta, xi)
    psi = s2 / m ** 2

    v_next = np.empty_like(v)
    a_arr = np.full_like(v, np.nan)
    b_arr = np.full_like(v, np.nan)
    p_arr = np.full_like(v, np.nan)
    beta_arr = np.full_like(v, np.nan)

    quad = psi <= psi_c
    expo = ~quad

    if np.any(quad):
        inv_psi = 2.0 / psi[quad]
        b2 = inv_psi - 1.0 + np.sqrt(inv_psi * (inv_psi - 1.0))
        a = m[quad] / (1.0 + b2)
        b = np.sqrt(b2)
        z = ndtri(u[quad])
        v_next[quad] = a * (b + z) ** 2
        a_arr[quad] = a
        b_arr[quad] = b

    if np.any(expo):
        psi_e = psi[expo]
        p = (psi_e - 1.0) / (psi_e + 1.0)
        beta = (1.0 - p) / m[expo]
        u_e = u[expo]
        vals = np.zeros(int(np.sum(expo)))
        tail = u_e > p
        vals[tail] = np.log((1.0 - p[tail]) / (1.0 - u_e[tail])) / beta[tail]
        v_next[expo] = vals
        p_arr[expo] = p
        beta_arr[expo] = beta

    return v_next, quad, a_arr, b_arr, p_arr, beta_arr


def _qe_draw(v, dt, kappa, theta, xi, rng, psi_c):
    """One Andersen QE variance step drawing fresh (non-antithetic)
    randomness from rng -- see _qe_transform for the branch math."""
    v = np.asarray(v, dtype=float)
    u = rng.random(len(v))
    return _qe_transform(v, dt, kappa, theta, xi, psi_c, u)


def qe_step(v, dt, kappa, theta, xi, rng, psi_c=DEFAULT_PSI_C):
    """One Andersen QE step: v (array, current variance) -> v_{t+dt} (array),
    matching conditional_moments(v, dt, ...) exactly within each branch."""
    v_next, *_ = _qe_draw(v, dt, kappa, theta, xi, rng, psi_c)
    return v_next


def simulate_variance_qe(v0, kappa, theta, xi, t, n_steps, n_paths, seed=0,
                          psi_c=DEFAULT_PSI_C):
    """Simulates n_paths independent variance paths via Andersen QE.

    Returns an (n_paths, n_steps+1) array; column 0 is v0.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = v0
    for i in range(n_steps):
        paths[:, i + 1] = qe_step(paths[:, i], dt, kappa, theta, xi, rng, psi_c)
    return paths


def _log_price_martingale_correction(v_t, quad, a, b, p, beta, K1, K3, A):
    """K0(v_t) making E[F_{t+dt}/F_t | v_t] = 1 exactly, given v_{t+dt}'s
    known branch distribution (Andersen 2008, section 4.3):

      - quadratic branch: v_{t+dt}=a*(b+Z)^2 is a scaled noncentral-chi2(1,
        b^2) variable, whose MGF at s=A is exp(A*a*b^2/(1-2*A*a)) /
        sqrt(1-2*A*a) for A*a < 1/2.
      - exponential branch: v_{t+dt} is 0 w.p. p, Exponential(beta)
        otherwise, whose MGF at s=A is p + (1-p)*beta/(beta-A) for A<beta.

    In both cases K0 = -(K1 + 0.5*K3)*v_t - log(MGF_{v_{t+dt}}(A)).
    """
    K0 = np.empty_like(v_t)
    base = -(K1 + 0.5 * K3) * v_t

    if np.any(quad):
        aq, bq = a[quad], b[quad]
        denom = 1.0 - 2.0 * A * aq
        log_mgf = (A * aq * bq ** 2) / denom - 0.5 * np.log(denom)
        K0[quad] = base[quad] - log_mgf

    if np.any(~quad):
        pe, betae = p[~quad], beta[~quad]
        mgf = pe + (1.0 - pe) * betae / (betae - A)
        K0[~quad] = base[~quad] - np.log(mgf)

    return K0


def simulate_heston_qe(forward, v0, kappa, theta, xi, rho, t, n_steps, n_paths,
                        seed=0, psi_c=DEFAULT_PSI_C, gamma1=DEFAULT_GAMMA1,
                        gamma2=DEFAULT_GAMMA2, antithetic=False):
    """Simulates the Heston forward price alongside its variance via
    Andersen's QE scheme with the martingale-correcting log-price drift.

    antithetic=True pairs up paths: the first n_paths/2 rows ("primal") draw
    fresh Uniform(0,1) innovations each step for both the variance branch
    and the price residual; the last n_paths/2 rows ("antithetic") reuse the
    exact same per-step uniforms, complemented (u -> 1-u). Since both
    branches' transforms in _qe_transform are inverse-uniform draws,
    complementing u is equivalent to negating the underlying normal (for
    the quadratic branch, ndtri(1-u) = -ndtri(u) by symmetry) while still
    being well-defined if a path and its antithetic partner diverge into
    different branches after the first step. n_paths must be even.

    Returns (f_paths, v_paths), each an (n_paths, n_steps+1) array; column 0
    is (forward, v0). Row i and row i+n_paths//2 are an antithetic pair when
    antithetic=True.
    """
    if antithetic and n_paths % 2 != 0:
        raise ValueError("n_paths must be even for antithetic sampling (paths come in pairs)")

    rng = np.random.default_rng(seed)
    dt = t / n_steps
    n_pairs = n_paths // 2 if antithetic else None

    K1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    K2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    K3 = gamma1 * dt * (1.0 - rho ** 2)
    K4 = gamma2 * dt * (1.0 - rho ** 2)
    A = K2 + 0.5 * K4

    v_paths = np.empty((n_paths, n_steps + 1))
    x_paths = np.zeros((n_paths, n_steps + 1))  # X_t = ln(F_t / forward)
    v_paths[:, 0] = v0

    for i in range(n_steps):
        v_t = v_paths[:, i]

        if antithetic:
            u_v_half = rng.random(n_pairs)
            u_v = np.concatenate([u_v_half, 1.0 - u_v_half])
            u_price_half = rng.random(n_pairs)
            u_price = np.concatenate([u_price_half, 1.0 - u_price_half])
        else:
            u_v = rng.random(n_paths)
            u_price = rng.random(n_paths)

        v_next, quad, a, b, p, beta = _qe_transform(v_t, dt, kappa, theta, xi, psi_c, u_v)
        v_paths[:, i + 1] = v_next

        K0 = _log_price_martingale_correction(v_t, quad, a, b, p, beta, K1, K3, A)
        zperp = ndtri(u_price)
        vol_term = np.sqrt(np.maximum(K3 * v_t + K4 * v_next, 0.0))
        x_paths[:, i + 1] = x_paths[:, i] + K0 + K1 * v_t + K2 * v_next + vol_term * zperp

    f_paths = forward * np.exp(x_paths)
    return f_paths, v_paths


def _control_variate_adjust(sample, control_sample, control_mean):
    """Standard control-variate adjustment: sample - c*(control_sample -
    control_mean), with the variance-minimizing c = Cov(sample,
    control_sample) / Var(control_sample) estimated from the same simulated
    sample (plug-in sample covariance/variance, ddof=1)."""
    cov_matrix = np.cov(sample, control_sample, ddof=1)
    c_hat = cov_matrix[0, 1] / cov_matrix[1, 1]
    return sample - c_hat * (control_sample - control_mean)


def heston_qe_mc_price(forward, strike, t, v0, kappa, theta, xi, rho, discount,
                        option_type, n_paths=100_000, n_steps=50, seed=0,
                        psi_c=DEFAULT_PSI_C, gamma1=DEFAULT_GAMMA1, gamma2=DEFAULT_GAMMA2,
                        antithetic=False, control_variate=False):
    """European option price via QE Monte Carlo (martingale-corrected drift).
    Returns (price, standard_error), both already multiplied by discount --
    mirrors heston.montecarlo.heston_mc_price's signature/convention.

    antithetic=True averages each (primal, antithetic) pair's payoff into
    one sample first, so the standard error reflects the n_paths/2
    *independent* pair-averages, not n_paths individually-correlated draws.

    control_variate=True uses this same option's own simulated (undiscounted)
    payoff as its control, with known mean heston_price(...)/discount (the
    semi-analytic price, undiscounted) -- target and control are identical,
    so this is a degenerate, perfectly-correlated case that isolates whether
    the control-variate machinery (coefficient estimation, variance
    reduction) is implemented correctly: stderr should collapse to
    (near-)zero, limited only by floating-point precision, rather than
    reduce the price's real sampling noise the way a genuinely different,
    imperfectly-correlated control would. Combines with antithetic by
    applying the adjustment to each pair's average, not to the raw n_paths
    payoffs.
    """
    f_paths, _ = simulate_heston_qe(forward, v0, kappa, theta, xi, rho, t, n_steps,
                                     n_paths, seed, psi_c, gamma1, gamma2, antithetic)
    f_t = f_paths[:, -1]
    if option_type == "CE":
        payoff = np.maximum(f_t - strike, 0.0)
    else:
        payoff = np.maximum(strike - f_t, 0.0)

    if antithetic:
        n_pairs = n_paths // 2
        sample = 0.5 * (payoff[:n_pairs] + payoff[n_pairs:])
        n_eff = n_pairs
    else:
        sample = payoff
        n_eff = n_paths

    if control_variate:
        analytic_price = heston_price(forward, strike, t, v0, kappa, theta, xi, rho,
                                       discount, option_type)
        control_mean = analytic_price / discount  # undiscounted, matching `sample`'s units
        sample = _control_variate_adjust(sample, sample, control_mean)

    price = discount * sample.mean()
    stderr = discount * sample.std(ddof=1) / np.sqrt(n_eff)
    return price, stderr


def heston_qe_down_barrier_call_mc_price(forward, strike, barrier, t, v0, kappa, theta, xi, rho,
                                          discount, knock, n_paths=100_000, n_steps=50, seed=0,
                                          psi_c=DEFAULT_PSI_C, gamma1=DEFAULT_GAMMA1,
                                          gamma2=DEFAULT_GAMMA2, antithetic=False,
                                          control_variate=False):
    """Down-and-{knock} barrier call via QE Monte Carlo, monitored at every
    simulated step (discrete monitoring on the simulation grid itself, not
    corrected to a continuous-barrier analytic).

    knock='out': payoff is the standard call payoff at expiry unless the
    path's running minimum ever breaches (touches or crosses) `barrier`
    before expiry, in which case it's 0.
    knock='in': the complement -- payoff triggers only if breached.

    For a fixed path, knock='out' and knock='in' payoffs are pathwise
    complementary and sum EXACTLY to the vanilla call payoff at that path's
    terminal F_T (exactly one of "breached"/"not breached" holds) -- a
    strong per-path identity, not just an in-expectation one.

    control_variate=True uses the vanilla European call at the same strike
    (priced analytically via heston_price) as the control. Unlike
    heston_qe_mc_price's degenerate self-control, target and control are
    NOT identical here (the barrier payoff is 0 whenever knocked out even
    if the vanilla payoff is positive), so this is a genuine,
    imperfectly-correlated control variate rather than the trivial case.
    """
    if knock not in ("in", "out"):
        raise ValueError(f"knock must be 'in' or 'out', got {knock!r}")

    f_paths, _ = simulate_heston_qe(forward, v0, kappa, theta, xi, rho, t, n_steps,
                                     n_paths, seed, psi_c, gamma1, gamma2, antithetic)
    f_t = f_paths[:, -1]
    breached = f_paths.min(axis=1) <= barrier

    vanilla_payoff = np.maximum(f_t - strike, 0.0)
    barrier_payoff = np.where(breached, 0.0, vanilla_payoff) if knock == "out" \
        else np.where(breached, vanilla_payoff, 0.0)

    if antithetic:
        n_pairs = n_paths // 2
        sample = 0.5 * (barrier_payoff[:n_pairs] + barrier_payoff[n_pairs:])
        control_sample = 0.5 * (vanilla_payoff[:n_pairs] + vanilla_payoff[n_pairs:])
        n_eff = n_pairs
    else:
        sample = barrier_payoff
        control_sample = vanilla_payoff
        n_eff = n_paths

    if control_variate:
        analytic_vanilla = heston_price(forward, strike, t, v0, kappa, theta, xi, rho,
                                         discount, "CE")
        control_mean = analytic_vanilla / discount
        sample = _control_variate_adjust(sample, control_sample, control_mean)

    price = discount * sample.mean()
    stderr = discount * sample.std(ddof=1) / np.sqrt(n_eff)
    return price, stderr
