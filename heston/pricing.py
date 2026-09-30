"""Semi-analytic Heston European option pricing via Gil-Pelaez Fourier
inversion, forward-measure form (mirrors implied_vol.black76_price's
convention: an explicit `discount` factor, no separate risk-free rate).

    P_j(K) = 0.5 + (1/pi) * integral_0^inf Re[ exp(-i*u*x) * phi_j(u) / (i*u) ] du
    x = ln(K/F)
    C_undiscounted = F*P1 - K*P2
    P_undiscounted = K*(1-P2) - F*(1-P1)

phi_2 is heston_char_fn (the risk-neutral/forward-measure char. function of
X_T=ln(F_T/F_0)); phi_1(u) = phi_2(u - i) follows from the martingale
identity E[F_T]=F_0 (see characteristic.py), so only one characteristic
function needs implementing.

Integrates with a Gauss-Legendre rule over (0, U_MAX): nodes are strictly
interior so the removable singularity at u=0 is never evaluated. U_MAX is
NOT a single fixed constant -- how fast phi decays in u depends heavily on
t and the parameters (a short maturity or a large vol-of-vol makes phi
decay much more slowly, needing a much wider integration range: U_MAX=400
converges fine at T=30 days but gives 27 outright negative prices at
T=4 days for the same parameters). _select_u_max adaptively widens the
range per call until phi has genuinely decayed, so accuracy doesn't
silently depend on which maturity/parameter regime happens to get priced.

At some extreme-but-in-bounds corners (near-zero v0/theta, kappa and xi
near their calibration ceilings, T of a few days) phi decays so slowly
that it doesn't even reach tolerance by U_MAX_CAP=20000 -- this isn't a
resolution problem (N=4096 vs 16384 at the same cutoff gives an identical,
still-wrong answer); it's total variance genuinely near zero combined with
a heavy-tailed contribution from the huge vol-of-vol, a real edge of
Heston's moment-existence domain, not a quadrature bug. _select_u_max
reports whether it actually converged, and heston_price_batch falls back
to Black-76 at the (tiny) expected-variance-implied vol whenever it
didn't, rather than silently trusting an under-converged integral.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from vol_surface.implied_vol import black76_price

from .characteristic import heston_char_fn

DECAY_TOL = 1e-10
U_START = 50.0
U_GROWTH = 1.6
U_MAX_CAP = 20_000.0
NODES_PER_UNIT = 0.7
MIN_NODES = 128
MAX_NODES = 4096

# Below this expected average variance, X_T's true distribution is close to
# a point mass at 0 (little time and little vol for anything to happen), so
# phi barely decays in u at all -- Fourier inversion is fundamentally
# ill-conditioned there (no finite cutoff resolves it), regardless of how
# large U_MAX_CAP is set. Black-76 at the tiny effective vol is the correct
# limit (it's what Heston converges to as vol-of-vol's realized impact
# vanishes) and is used directly instead of integrating.
MIN_AVG_VARIANCE = 1e-6


def _expected_total_variance(t, v0, kappa, theta):
    """E[integral_0^T v_s ds] for the Heston variance process -- exact,
    from dv = kappa*(theta - v)dt + ..., E[v_t] = theta + (v0-theta)*e^{-kappa t}."""
    if kappa * t < 1e-8:
        return v0 * t  # avoid 0/0 in the closed form as kappa*t -> 0
    return theta * t + (v0 - theta) * (1 - np.exp(-kappa * t)) / kappa


@lru_cache(maxsize=256)
def _gauss_legendre_unit(n_nodes):
    """Gauss-Legendre nodes/weights on [-1,1], cached by node count -- these
    don't depend on t or the Heston parameters, only on how many nodes a
    given call's U_MAX needs."""
    return np.polynomial.legendre.leggauss(n_nodes)


def _select_u_max(t, v0, kappa, theta, xi, rho):
    """Grow u until |phi(u)| and |phi(u-i)| have both decayed below
    DECAY_TOL relative to phi(0)=1. Returns (u_max, converged) -- converged
    is False if DECAY_TOL was never reached by U_MAX_CAP, which signals the
    integral is not trustworthy at any practical resolution (see module
    docstring), not merely under-resolved."""
    u = U_START
    while u < U_MAX_CAP:
        phi2 = heston_char_fn(np.array([u]), t, v0, kappa, theta, xi, rho)[0]
        phi1 = heston_char_fn(np.array([u - 1j]), t, v0, kappa, theta, xi, rho)[0]
        if abs(phi2) < DECAY_TOL and abs(phi1) < DECAY_TOL:
            return u, True
        u *= U_GROWTH
    return U_MAX_CAP, False


def _quadrature_nodes(t, v0, kappa, theta, xi, rho):
    u_max, converged = _select_u_max(t, v0, kappa, theta, xi, rho)
    n_nodes = int(np.clip(u_max * NODES_PER_UNIT, MIN_NODES, MAX_NODES))
    nodes, weights = _gauss_legendre_unit(n_nodes)
    U = 0.5 * u_max * (nodes + 1.0)            # map [-1,1] -> (0, u_max)
    W = 0.5 * u_max * weights
    return U, W, converged


def _probabilities(forward, strikes, t, v0, kappa, theta, xi, rho):
    """Returns (P1, P2, converged) -- one P1/P2 entry per strike."""
    U, W, converged = _quadrature_nodes(t, v0, kappa, theta, xi, rho)

    phi2 = heston_char_fn(U, t, v0, kappa, theta, xi, rho)                 # (N,)
    phi1 = heston_char_fn(U - 1j, t, v0, kappa, theta, xi, rho)            # (N,)

    x = np.log(strikes / forward)                                          # (n_strikes,)
    iu = 1j * U[:, None]                                                   # (N,1)
    phase = np.exp(-iu * x[None, :])                                       # (N,n_strikes)

    integrand1 = np.real(phase * phi1[:, None] / iu)                       # (N,n_strikes)
    integrand2 = np.real(phase * phi2[:, None] / iu)

    P1 = 0.5 + (1.0 / np.pi) * (W[:, None] * integrand1).sum(axis=0)
    P2 = 0.5 + (1.0 / np.pi) * (W[:, None] * integrand2).sum(axis=0)
    return P1, P2, converged


def _black76_fallback(forward, strikes, t, v0, kappa, theta, discount, option_types):
    effective_vol = np.sqrt(max(_expected_total_variance(t, v0, kappa, theta) / t, 0.0))
    return np.array([
        black76_price(forward, k, t, effective_vol, discount, ty)
        for k, ty in zip(strikes, option_types)
    ])


def heston_price_batch(forward, strikes, t, v0, kappa, theta, xi, rho,
                        discount, option_types):
    """Undiscounted-then-discounted Heston price for an array of strikes,
    each with its own option_type ('CE' or 'PE')."""
    strikes = np.asarray(strikes, dtype=float)
    option_types = np.asarray(option_types)

    avg_variance = _expected_total_variance(t, v0, kappa, theta) / t
    if avg_variance < MIN_AVG_VARIANCE:
        return _black76_fallback(forward, strikes, t, v0, kappa, theta, discount, option_types)

    P1, P2, converged = _probabilities(forward, strikes, t, v0, kappa, theta, xi, rho)
    if not converged:
        return _black76_fallback(forward, strikes, t, v0, kappa, theta, discount, option_types)

    call = forward * P1 - strikes * P2
    put = strikes * (1 - P2) - forward * (1 - P1)

    undiscounted = np.where(option_types == "CE", call, put)
    return discount * undiscounted


def heston_price(forward, strike, t, v0, kappa, theta, xi, rho, discount, option_type):
    """Single-strike convenience wrapper around heston_price_batch."""
    return float(heston_price_batch(
        forward, [strike], t, v0, kappa, theta, xi, rho, discount, [option_type],
    )[0])
