import itertools

import numpy as np
import pytest

from heston.calibration import LOWER_BOUNDS, UPPER_BOUNDS
from heston.pricing import heston_price, heston_price_batch
from vol_surface.implied_vol import black76_price

PARAM_SETS = [
    dict(v0=0.02, kappa=2.0, theta=0.02, xi=0.3, rho=-0.5),
    dict(v0=0.04, kappa=1.0, theta=0.06, xi=0.8, rho=0.3),
    dict(v0=0.01, kappa=5.0, theta=0.015, xi=0.2, rho=-0.9),
    dict(v0=0.05, kappa=0.5, theta=0.05, xi=1.2, rho=0.0),
]


@pytest.mark.parametrize("params", PARAM_SETS)
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_put_call_parity_near_machine_precision(params, strike):
    """C and P are computed from independently-integrated P1, P2 (not by
    algebraically deriving one from the other), so this is a genuine check
    that the two Fourier integrals are numerically self-consistent."""
    forward, t, discount = 100.0, 0.1, 0.995

    call = heston_price(forward, strike, t, **params, discount=discount, option_type="CE")
    put = heston_price(forward, strike, t, **params, discount=discount, option_type="PE")

    assert call - put == pytest.approx(discount * (forward - strike), abs=1e-6)


def test_reduces_to_black76_when_vol_of_vol_is_tiny():
    """With xi -> 0 and v0 == theta, Heston's stochastic vol collapses to a
    near-constant instantaneous vol, so its price should match Black-76 at
    that flat vol -- a strong, independent correctness check beyond parity."""
    forward, t, discount = 25000.0, 0.1, 0.999
    flat_vol = 0.18
    params = dict(v0=flat_vol ** 2, kappa=5.0, theta=flat_vol ** 2, xi=1e-4, rho=0.0)

    for strike in (22000.0, 25000.0, 27000.0):
        for option_type in ("CE", "PE"):
            heston = heston_price(forward, strike, t, **params, discount=discount,
                                   option_type=option_type)
            black76 = black76_price(forward, strike, t, flat_vol, discount, option_type)
            assert heston == pytest.approx(black76, rel=1e-3, abs=1e-2)


def test_stress_short_maturities_with_extreme_bounds_params_stay_arbitrage_free():
    """Regression test for a real bug: a fixed Gauss-Legendre integration
    range (N=256, cutoff 400) that passed every test above at moderate
    maturities/parameters gave outright NEGATIVE prices once a real
    calibration run pushed kappa and xi toward their bounds at a 4-day
    expiry -- phi decays much more slowly in u for short T and large
    vol-of-vol, so a fixed cutoff silently under-integrates. This sweeps the
    short end of the term structure (1-7 days) crossed with extreme-but-
    in-bounds corners of the calibration bounds (near-zero v0, kappa/xi
    near their ceilings, rho at both extremes) and would have caught the
    original bug directly.
    """
    forward, discount = 100.0, 0.995
    strikes = forward * np.exp(np.linspace(-0.3, 0.3, 15))
    all_strikes = np.concatenate([strikes, strikes])
    all_types = np.array(["CE"] * len(strikes) + ["PE"] * len(strikes))
    intrinsic = discount * np.where(
        all_types == "CE",
        np.maximum(forward - all_strikes, 0.0),
        np.maximum(all_strikes - forward, 0.0),
    )

    t_days_grid = [1, 2, 4, 7]
    v0_values = [LOWER_BOUNDS[0], LOWER_BOUNDS[0] * 10]          # near-zero v0
    kappa_values = [UPPER_BOUNDS[1] * 0.95, UPPER_BOUNDS[1]]     # kappa near ceiling
    theta_values = [LOWER_BOUNDS[2] * 10, UPPER_BOUNDS[2]]       # small and near-ceiling
    xi_values = [UPPER_BOUNDS[3] * 0.95, UPPER_BOUNDS[3]]        # xi near ceiling
    rho_values = [LOWER_BOUNDS[4], UPPER_BOUNDS[4]]              # both skew extremes

    violations = []
    for t_days in t_days_grid:
        t = t_days / 365.0
        for v0, kappa, theta, xi, rho in itertools.product(
            v0_values, kappa_values, theta_values, xi_values, rho_values,
        ):
            prices = heston_price_batch(forward, all_strikes, t, v0, kappa, theta, xi, rho,
                                         discount, all_types)
            bad_negative = prices < -1e-8
            bad_subintrinsic = prices < intrinsic - 1e-6
            if bad_negative.any() or bad_subintrinsic.any():
                violations.append(dict(
                    t_days=t_days, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
                    n_negative=int(bad_negative.sum()),
                    n_subintrinsic=int(bad_subintrinsic.sum()),
                    min_price=float(prices.min()),
                ))

    assert not violations, (
        f"{len(violations)} (maturity, params) combos gave negative or "
        f"sub-intrinsic prices, e.g. {violations[:3]}"
    )
