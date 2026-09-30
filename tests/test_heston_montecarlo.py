import pytest

from heston.montecarlo import heston_mc_price
from heston.pricing import heston_price


@pytest.mark.parametrize("strike,option_type", [
    (100.0, "CE"),
    (105.0, "CE"),
    (95.0, "PE"),
])
def test_mc_matches_semianalytic_price_within_noise(strike, option_type):
    forward, t, discount = 100.0, 0.1, 0.995
    params = dict(v0=0.02, kappa=2.0, theta=0.02, xi=0.3, rho=-0.5)

    analytic = heston_price(forward, strike, t, **params, discount=discount,
                             option_type=option_type)
    mc_price, mc_stderr = heston_mc_price(forward, strike, t, **params, discount=discount,
                                           option_type=option_type,
                                           n_paths=200_000, n_steps=50, seed=0)

    assert abs(mc_price - analytic) < 5 * mc_stderr
