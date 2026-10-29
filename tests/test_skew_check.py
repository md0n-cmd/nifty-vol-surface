import numpy as np

import skew_check as sc
from vol_surface.svi import SVIFit


def test_svi_atm_skew_matches_finite_difference():
    a, b, rho, m, sigma, t = 0.001, 0.05, -0.3, 0.01, 0.05, 0.05
    iv, skew = sc.svi_atm(a, b, rho, m, sigma, t)
    fit = SVIFit(a=a, b=b, rho=rho, m=m, sigma=sigma, n_points=0, cost=0.0)
    h = 1e-6
    fd = (fit.iv(h, t) - fit.iv(-h, t)) / (2 * h)
    assert np.isclose(skew, fd, rtol=1e-5)
    assert np.isclose(iv, fit.iv(0.0, t))


def test_market_skew_recovers_slope_and_ci_covers_it():
    k = np.linspace(-0.02, 0.02, 15)
    iv = 0.12 - 0.8 * k + np.random.default_rng(1).normal(0, 0.0005, len(k))
    slope, lo, hi = sc.market_skew(k, iv)
    assert lo < -0.8 < hi and abs(slope + 0.8) < 0.1


def test_market_skew_nan_with_too_few_quotes():
    assert np.isnan(sc.market_skew([0.0, 0.01], [0.1, 0.11])[0])
