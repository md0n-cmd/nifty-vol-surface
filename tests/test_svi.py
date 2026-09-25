import numpy as np
import pytest

from vol_surface.svi import SVIFit, fit_all_expiries, fit_svi_slice
from vol_surface.clean import clean_chain
from vol_surface.forward import forwards_by_expiry
from vol_surface.implied_vol import attach_iv_surface
from tests.conftest import make_chain


def test_fit_svi_slice_recovers_known_params(known_svi):
    rng = np.random.default_rng(0)
    k = np.linspace(-0.3, 0.3, 41)
    w_true = known_svi.w(k)
    w_noisy = w_true + rng.normal(scale=1e-5, size=k.shape)

    fit = fit_svi_slice(k, w_noisy)

    assert fit.a == pytest.approx(known_svi.a, abs=5e-3)
    assert fit.b == pytest.approx(known_svi.b, abs=5e-2)
    assert fit.rho == pytest.approx(known_svi.rho, abs=0.1)
    assert np.allclose(fit.w(k), w_true, atol=5e-3)


def test_fit_svi_slice_satisfies_no_arbitrage_conditions(known_svi):
    rng = np.random.default_rng(1)
    k = np.linspace(-0.4, 0.4, 51)
    w_noisy = known_svi.w(k) + rng.normal(scale=2e-5, size=k.shape)

    fit = fit_svi_slice(k, w_noisy)

    min_variance = fit.a + fit.b * fit.sigma * np.sqrt(1 - fit.rho ** 2)
    wing_slack = 4.0 - fit.b * (1 + abs(fit.rho))
    assert min_variance >= -1e-6
    assert wing_slack >= -1e-6


def test_fit_svi_slice_requires_minimum_points():
    with pytest.raises(ValueError):
        fit_svi_slice([0.0, 0.1, -0.1], [0.02, 0.021, 0.022])


def test_fit_all_expiries_end_to_end(known_svi):
    chain = make_chain(svi=known_svi)
    cleaned = clean_chain(chain)
    forwards = forwards_by_expiry(cleaned)
    surface = attach_iv_surface(cleaned, forwards)

    fits = fit_all_expiries(surface)

    key = ("NIFTY", "2026-09-25")
    assert key in fits
    fit = fits[key]
    assert isinstance(fit, SVIFit)
    assert fit.n_points == len(surface)
