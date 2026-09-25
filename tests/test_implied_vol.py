import numpy as np
import pytest

from vol_surface.implied_vol import attach_iv_surface, black76_price, implied_vol
from vol_surface.clean import clean_chain
from vol_surface.forward import forwards_by_expiry
from tests.conftest import make_chain


@pytest.mark.parametrize("option_type", ["CE", "PE"])
@pytest.mark.parametrize("sigma_true", [0.05, 0.15, 0.35, 0.8])
def test_implied_vol_roundtrips_black76_price(option_type, sigma_true):
    forward, strike, t, discount = 25000.0, 25500.0, 0.05, 0.999
    price = black76_price(forward, strike, t, sigma_true, discount, option_type)

    sigma_hat = implied_vol(price, forward, strike, t, discount, option_type)

    assert sigma_hat == pytest.approx(sigma_true, rel=1e-4)


def test_implied_vol_below_intrinsic_is_nan():
    forward, strike, t, discount = 25000.0, 20000.0, 0.05, 0.999
    intrinsic = black76_price(forward, strike, t, 0.0, discount, "CE")

    sigma_hat = implied_vol(intrinsic * 0.5, forward, strike, t, discount, "CE")

    assert np.isnan(sigma_hat)


def test_attach_iv_surface_adds_k_and_w(known_svi):
    chain = make_chain(svi=known_svi)
    cleaned = clean_chain(chain)
    forwards = forwards_by_expiry(cleaned)

    surface = attach_iv_surface(cleaned, forwards)

    assert {"iv", "k", "w"}.issubset(surface.columns)
    assert (surface["iv"] > 0).all()
    assert np.allclose(surface["w"], surface["iv"] ** 2 * surface["t"])
    # recovered IVs should track the SVI slice they were generated from
    expected_iv = np.sqrt(known_svi.w(surface["k"]) / surface["t"])
    assert np.allclose(surface["iv"], expected_iv, atol=5e-3)
