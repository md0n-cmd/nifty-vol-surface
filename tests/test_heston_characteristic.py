import numpy as np
import pytest

from heston.characteristic import heston_char_fn

PARAM_SETS = [
    dict(v0=0.02, kappa=2.0, theta=0.02, xi=0.3, rho=-0.5),
    dict(v0=0.04, kappa=1.0, theta=0.06, xi=0.8, rho=0.3),
    dict(v0=0.01, kappa=5.0, theta=0.015, xi=0.2, rho=-0.9),
]


@pytest.mark.parametrize("params", PARAM_SETS)
@pytest.mark.parametrize("t", [0.02, 0.1, 1.0])
def test_martingale_identity_phi_at_minus_i(params, t):
    """E[F_T] = F_0 must hold exactly for any valid Heston parameter set,
    i.e. phi(-1j) == 1 (Ito drift compensation is exact by construction)."""
    phi_at_minus_i = heston_char_fn(np.array([-1j]), t, **params)[0]

    assert phi_at_minus_i == pytest.approx(1.0 + 0.0j, abs=1e-9)


@pytest.mark.parametrize("params", PARAM_SETS)
def test_phi_at_zero_is_one(params):
    """phi(0) = E[1] = 1 trivially, for any t."""
    phi_at_zero = heston_char_fn(np.array([0.0]), 0.1, **params)[0]

    assert phi_at_zero == pytest.approx(1.0 + 0.0j, abs=1e-9)
