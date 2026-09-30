import numpy as np
import pandas as pd
import pytest

from heston.calibration import calibrate_heston, calibrate_heston_joint
from heston.pricing import heston_price_batch
from vol_surface.implied_vol import implied_vol


def _synthetic_otm_slice(true_params, forward=25000.0, discount=0.999, t=0.05,
                          n_strikes=41, strike_step=200.0, spread_frac=0.03, seed=0):
    """Strikes priced exactly off known Heston params, OTM side only, with
    a small synthetic bid-ask spread -- mirrors what select_otm() hands to
    calibrate_heston() for a real slice."""
    rng = np.random.default_rng(seed)
    center = round(forward / strike_step) * strike_step
    strikes = center + strike_step * np.arange(-n_strikes // 2, n_strikes // 2 + 1)
    types = np.where(strikes >= forward, "CE", "PE")

    prices = heston_price_batch(forward, strikes, t, *true_params, discount, types)
    ivs = np.array([
        implied_vol(p, forward, k, t, discount, ty)
        for p, k, ty in zip(prices, strikes, types)
    ])
    mid = prices
    half_spread = np.maximum(mid * spread_frac / 2, 0.01)
    noise = rng.normal(scale=1e-4, size=len(strikes))

    return pd.DataFrame({
        "strike": strikes, "type": types,
        "bid": mid - half_spread, "ask": mid + half_spread, "mid": mid,
        "iv": ivs + noise,
    })


def test_calibrate_heston_recovers_known_params():
    true_params = (0.02, 2.0, 0.025, 0.6, -0.4)  # v0, kappa, theta, xi, rho
    forward, discount, t = 25000.0, 0.999, 0.05
    slice_df = _synthetic_otm_slice(true_params, forward=forward, discount=discount, t=t)

    fit = calibrate_heston(slice_df, forward, discount, t)
    clean = slice_df[slice_df["iv"].notna()]

    # v0 and theta pin down the smile level/ATM curvature tightly; kappa/xi/rho
    # trade off against each other more, so allow looser tolerances there.
    assert fit.v0 == pytest.approx(true_params[0], abs=3e-3)
    assert fit.theta == pytest.approx(true_params[2], abs=5e-3)
    assert fit.rho == pytest.approx(true_params[4], abs=0.2)

    fitted_iv = fit.implied_vol_curve(forward, clean["strike"].to_numpy(),
                                       t, discount, clean["type"].to_numpy())
    assert np.allclose(fitted_iv, clean["iv"].to_numpy(), atol=5e-3)


def test_calibrate_heston_joint_recovers_shared_params_across_expiries():
    """A single kappa/xi are far better identified with two maturities at
    once than either slice alone -- this is the whole point of joint
    calibration, so check recovery is at least as tight as the single-slice
    test above, not just that it runs."""
    true_params = (0.02, 2.0, 0.025, 0.6, -0.4)  # v0, kappa, theta, xi, rho
    forward, discount = 25000.0, 0.999

    slice_near = _synthetic_otm_slice(true_params, forward=forward, discount=discount,
                                       t=0.03, seed=1)
    slice_far = _synthetic_otm_slice(true_params, forward=forward, discount=discount,
                                      t=0.25, seed=2)
    otm_slices = {
        ("NIFTY", "2026-10-01"): slice_near,
        ("NIFTY", "2026-12-20"): slice_far,
    }
    forwards_df = pd.DataFrame([
        {"underlying": "NIFTY", "expiry": "2026-10-01", "forward": forward,
         "discount": discount, "t": 0.03},
        {"underlying": "NIFTY", "expiry": "2026-12-20", "forward": forward,
         "discount": discount, "t": 0.25},
    ])

    fit = calibrate_heston_joint(otm_slices, forwards_df)

    assert fit.v0 == pytest.approx(true_params[0], abs=3e-3)
    assert fit.kappa == pytest.approx(true_params[1], abs=0.5)
    assert fit.theta == pytest.approx(true_params[2], abs=3e-3)
    assert fit.xi == pytest.approx(true_params[3], abs=0.3)
    assert fit.rho == pytest.approx(true_params[4], abs=0.15)
    expected_n = slice_near["iv"].notna().sum() + slice_far["iv"].notna().sum()
    assert fit.n_points == expected_n


def test_calibrate_heston_requires_minimum_points():
    tiny = pd.DataFrame({
        "strike": [100.0, 110.0], "type": ["CE", "PE"],
        "bid": [1.0, 1.0], "ask": [1.1, 1.1], "mid": [1.05, 1.05], "iv": [0.2, 0.2],
    })
    with pytest.raises(ValueError):
        calibrate_heston(tiny, forward=100.0, discount=0.999, t=0.05)
