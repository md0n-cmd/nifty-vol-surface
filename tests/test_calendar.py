from vol_surface.calendar import check_calendar_arbitrage
from vol_surface.svi import SVIFit


def _flat_fit(level, m=0.0, sigma=0.2):
    return SVIFit(a=level, b=0.0, rho=0.0, m=m, sigma=sigma, n_points=0, cost=0.0)


def test_passes_when_variance_grows_with_maturity():
    fits = {
        ("NIFTY", "2026-10-01"): _flat_fit(0.01),
        ("NIFTY", "2026-10-29"): _flat_fit(0.03),
    }
    t_by_key = {("NIFTY", "2026-10-01"): 0.02, ("NIFTY", "2026-10-29"): 0.10}

    assert check_calendar_arbitrage(fits, t_by_key) == []


def test_flags_variance_decreasing_with_maturity():
    fits = {
        ("NIFTY", "2026-10-01"): _flat_fit(0.05),
        ("NIFTY", "2026-10-29"): _flat_fit(0.01),
    }
    t_by_key = {("NIFTY", "2026-10-01"): 0.02, ("NIFTY", "2026-10-29"): 0.10}

    violations = check_calendar_arbitrage(fits, t_by_key)

    assert len(violations) > 0
    assert all(v["underlying"] == "NIFTY" for v in violations)
    assert all(v["w_far"] < v["w_near"] for v in violations)


def test_independent_across_underlyings():
    fits = {
        ("NIFTY", "2026-10-01"): _flat_fit(0.01),
        ("NIFTY", "2026-10-29"): _flat_fit(0.03),
        ("BANKNIFTY", "2026-10-01"): _flat_fit(0.05),
        ("BANKNIFTY", "2026-10-29"): _flat_fit(0.01),
    }
    t_by_key = {
        ("NIFTY", "2026-10-01"): 0.02, ("NIFTY", "2026-10-29"): 0.10,
        ("BANKNIFTY", "2026-10-01"): 0.02, ("BANKNIFTY", "2026-10-29"): 0.10,
    }

    violations = check_calendar_arbitrage(fits, t_by_key)

    assert all(v["underlying"] == "BANKNIFTY" for v in violations)
    assert len(violations) > 0
