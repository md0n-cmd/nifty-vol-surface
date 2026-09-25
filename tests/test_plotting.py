from vol_surface.plotting import plot_svi_params
from vol_surface.svi import SVIFit


def test_plot_svi_params_writes_file(tmp_path):
    fits = {
        ("NIFTY", "2026-09-29"): SVIFit(a=0.0, b=0.02, rho=0.7, m=0.05, sigma=0.01, n_points=168, cost=1e-6),
        ("NIFTY", "2026-10-06"): SVIFit(a=0.0, b=0.015, rho=0.0, m=0.01, sigma=0.02, n_points=133, cost=5e-7),
        ("BANKNIFTY", "2026-09-29"): SVIFit(a=0.0, b=0.036, rho=0.89, m=0.08, sigma=0.02, n_points=149, cost=4e-6),
        ("BANKNIFTY", "2026-10-27"): SVIFit(a=0.0, b=0.026, rho=0.15, m=0.04, sigma=0.05, n_points=122, cost=6e-7),
    }

    out_path = plot_svi_params(fits, tmp_path / "params.png")

    assert out_path.exists()
    assert out_path.stat().st_size > 0
