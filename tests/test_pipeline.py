from vol_surface import pipeline
from tests.conftest import make_chain


def _write_date_partition(tmp_path, chain, underlying, expiry, ts):
    day_dir = tmp_path / f"date={ts.date()}"
    day_dir.mkdir(parents=True, exist_ok=True)
    out = day_dir / f"{underlying}_{expiry}_{ts:%H%M%S}.parquet"
    chain.to_parquet(out, index=False)
    return day_dir


def test_pipeline_runs_end_to_end(tmp_path, known_svi):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"

    chain = make_chain(svi=known_svi)
    ts, expiry, underlying = chain["ts"].iloc[0], chain["expiry"].iloc[0], chain["underlying"].iloc[0]
    _write_date_partition(data_dir, chain, underlying, expiry, ts)

    result = pipeline.run(data_dir, out_dir)

    assert ("NIFTY", expiry) in result["fits"]
    assert (out_dir / f"date={ts.date()}_surface.parquet").exists()
    assert (out_dir / f"date={ts.date()}_svi_params.json").exists()
    assert len(result["plots"]) == 1
    assert result["plots"][0].exists()
