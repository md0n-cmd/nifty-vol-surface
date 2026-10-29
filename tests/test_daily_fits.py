import pandas as pd

import daily_fits as df_


def _idx(rows):
    return pd.DataFrame([(d, u, e, pd.Timestamp(f"{d} {t}"), f"{d}/{u}{e}{t}")
                         for d, u, e, t in rows],
                        columns=["date", "underlying", "expiry", "ts", "path"])


def test_nearest_snapshot_buckets_and_dte_filter():
    idx = _idx([
        ("2026-10-01", "NIFTY", "2026-10-06", "14:20:00"),
        ("2026-10-01", "NIFTY", "2026-10-06", "14:31:00"),
        ("2026-10-01", "NIFTY", "2026-10-13", "14:29:00"),
        ("2026-10-01", "NIFTY", "2026-10-20", "14:29:00"),
        ("2026-10-05", "NIFTY", "2026-10-06", "14:29:00"),   # DTE 1 -> dropped
        ("2026-10-05", "NIFTY", "2026-10-13", "14:29:00"),
    ])
    picked, skipped, dropped = df_.pick_snapshots(idx)
    assert skipped == []
    day1 = picked[picked["date"] == "2026-10-01"].set_index("expiry")
    assert day1.loc["2026-10-06", "ts"].strftime("%H:%M") == "14:31"
    assert day1.loc["2026-10-06", "bucket"] == "near" and day1.loc["2026-10-13", "bucket"] == "next"
    assert set(dropped["reason"]) == {"not near/next bucket", "DTE 1 < 2"}
    assert (picked["date"] == "2026-10-05").sum() == 1  # next keeps its label


def test_date_skipped_without_snapshot_near_1430_and_stale_excluded():
    idx = _idx([
        ("2026-09-25", "NIFTY", "2026-10-06", "11:30:00"),
        ("2026-10-02", "NIFTY", "2026-10-06", "14:30:00"),
    ])
    picked, skipped, _ = df_.pick_snapshots(idx)
    assert picked.empty and [d for d, _ in skipped] == ["2026-09-25"]


def test_bound_hits_flags_params_at_bounds():
    assert df_._bound_hits(["a", "b"], [0.0, 1.0], [0.0, 0.0], [5.0, 5.0]) == "a"
