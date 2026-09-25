import pandas as pd
import pytest

from vol_surface.clean import clean_chain
from vol_surface.forward import forwards_by_expiry, parity_forward
from tests.conftest import make_chain


def test_parity_forward_recovers_known_forward_and_discount(known_svi):
    true_forward, true_discount = 25000.0, 0.9992
    chain = make_chain(forward=true_forward, discount=true_discount, svi=known_svi)
    cleaned = clean_chain(chain)

    result = parity_forward(cleaned)

    assert result["forward"] == pytest.approx(true_forward, rel=1e-3)
    assert result["discount"] == pytest.approx(true_discount, rel=1e-3)
    assert result["n_strikes"] > 0


def test_forwards_by_expiry_covers_all_groups(known_svi):
    nifty = make_chain(forward=25000.0, underlying="NIFTY", svi=known_svi)
    banknifty = make_chain(forward=52000.0, underlying="BANKNIFTY", svi=known_svi)
    cleaned = clean_chain(pd.concat([nifty, banknifty], ignore_index=True))

    forwards = forwards_by_expiry(cleaned)

    assert set(forwards["underlying"]) == {"NIFTY", "BANKNIFTY"}
    assert (forwards["forward"] > 0).all()


def test_parity_forward_requires_paired_strikes(known_svi):
    chain = make_chain(svi=known_svi)
    calls_only = chain[chain["type"] == "CE"]
    cleaned = clean_chain(calls_only)

    with pytest.raises(ValueError):
        parity_forward(cleaned)
