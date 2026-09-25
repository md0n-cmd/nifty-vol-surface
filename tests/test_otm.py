import pandas as pd
import pytest

from vol_surface.clean import clean_chain
from vol_surface.forward import forwards_by_expiry
from vol_surface.otm import select_otm
from tests.conftest import make_chain


def test_select_otm_keeps_one_side_per_strike(known_svi):
    chain = make_chain(forward=25000.0, svi=known_svi)
    cleaned = clean_chain(chain)
    forwards = forwards_by_expiry(cleaned)

    otm = select_otm(cleaned, forwards)

    forward = forwards["forward"].iloc[0]
    otm_calls = otm[otm["type"] == "CE"]
    otm_puts = otm[otm["type"] == "PE"]
    assert (otm_calls["strike"] >= forward).all()
    assert (otm_puts["strike"] <= forward).all()

    # every strike below the fixture's forward should have dropped its call leg,
    # every strike above it should have dropped its put leg
    below = cleaned[cleaned["strike"] < forward]
    above = cleaned[cleaned["strike"] > forward]
    assert not ((otm["strike"].isin(below["strike"])) & (otm["type"] == "CE")).any()
    assert not ((otm["strike"].isin(above["strike"])) & (otm["type"] == "PE")).any()


def test_select_otm_keeps_both_sides_at_exact_atm(known_svi):
    chain = make_chain(forward=25000.0, svi=known_svi, strike_step=100.0)
    cleaned = clean_chain(chain)
    forward = 25000.0
    forwards = pd.DataFrame([{
        "underlying": "NIFTY", "expiry": cleaned["expiry"].iloc[0],
        "forward": forward, "discount": 0.9995, "t": 0.05, "n_strikes": 10,
    }])

    otm = select_otm(cleaned, forwards)

    atm = otm[otm["strike"] == forward]
    assert set(atm["type"]) == {"CE", "PE"}


def test_select_otm_drops_the_itm_side_entirely(known_svi):
    chain = make_chain(forward=25000.0, svi=known_svi)
    cleaned = clean_chain(chain)
    forwards = forwards_by_expiry(cleaned)

    otm = select_otm(cleaned, forwards)

    # every non-ATM strike loses exactly one leg, so roughly half of the
    # cleaned CE+PE rows survive
    assert 0 < len(otm) < len(cleaned)
    assert len(otm) == pytest.approx(len(cleaned) / 2, rel=0.15)
