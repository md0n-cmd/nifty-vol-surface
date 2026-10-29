import pytest

from heston.barrier_ref import (bgk_barrier, bs_call, down_out_call_continuous,
                                 down_out_call_discrete_bgk)

F, D, T, SIGMA = 56022.49113, 0.987123, 0.088173, 0.14


def test_down_out_call_tends_to_vanilla_as_barrier_goes_to_zero():
    vanilla = bs_call(F, F, T, SIGMA, D)
    assert down_out_call_continuous(F, F, 1e-6 * F, T, SIGMA, D) == pytest.approx(vanilla, rel=1e-9)


def test_down_out_call_is_zero_when_barrier_equals_forward():
    assert down_out_call_continuous(F, F, F, T, SIGMA, D) == pytest.approx(0.0, abs=1e-8)


def test_down_out_price_rises_as_barrier_moves_away_and_discrete_exceeds_continuous():
    prices = [down_out_call_continuous(F, F, F * (1 - b), T, SIGMA, D) for b in (0.03, 0.05, 0.10)]
    assert prices[0] < prices[1] < prices[2] < bs_call(F, F, T, SIGMA, D)
    assert bgk_barrier(F * 0.95, SIGMA, T, 50) < F * 0.95
    assert (down_out_call_discrete_bgk(F, F, F * 0.95, T, SIGMA, D, 50)
            > down_out_call_continuous(F, F, F * 0.95, T, SIGMA, D))
