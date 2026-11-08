"""Umschlag-Spiel (inl_envelopes): Formel gegen Handwert und Simulation."""

import math

import pytest

import inl_envelopes as EN


def test_win_probability_by_hand():
    """a = 1, b = 2, Mittel 10: 1/2 + 1/2 (e^−0,1 − e^−0,2) = 0,5431."""
    assert EN.win_probability(1, 2, 10) == pytest.approx(0.5 + 0.5 * (math.exp(-0.1) - math.exp(-0.2)))
    assert EN.win_probability(1, 2, 10) == pytest.approx(0.5431, abs=5e-5)


def test_equal_amounts_give_a_coin_flip_and_bad_input_is_rejected():
    assert EN.win_probability(5, 5, 10) == 0.5
    for a, b in ((0, 1), (-1, 1), (3, 2)):
        with pytest.raises(ValueError):
            EN.win_probability(a, b, 10)


def test_excess_is_positive_even_when_the_probability_rounds_to_one_half():
    assert EN.win_probability(300, 3000, 5) == 0.5                 # auf Gleitkomma gerundet
    assert 0 < EN.excess(300, 3000, 5) < 1e-20


def test_simulation_agrees_with_the_formula_within_four_standard_errors():
    runs = 200_000
    for a, b, scale in ((1, 2, 10), (5, 15, 10), (10, 20, 50)):
        p = EN.win_probability(a, b, scale)
        sim = EN.simulate(a, b, scale, runs, seed=11)
        assert abs(sim - p) < 4 * math.sqrt(p * (1 - p) / runs), (a, b, scale)


def test_simulation_is_reproducible():
    assert EN.simulate(1, 2, 10, 5000, 3) == EN.simulate(1, 2, 10, 5000, 3)
