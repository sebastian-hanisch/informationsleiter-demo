"""Stufe „Werte bekannt“ (inl_values): Einheiten an Handrechnungen und exakten kleinen Fällen."""

import math

import numpy as np
import pytest

import inl_values as VA


def test_cayley_values_by_hand():
    """v1 = 1/2 (ein Angebot kommt noch: Mittelwert), v2 = (1 + 1/4)/2 = 5/8, v3 = (1 + 25/64)/2 = 89/128."""
    v = VA.cayley_values(3)
    assert v.tolist() == pytest.approx([0.0, 0.5, 0.625, 89 / 128])


def test_cayley_values_increase_and_stay_below_one():
    v = VA.cayley_values(200)
    assert np.all(np.diff(v) > 0) and v[-1] < 1.0


def test_win_thresholds_by_hand():
    """b1 = 1/2 (Nehmen x gegen Weitermachen 1 − x); b2 ist die positive Wurzel von 5x² − 2x − 1 = 0, also (1 + √6)/5."""
    b = VA.win_thresholds(3)
    assert b[0] == 0.0 and b[1] == pytest.approx(0.5) and b[2] == pytest.approx((1 + math.sqrt(6)) / 5)


def test_win_thresholds_solve_their_equation_and_rise_with_the_remaining_offers():
    b = VA.win_thresholds(60)
    for k in (1, 5, 30, 59):
        assert sum((b[k] ** -i - 1) / i for i in range(1, k + 1)) == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(b[1:]) > 0)                         # je mehr Angebote kommen, desto wählerischer


def test_full_info_win_for_one_two_and_three_offers():
    """n = 1: 1. n = 2: 3/4 (nehmen, wenn x >= 1/2: 1/2 · 1/2 + ∫_{1/2}^1 x dx + ... exakt 3/4). n = 3: Integral von max(x², f_2(x)) mit exakt integrierter Wertfunktion f_2."""
    assert VA.full_info_win(1, 2000) == pytest.approx(1.0)
    assert VA.full_info_win(2, 4000) == pytest.approx(0.75, abs=1e-9)
    # n = 3 von Hand: f_2(m) = m(1 − m) + ∫_m^1 max(x, 1 − x) dx; Schnittstelle von x² und f_2 bei b_2
    N = 400_000
    x = (np.arange(N) + 0.5) / N
    tail = np.where(x >= 0.5, (1 - x ** 2) / 2, (0.5 - x) - (0.25 - x ** 2) / 2 + 0.375)
    f2 = x * (1 - x) + tail
    exact = float(np.maximum(x ** 2, f2).mean())
    assert VA.full_info_win(3, 8000) == pytest.approx(exact, abs=1e-6)
    assert exact == pytest.approx(0.684293, abs=1e-6)


def test_full_info_win_decreases_with_n_and_refines_with_the_grid():
    wins = [VA.full_info_win(n, 4000) for n in range(1, 12)]
    assert all(x > y for x, y in zip(wins, wins[1:]))
    assert abs(VA.full_info_win(40, 4000) - VA.full_info_win(40, 16000)) < 1e-5
