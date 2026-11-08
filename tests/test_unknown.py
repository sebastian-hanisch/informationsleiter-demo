"""Stufe „Anzahl unbekannt“ (inl_unknown): Einheiten an Handrechnungen."""

import numpy as np
import pytest

import inl_unknown as UN


def test_harmonic_numbers():
    assert UN.harmonic(4).tolist() == pytest.approx([0.0, 1.0, 1.5, 11 / 6, 25 / 12])


def test_one_and_two_offers_by_hand():
    """m = 1: das einzige Angebot nehmen, 1. m = 2: N = 1 oder 2 gleich wahrscheinlich; das erste Angebot nehmen gewinnt bei N = 1 immer, bei N = 2 mit 1/2: 3/4.
    Das erste ablehnen und den zweiten Rekord nehmen: nur bei N = 2 und Wahrscheinlichkeit 1/2 ein Rekord: 1/4."""
    assert UN.optimal_rule(1) == (pytest.approx(1.0), 1)
    p, t = UN.optimal_rule(2)
    assert p == pytest.approx(0.75) and t == 1


def test_win_given_n_by_hand():
    """Ab Zeitpunkt 2 den ersten Rekord, N = 3: 1/3 · (1 + 1/2) = 1/2 (Angebot 2 mit Wahrscheinlichkeit 1, Angebot 3 mit 1/2 noch nicht genommen). Ab 1: 1/N. Vor dem Einsatzzeitpunkt zu Ende: 0."""
    assert UN.win_given_n(2, 3) == pytest.approx(0.5)
    assert UN.win_given_n(1, 7) == pytest.approx(1 / 7)
    assert UN.win_given_n(5, 4) == 0.0


def test_win_given_n_matches_the_set_form_of_the_rule():
    """Zweite Rechenart: Schleife über die Zeitpunkte mit der Wahrscheinlichkeit, noch nicht genommen zu haben."""
    for stop_from, n in ((2, 9), (4, 30), (7, 7), (3, 3)):
        alive, w = 1.0, 0.0
        for t in range(1, n + 1):
            if t >= stop_from:
                w += alive * (1 / t) * (t / n)
                alive *= 1 - 1 / t
        assert UN.win_given_n(stop_from, n) == pytest.approx(w, abs=1e-12)


def test_optimal_rule_equals_its_own_evaluation_over_the_uniform_length():
    for m in (3, 10, 40):
        p, t = UN.optimal_rule(m)
        assert np.mean([UN.win_given_n(t, N) for N in range(1, m + 1)]) == pytest.approx(p, abs=1e-12)


def test_the_stopping_set_is_a_tail_for_every_m_up_to_200():
    """Prüfung der Form (nicht Beweis): die Menge der Zeitpunkte, zu denen ein Rekord genommen wird, ist {t0, …, m}."""
    for m in range(1, 201):
        _, stops = UN.stop_set(m)
        assert stops == list(range(stops[0], m + 1)), m


def test_win_curve_has_one_value_per_length():
    curve = UN.win_curve(5, 12)
    assert len(curve) == 12 and curve[:4].tolist() == [0.0, 0.0, 0.0, 0.0]
