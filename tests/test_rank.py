"""Stufe „nur Rangordnung“ (inl_rank): Einheiten an Handrechnungen."""

import numpy as np
import pytest

import inl_rank as RK


def test_skip_policy_marks_the_records_after_the_waiting_time():
    s = RK.skip_policy(5, 2)
    assert s.tolist() == [0, 0, 0, 1, 1, 5]                    # ab Angebot 3 nur Relativrang 1; das letzte Angebot wird erzwungen
    assert RK.skip_policy(5, 0).tolist() == [0, 1, 1, 1, 1, 5]
    assert RK.skip_policy(5, 4).tolist() == [0, 0, 0, 0, 0, 5]


def test_skip_policy_rejects_waiting_times_outside_the_range():
    for skip in (-1, 5):
        with pytest.raises(ValueError):
            RK.skip_policy(5, skip)


def test_classic_win_by_hand():
    """n = 3, skip = 1: 1/3 · (1/1 + 1/2) = 1/2. n = 4, skip = 1: 1/4 · (1 + 1/2 + 1/3) = 11/24. skip = 0: 1/n."""
    assert RK.classic_win(3, 1) == pytest.approx(0.5)
    assert RK.classic_win(4, 1) == pytest.approx(11 / 24)
    assert RK.classic_win(4, 0) == pytest.approx(0.25)


def test_evaluate_policy_agrees_with_the_closed_formula_and_stops_with_probability_one():
    for n in (2, 3, 7, 30):
        for skip in range(n):
            p, e = RK.evaluate_policy(n, RK.skip_policy(n, skip))
            assert p == pytest.approx(RK.classic_win(n, skip), abs=1e-12)
            assert 1.0 <= e <= n


def test_evaluate_policy_for_the_first_offer_is_random_choice():
    """skip = 0 nimmt das erste Angebot: Treffer 1/n, erwarteter Rang (n + 1)/2."""
    p, e = RK.evaluate_policy(10, RK.skip_policy(10, 0))
    assert p == pytest.approx(0.1) and e == pytest.approx(5.5)


def test_expected_rank_dp_by_hand_for_n_3():
    """n = 3: W_2 = 2 (letzte Wahl, Mittel von 1..3), Stoppen bei Relativrang j ab Zeit 2 kostet j · 4/3; Mittel von min(4/3, 2), min(8/3, 2) = (4/3 + 2)/2 = 5/3."""
    w, s = RK.expected_rank_dp(3)
    assert w == pytest.approx(5 / 3)
    assert s.tolist() == [0, 0, 1, 3]                           # Zeit 1: nicht nehmen; Zeit 2: nur Relativrang 1; Zeit 3: alles


def test_expected_rank_dp_matches_the_exact_evaluation_of_its_own_policy():
    for n in (2, 5, 12, 40):
        w, s = RK.expected_rank_dp(n)
        assert RK.evaluate_policy(n, s)[1] == pytest.approx(w, rel=1e-12)


def test_expected_rank_dp_is_at_least_as_good_as_every_waiting_time_rule():
    for n in (3, 10, 25):
        w, _ = RK.expected_rank_dp(n)
        assert w <= RK.skip_curve(n)["rang"].min() + 1e-12


def test_value_from_rank_is_the_mean_of_the_order_statistic():
    assert RK.value_from_rank(1.0, 1) == pytest.approx(0.5)          # ein Angebot: Erwartungswert 1/2
    assert RK.value_from_rank(1.0, 3) == pytest.approx(0.75)         # Maximum von drei Gleichverteilten: 3/4
    assert RK.value_from_rank(3.0, 3) == pytest.approx(0.25)


def test_best_skip_prefers_the_longer_waiting_time_on_ties():
    assert RK.best_skip(np.array([0.5, 0.5]), True) == 1
    assert RK.best_skip(np.array([2.0, 1.0, 1.0]), False) == 2
    assert RK.best_skip(np.array([0.1, 0.9, 0.3]), True) == 1


def test_best_classic_curve_equals_the_maximum_of_the_skip_curve():
    curve = RK.best_classic_curve(40)
    for n in range(1, 41):
        assert curve[n - 1] == pytest.approx(RK.skip_curve(n)["treffer"].max(), abs=1e-12)
