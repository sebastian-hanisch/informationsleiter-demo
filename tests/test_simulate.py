"""Regeln auf konkreten Folgen (inl_simulate): jede Einheit an der Mini-Folge von Hand."""

import numpy as np
import pytest

import inl_rank as RK
import inl_simulate as SIM


def test_records_by_hand(mini_day):
    assert SIM.records(mini_day).tolist() == [[True, True, False, True, False]]


def test_first_true_and_fallback():
    mask = np.array([[False, True, True], [False, False, False]])
    assert SIM.first_true(mask, 2).tolist() == [1, 2]
    assert SIM.first_true(mask, -1).tolist() == [1, -1]


def test_relative_ranks_by_hand(mini_day):
    assert SIM.relative_ranks(mini_day).tolist() == [[1, 1, 2, 1, 5]]


def test_pick_skip_by_hand(mini_day):
    assert SIM.pick_skip(mini_day, 0).tolist() == [0]            # erstes Angebot
    assert SIM.pick_skip(mini_day, 1).tolist() == [1]            # Angebot 2 ist Rekord
    assert SIM.pick_skip(mini_day, 2).tolist() == [3]            # Angebot 3 ist kein Rekord, Angebot 4 schon
    assert SIM.pick_skip(mini_day, 4).tolist() == [4]            # kein Rekord mehr: das letzte Angebot


def test_pick_rank_policy_equals_pick_skip_for_the_skip_policy(mini_day):
    rng = np.random.default_rng(1)
    vals = rng.random((50, 12))
    for skip in (0, 3, 11):
        assert SIM.pick_rank_policy(vals, RK.skip_policy(12, skip)).tolist() == SIM.pick_skip(vals, skip).tolist()


def test_pick_rank_policy_takes_second_best_when_allowed(mini_day):
    s = np.array([0, 0, 0, 2, 2, 5])                              # ab Angebot 3 bis Relativrang 2
    assert SIM.pick_rank_policy(mini_day, s).tolist() == [2]       # Angebot 3 hat Relativrang 2


def test_pick_value_win_and_cayley_by_hand(mini_day):
    b = np.array([0.0, 0.5, 0.69, 0.78, 0.84])                     # b[k], k = noch kommende
    # Angebot 1: 0,2 < b[4] = 0,84 ablehnen; Angebot 2: 0,9 ist Rekord und >= b[3] = 0,78: nehmen
    assert SIM.pick_value_win(mini_day, b).tolist() == [1]
    v = np.array([0.0, 0.5, 0.625, 0.7, 0.75])
    # Angebot 1: 0,2 < v[4]; Angebot 2: 0,9 >= v[3] = 0,7: nehmen
    assert SIM.pick_cayley(mini_day, v).tolist() == [1]
    late = np.array([[0.2, 0.3, 0.4, 0.1, 0.35]])
    assert SIM.pick_cayley(late, v).tolist() == [4]                # nichts über den Schwellen: das letzte (v[0] = 0)


def test_pick_unknown_returns_minus_one_without_a_record_after_the_start(mini_day):
    assert SIM.pick_unknown(mini_day, 3).tolist() == [3]           # ab Angebot 3 der erste Rekord ist Angebot 4
    assert SIM.pick_unknown(mini_day, 5).tolist() == [-1]          # Angebot 5 ist kein Rekord: keine Wahl


def test_hindsight_and_random_picks(mini_day):
    assert SIM.pick_hindsight(mini_day).tolist() == [3]
    assert SIM.pick_random(5, np.array([0.0, 0.19, 0.2, 0.999999])).tolist() == [0, 0, 1, 4]


def test_score_rank_value_and_no_choice(mini_day):
    sc = SIM.score(mini_day, np.array([1]))
    assert sc["rang"].tolist() == [2.0] and sc["wert"].tolist() == [0.9] and sc["treffer"].tolist() == [0.0]
    best = SIM.score(mini_day, np.array([3]))
    assert best["treffer"].tolist() == [1.0] and best["rang"].tolist() == [1.0]
    none = SIM.score(mini_day, np.array([-1]))
    assert none["treffer"].tolist() == [0.0] and np.isnan(none["rang"][0]) and np.isnan(none["wert"][0])


def test_summarize_counts_runs_with_a_choice():
    sc = {"treffer": np.array([1.0, 0.0, 0.0]), "rang": np.array([1.0, np.nan, 3.0]), "wert": np.array([0.9, np.nan, 0.5])}
    s = SIM.summarize(sc)
    assert s["treffer"][0] == pytest.approx(1 / 3) and s["treffer"][2] == 3
    assert s["rang"][0] == pytest.approx(2.0) and s["rang"][2] == 2


def test_unknown_days_ignore_offers_after_the_end(mini_day):
    days = SIM.unknown_n_days(mini_day, np.array([3]))
    assert np.isneginf(days[0, 3:]).all()
    assert SIM.pick_unknown(days, 3).tolist() == [-1]              # bis Angebot 3 kein Rekord ab Zeitpunkt 3
    assert SIM.simulate_unknown_days(mini_day, np.array([5]), 3) == 1.0
