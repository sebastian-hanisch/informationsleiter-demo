"""Eigenschaftstests: Strukturen, die für alle n und alle Ziele gelten müssen (nicht an einzelnen Zahlen festgemacht)."""

import numpy as np
import pytest

import inl_constants as C
import inl_evaluation as V
import inl_rank as RK
import inl_scenario as S
import inl_simulate as SIM
import inl_unknown as UN
import inl_values as VA


@pytest.mark.parametrize("n", [2, 3, 5, 12, 40])
def test_the_optimal_rank_rule_is_not_worse_than_any_other_rule_for_rank_and_value(n):
    w, s = RK.expected_rank_dp(n)
    curve = RK.skip_curve(n)
    assert w <= curve["rang"].min() + 1e-12
    assert RK.value_from_rank(w, n) >= curve["wert"].max() - 1e-12


@pytest.mark.parametrize("n", [2, 3, 5, 12, 40])
def test_the_classic_rule_has_the_highest_win_probability_among_rank_rules(n):
    """Die Regel fürs höchste Angebot schlägt die Rang-Regel am Ziel Treffer, die Rang-Regel schlägt sie am Ziel Rang (gleiche Information, Ziel ändert die Regel)."""
    best = RK.skip_policy(n, RK.best_skip(RK.skip_curve(n)["treffer"], True))
    w, s = RK.expected_rank_dp(n)
    p_best, r_best = RK.evaluate_policy(n, best)
    p_rank, r_rank = RK.evaluate_policy(n, s)
    assert p_best >= p_rank - 1e-12 and r_rank <= r_best + 1e-12


def test_wait_time_curve_has_a_single_interior_peak_for_the_hit_goal():
    """Die Trefferkurve steigt bis zur besten Wartezeit und fällt danach (Form, Zählung über n = 5 bis 60)."""
    for n in range(5, 61):
        y = RK.skip_curve(n)["treffer"]
        k = int(np.argmax(y))
        assert np.all(np.diff(y[:k + 1]) >= -1e-12) and np.all(np.diff(y[k:]) <= 1e-12), n


def test_ladder_orders_for_every_n_up_to_30():
    for n in range(2, 31):
        rows = {r.key: r for r in V.ladder(n)}
        assert rows["hellsicht"].treffer >= rows["werte"].treffer >= rows["rang"].treffer >= rows["zufall"].treffer - 1e-12, n
        assert rows["hellsicht"].wert >= rows["werte"].wert >= rows["rang"].wert >= rows["zufall"].wert - 1e-12, n
        assert rows["rang"].rang <= rows["zufall"].rang + 1e-12, n


def test_unknown_n_rule_loses_against_knowing_n_on_the_same_days():
    """Der faire Vergleich (gleiche Tage, N gleichverteilt): unbekanntes N ist nie besser als bekanntes N, für alle m bis 120."""
    known = RK.best_classic_curve(120)
    for m in range(2, 121):
        assert UN.optimal_rule(m)[0] <= known[:m].mean() + 1e-12, m


def test_value_thresholds_fall_with_fewer_remaining_offers_and_cayley_stays_below_one():
    b = VA.win_thresholds(100)
    v = VA.cayley_values(100)
    assert np.all(np.diff(b[1:]) > 0) and np.all(v < 1) and np.all(b < 1)
    assert np.all(b[1:] >= 0.5 - 1e-12)                       # schon bei einem kommenden Angebot muss ein Rekord mindestens 1/2 sein


def test_rank_policy_thresholds_never_decrease_in_time():
    """Je später, desto großzügiger (Zählung über n = 3 bis 100): der größte genommene Relativrang wird nie kleiner."""
    for n in range(3, 101):
        s = RK.expected_rank_dp(n)[1][1:]
        assert np.all(np.diff(s) >= 0), n


def test_same_seed_same_day_and_different_seed_different_day():
    a1 = V.analyse(S.Settings(seed=5))
    a2 = V.analyse(S.Settings(seed=5))
    a3 = V.analyse(S.Settings(seed=6))
    assert np.array_equal(a1.values, a2.values) and not np.array_equal(a1.values, a3.values)
    assert a1.picks == a2.picks


def test_hindsight_is_never_beaten_on_the_same_days():
    """Hellsicht dominiert jede Regel Tag für Tag: ihr Treffer ist immer 1, ihr Wert mindestens der jeder anderen Regel."""
    a = V.analyse(S.Settings(n=30))
    vals = S.make_values(3, 30, 500)
    hind = SIM.score(vals, SIM.pick_hindsight(vals))["wert"]
    for picker in (lambda v: SIM.pick_skip(v, 11), lambda v: SIM.pick_cayley(v, a.v), lambda v: SIM.pick_value_win(v, a.b), lambda v: SIM.pick_rank_policy(v, a.s_rank)):
        other = SIM.score(vals, picker(vals))["wert"]
        assert np.all(hind >= other)


def test_every_goal_has_a_label_and_a_unit():
    assert set(C.GOAL_LABELS) == set(C.GOALS) == set(V.GOAL_UNITS)
