"""Randfälle (DEMO-PLAYBOOK §12): n = 1, n = 2 (Zweige `n < 2` greifen nicht), Gleichstand, inklusive Grenzen, Regler am Rand."""

import numpy as np
import pytest

import inl_constants as C
import inl_evaluation as V
import inl_rank as RK
import inl_scenario as S
import inl_simulate as SIM
import inl_unknown as UN
import inl_values as VA


def test_exactly_two_offers():
    """n = 2: ohne Werte 1/2 (nehmen oder wechseln ist gleich gut), mit bekannter Verteilung 3/4; der erwartete Rang ist 3/2, der erwartete Wert 5/8 bei Cayley."""
    rows = {r.key: r for r in V.ladder(2)}
    assert rows["rang"].treffer == pytest.approx(0.5) and rows["rang"].rang == pytest.approx(1.5)
    assert rows["werte"].treffer == pytest.approx(0.75, abs=1e-9) and rows["werte"].wert == pytest.approx(0.625)
    assert rows["zufall"].treffer == pytest.approx(0.5) and rows["hellsicht"].wert == pytest.approx(2 / 3)


def test_single_offer():
    """n = 1: das einzige Angebot wird genommen, alles ist 1 (Treffer, Rang), der Wert ist der Erwartungswert 1/2."""
    rows = {r.key: r for r in V.ladder(1)}
    for key in ("zufall", "unbekannt", "rang", "werte", "hellsicht"):
        assert rows[key].treffer == pytest.approx(1.0, abs=1e-9), key
    assert rows["rang"].rang == pytest.approx(1.0) and rows["rang"].wert == pytest.approx(0.5)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("goal", C.GOALS)
def test_analysis_runs_for_tiny_days(n, goal):
    a = V.analyse(S.Settings(n=n, goal=goal, wait=100, seed=1))
    assert a.skip_own == n - 1
    assert 0 <= a.picks["hellsicht"] < n
    cross = V.simulate_cross(a)
    assert set(cross) == {k for k, _, _ in V.CROSS_RULES}


def test_ties_pick_the_first_of_equal_values():
    """Gleiche Werte: ein Angebot ist nur Rekord, wenn es ALLE früheren echt übertrifft; Hellsicht nimmt das erste Maximum."""
    vals = np.array([[0.5, 0.5, 0.5]])
    assert SIM.records(vals).tolist() == [[True, False, False]]
    assert SIM.pick_hindsight(vals).tolist() == [0]


def test_waiting_time_bounds_are_inclusive():
    assert S.skip_count(100, 0) == 0 and S.skip_count(100, 100) == 99          # 100 Prozent: alle bis auf das letzte
    assert S.skip_count(2, 50) == 1 and S.skip_count(2, 49) == 0 and S.skip_count(100, 37) == 37 and S.skip_count(10, 99) == 9
    assert S.skip_count(200, C.DEFAULT_WAIT) == 74


def test_waiting_time_is_never_a_floating_point_rounding_artifact():
    for n in range(2, 201):
        for pct in range(0, 101):
            assert S.skip_count(n, pct) == min(n - 1, (n * pct) // 100)


def test_unknown_n_with_one_possible_length():
    p, t = UN.optimal_rule(1)
    assert p == pytest.approx(1.0) and t == 1
    assert UN.win_given_n(1, 1) == pytest.approx(1.0)


def test_largest_day_runs_within_a_generous_timeout():
    """Worst-Case am Slider-Maximum (DEMO-PLAYBOOK §1c): Hauptansicht und Simulation für n = 200 laufen durch; geprüft wird nur, dass sie enden und plausibel sind."""
    a = V.analyse(S.Settings(n=C.N_MAX, goal="treffer"))
    sim = V.simulate_cross(a)
    assert 0.3 < sim["rang_treffer"]["treffer"][0] < 0.45 and 0.5 < sim["werte_treffer"]["treffer"][0] < 0.65


def test_value_thresholds_at_the_slider_maximum_are_finite_and_in_range():
    b = VA.win_thresholds(C.N_MAX)
    assert np.all(np.isfinite(b)) and b[0] == 0.0 and np.all((b[1:] > 0.4) & (b[1:] < 1.0))
    s = RK.expected_rank_dp(C.N_MAX)[1]
    assert s[C.N_MAX] == C.N_MAX and np.all(s[1:C.N_MAX] < C.N_MAX)
