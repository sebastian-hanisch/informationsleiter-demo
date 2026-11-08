"""Orakel-Test informationsleiter-demo: jede Kernrechnung gegen einen ANDEREN Rechenweg (DEMO-PLAYBOOK §12).

* Rangordnung, Ziel Treffer: Formel gegen Aufzählung aller Reihenfolgen (n <= 7) und gegen das Optimum über ALLE Stopp-Mengen.
* Rangordnung, Ziel Rang: Rückwärts-DP gegen den Entscheidungsbaum über alle Reihenfolgen (n <= 7), ohne die Formel j (n + 1)/(k + 1); exakte Bewertung gegen Aufzählung.
* Werte bekannt: Raster-DP gegen die Schwellenregel in Simulation (Wurzeln einer Gleichung statt Raster) und gegen exakte kleine Fälle (in test_values.py).
* Ziel Wert: Rekursion von Cayley gegen Simulation; Wert aus Rang gegen Simulation.
* Anzahl unbekannt: DP gegen Aufzählung aller Stopp-Mengen und Reihenfolgen (m <= 6) und gegen Simulation.
Alle Läufe haben feste Zufalls-Seeds und eine Marge von vier Standardfehlern. Die Aufzählungen liegen nur hier, nicht im Kern."""

import itertools
import math

import numpy as np
import pytest

import inl_rank as RK
import inl_scenario as S
import inl_simulate as SIM
import inl_unknown as UN
import inl_values as VA


# ------------------------------------------------------------------ Aufzählungen (unabhängig vom Kern)
def record_times(perm):
    best, out = -1, []
    for t, x in enumerate(perm, start=1):
        if x > best:
            best = x
            out.append(t)
    return out


def best_over_all_stop_sets(Ns):
    """Beste Trefferwahrscheinlichkeit über alle Stopp-Mengen S (erster Rekord zu einer Zeit in S wird genommen); N gleichverteilt über die Liste Ns, alle Reihenfolgen aufgezählt."""
    m = max(Ns)
    stats = []
    for N in Ns:
        count = math.factorial(N)
        for perm in itertools.permutations(range(N)):
            rt = record_times(perm)
            stats.append((1 / len(Ns) / count, set(rt[:-1]), rt[-1]))      # Rekorde vor dem Maximum, Zeit des Maximums
    best = 0.0
    for mask in range(1, 1 << m):
        S_ = {t for t in range(1, m + 1) if mask >> (t - 1) & 1}
        best = max(best, sum(w for w, before, tmax in stats if tmax in S_ and not (S_ & before)))
    return best


def expected_rank_decision_tree(n):
    """Optimaler erwarteter Endrang als Entscheidungsbaum über die Relativrang-Folgen aller n! Reihenfolgen (Endrang von Angebot k im Mittel über die Reihenfolgen mit demselben Präfix)."""
    levels = [dict() for _ in range(n + 1)]
    for perm in itertools.permutations(range(n)):
        rel = tuple(1 + sum(1 for y in perm[:k] if y > perm[k]) for k in range(n))
        for k in range(1, n + 1):
            d = levels[k].setdefault(rel[:k], [0, 0])
            d[0] += n - perm[k - 1]                      # Endrang des k-ten Angebots (1 = bestes)
            d[1] += 1
    V = {pre: d[0] / d[1] for pre, d in levels[n].items()}
    for k in range(n - 1, 0, -1):
        cont = {}
        for pre, v in V.items():
            cont.setdefault(pre[:-1], []).append(v)
        V = {pre: min(d[0] / d[1], sum(cont[pre]) / len(cont[pre])) for pre, d in levels[k].items()}
    return V[(1,)]


def enumerate_policy(n, s):
    """(Trefferwahrscheinlichkeit, erwarteter Rang) einer Schwellenliste s durch Aufzählung aller Reihenfolgen."""
    wins = ranks = 0
    total = math.factorial(n)
    for perm in itertools.permutations(range(n)):
        for k in range(1, n + 1):
            rel = 1 + sum(1 for y in perm[:k - 1] if y > perm[k - 1])
            if rel <= s[k]:
                wins += perm[k - 1] == n - 1
                ranks += n - perm[k - 1]
                break
    return wins / total, ranks / total


# ------------------------------------------------------------------ Rangordnung
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_classic_rule_is_optimal_among_all_stop_sets_and_equals_the_formula(n):
    curve = RK.skip_curve(n)["treffer"]
    assert curve.max() == pytest.approx(best_over_all_stop_sets([n]), abs=1e-12)
    for skip in range(n):
        assert curve[skip] == pytest.approx(enumerate_policy(n, RK.skip_policy(n, skip))[0], abs=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_expected_rank_dp_equals_the_decision_tree_over_all_permutations(n):
    w, s = RK.expected_rank_dp(n)
    assert w == pytest.approx(expected_rank_decision_tree(n), abs=1e-12)
    p_enum, r_enum = enumerate_policy(n, s)
    assert r_enum == pytest.approx(w, abs=1e-12)
    assert RK.evaluate_policy(n, s)[0] == pytest.approx(p_enum, abs=1e-12)


def test_policy_evaluation_matches_the_enumeration_for_random_threshold_lists():
    rng = np.random.default_rng(5)
    for _ in range(12):
        n = int(rng.integers(2, 7))
        s = np.concatenate([[0], rng.integers(0, 3, n)])
        s[n] = n
        p, r = RK.evaluate_policy(n, s)
        pe, re = enumerate_policy(n, s)
        assert p == pytest.approx(pe, abs=1e-12) and r == pytest.approx(re, abs=1e-12)


def test_rank_dp_policy_in_simulation_for_n_100():
    """Zweiter Weg für n = 100: Simulation der DP-Regel auf 20.000 Folgen (feste Zufalls-Seeds) gegen die exakte Bewertung."""
    n = 100
    w, s = RK.expected_rank_dp(n)
    vals = S.make_values(11, n, 20000)
    sc = SIM.score(vals, SIM.pick_rank_policy(vals, s))
    mean_rank = sc["rang"].mean()
    se = sc["rang"].std(ddof=1) / math.sqrt(len(vals))
    assert abs(mean_rank - w) < 4 * se
    p_exact = RK.evaluate_policy(n, s)[0]
    assert abs(sc["treffer"].mean() - p_exact) < 4 * math.sqrt(p_exact * (1 - p_exact) / len(vals))


def test_value_of_a_rank_rule_is_one_minus_rank_over_n_plus_one_in_simulation():
    """Mit echten Werten gemessen (nicht über die Formel): der Mittelwert des genommenen Wertes."""
    n = 30
    w, s = RK.expected_rank_dp(n)
    vals = S.make_values(23, n, 30000)
    sc = SIM.score(vals, SIM.pick_rank_policy(vals, s))
    se = sc["wert"].std(ddof=1) / math.sqrt(len(vals))
    assert abs(sc["wert"].mean() - RK.value_from_rank(w, n)) < 4 * se


# ------------------------------------------------------------------ Werte bekannt
def scalar_threshold(k):
    """Schwelle b_k durch skalare Bisektion (anderer Code als die gleichzeitige Bisektion im Kern)."""
    lo, hi = 1e-9, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lhs = sum((mid ** -i - 1) / i for i in range(1, k + 1))
        lo, hi = (mid, hi) if lhs > 1 else (lo, mid)
    return (lo + hi) / 2


def test_thresholds_equal_the_scalar_bisection():
    b = VA.win_thresholds(25)
    for k in (1, 2, 5, 12, 24):
        assert b[k] == pytest.approx(scalar_threshold(k), abs=1e-9)


@pytest.mark.parametrize("n,runs", [(5, 60000), (10, 60000), (40, 30000)])
def test_raster_dp_equals_the_simulated_threshold_rule(n, runs):
    """Die Raster-DP kennt keine Schwellen; die Simulation benutzt die Wurzel-Schwellen. Beide müssen dieselbe Trefferwahrscheinlichkeit liefern."""
    p = VA.full_info_win(n, 8000)
    vals = S.make_values(31 + n, n, runs)
    sc = SIM.score(vals, SIM.pick_value_win(vals, VA.win_thresholds(n)))
    assert abs(sc["treffer"].mean() - p) < 4 * math.sqrt(p * (1 - p) / runs)


def test_value_threshold_rule_beats_other_thresholds_in_simulation():
    """Optimalität in der Nachbarschaft: dieselbe Folge, Schwellen um ± 0,05 verschoben, gewinnt nicht mehr (n = 10, gemeinsame Folgen, Marge 4 Standardfehler der Differenz)."""
    n, runs = 10, 60000
    vals = S.make_values(77, n, runs)
    b = VA.win_thresholds(n)
    base = SIM.score(vals, SIM.pick_value_win(vals, b))["treffer"]
    for shift in (-0.05, 0.05):
        shifted = np.clip(b + shift, 0.0, 1.0)
        shifted[0] = 0.0
        other = SIM.score(vals, SIM.pick_value_win(vals, shifted))["treffer"]
        d = base - other
        assert d.mean() > -4 * d.std(ddof=1) / math.sqrt(runs)


@pytest.mark.parametrize("n", [1, 2, 5, 10, 50])
def test_cayley_recursion_equals_the_simulated_rule(n):
    runs = 40000
    vals = S.make_values(5 + n, n, runs)
    sc = SIM.score(vals, SIM.pick_cayley(vals, VA.cayley_values(n)))
    se = sc["wert"].std(ddof=1) / math.sqrt(runs)
    assert abs(sc["wert"].mean() - VA.cayley_values(n)[n]) < 4 * se


def test_cayley_rule_beats_nearby_thresholds_in_simulation():
    n, runs = 10, 40000
    vals = S.make_values(99, n, runs)
    v = VA.cayley_values(n)
    base = SIM.score(vals, SIM.pick_cayley(vals, v))["wert"]
    for shift in (-0.05, 0.05):
        shifted = np.clip(v + shift, 0.0, 1.0)
        shifted[0] = 0.0
        other = SIM.score(vals, SIM.pick_cayley(vals, shifted))["wert"]
        d = base - other
        assert d.mean() > -4 * d.std(ddof=1) / math.sqrt(runs)


# ------------------------------------------------------------------ Anzahl unbekannt
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_unknown_n_dp_equals_the_best_stop_set_over_all_permutations(m):
    p, _ = UN.optimal_rule(m)
    assert p == pytest.approx(best_over_all_stop_sets(list(range(1, m + 1))), abs=1e-12)


def test_unknown_n_rule_in_simulation_for_m_100():
    m, runs = 100, 40000
    p, t = UN.optimal_rule(m)
    vals = S.make_values(41, m, runs)
    lengths = 1 + (np.random.default_rng(3).integers(0, m, runs))
    sim = SIM.simulate_unknown_days(vals, lengths, t)
    assert abs(sim - p) < 4 * math.sqrt(p * (1 - p) / runs)


def test_unknown_n_rule_is_best_among_nearby_starting_times_in_simulation():
    """Nachbarschaft: ab t − 3 und ab t + 3 gewinnen nicht mehr (m = 100, gleiche Tage; exakte Werte als zweiter Weg zusätzlich)."""
    m = 100
    p, t = UN.optimal_rule(m)
    for other in (t - 3, t + 3):
        assert np.mean([UN.win_given_n(other, N) for N in range(1, m + 1)]) < p
