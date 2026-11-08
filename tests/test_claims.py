"""Jede Zahl aus der README wird hier nachgerechnet (DEMO-PLAYBOOK §4, §11a) und im README-Text gesucht.

Exakte Zahlen (Rückwärts-DP, Formeln, Raster-DP) werden mit ihren Nachkommastellen aus der README verglichen; Simulationen haben feste Zufalls-Seeds (SplitMix64) und
stehen mit Standardfehler im Text. Die Zahlen der Befundtabelle tragen die Nummern H1 bis H10 der README."""

import math
from pathlib import Path

import pytest

import inl_constants as C
import inl_envelopes as EN
import inl_evaluation as V
import inl_rank as RK
import inl_scenario as S
import inl_unknown as UN
import inl_values as VA

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def d(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def p1(x):
    """Anteil in Prozent mit einer Nachkommastelle und Komma, wie im README."""
    return d(100 * x, 1) + " %"


def has(*texts):
    for t in texts:
        assert t in README, f"nicht im README: {t!r}"


# ------------------------------------------------------------------ H1
def test_h1_best_waiting_time_and_win_probability():
    for n, skip, text in ((10, 3, "39,9 %"), (100, 37, "37,1 %"), (200, 73, "36,9 %")):
        curve = RK.skip_curve(n)["treffer"]
        assert RK.best_skip(curve, True) == skip and p1(curve[skip]) == text
    assert p1(RK.best_classic_curve(1000)[-1]) == "36,8 %" and p1(1 / math.e) == "36,8 %"
    has("n = 10: 39,9 % bei 3 abgelehnten", "n = 100: **37,1 % bei 37 abgelehnten**", "n = 200: 36,9 % bei 73", "n = 1000: 36,8 %", "Grenzwert 1/e = 36,8 %")


# ------------------------------------------------------------------ H2
def test_h2_expected_rank():
    w100, _ = RK.expected_rank_dp(100)
    w1000, _ = RK.expected_rank_dp(1000)
    assert d(w100, 2) == "3,60" and d((100 + 1) / 2, 1) == "50,5" and d(w1000, 2) == "3,83"
    assert p1(RK.value_from_rank(w100, 100)) == "96,4 %"
    assert w1000 < 3.8695                                           # der Grenzwert (laut Suchtreffern) liegt über allen endlichen n
    has("n = 100: **3,60** (Zufall 50,5)", "n = 1000: 3,83", "3,8695", "96,4 % des Wertebereichs")


# ------------------------------------------------------------------ H3
def test_h3_the_goal_changes_the_rule():
    a = V.analyse(S.Settings())
    p_t, r_t, w_t = a.best_rule
    p_r, r_r, w_r = a.rank_rule
    assert (p1(p_t), d(r_t, 2), p1(w_t)) == ("37,1 %", "20,01", "80,2 %")
    assert (p1(p_r), d(r_r, 2), p1(w_r)) == ("26,3 %", "3,60", "96,4 %")
    first_take = next(k for k in range(1, 101) if a.s_rank[k] >= 1)
    assert first_take == 28 and a.s_rank[90] > 1 and a.s_rank[99] > a.s_rank[90]         # später auch zweit- und drittbeste bisherige
    has("Treffer 37,1 %, Rang **20,01**, Wert 80,2 %", "Treffer 26,3 %, Rang **3,60**, Wert 96,4 %", "erst ab Angebot 28")
    kind, text = V.verdict_goal_changes_rule(100, a.best_rule, a.rank_rule, 37, first_take)
    assert "20,01" in text and "26,3 %" in text and "3,60" in text


# ------------------------------------------------------------------ H4
def test_h4_values_known_win_probability():
    expected = {2: "75,0 %", 3: "68,4 %", 10: "60,9 %", 100: "58,3 %", 200: "58,2 %"}
    for n, text in expected.items():
        assert p1(VA.full_info_win(n, C.GRID)) == text, n
    big = VA.full_info_win(1000, 64000)                              # n = 1000 mit 64 000 Rasterpunkten (Fehler gegen 256 000 Punkte: unter 1e-5)
    assert 0.5803 < big < 0.5806 and p1(big) == "58,0 %"
    assert abs(VA.full_info_win(1000, 4000) - big) > 0.0015          # das 4000er Raster des Plans lag um +0,002 daneben (Korrektur 1)
    assert VA.full_info_win(100, 4000) == pytest.approx(0.5829, abs=6e-5) and abs(VA.full_info_win(100, 16000) - 0.5829) < 6e-5
    rows = {r.key: r for r in V.ladder(100)}
    assert d(100 * (rows["werte"].treffer - rows["rang"].treffer), 1) == "21,2"
    has("n = 2: 75,0 %", "n = 3: 68,4 %", "n = 10: 60,9 %", "n = 100: **58,3 %**", "n = 200: 58,2 %", "n = 1000: 58,0 %", "21,2 Prozentpunkte", "0,5804", "0,5824", "0,5829 statt 0,5830")


def test_the_raster_error_claim_at_n_200():
    assert abs(VA.full_info_win(200, 16000) - VA.full_info_win(200, 64000)) < 1e-5
    has("rund 4 · 10⁻⁶")


# ------------------------------------------------------------------ H5
def test_h5_value_goal():
    v = VA.cayley_values(100)
    assert d(VA.cayley_values(10)[10], 3) == "0,861" and p1(VA.cayley_values(10)[10]) == "86,1 %"
    rows = {r.key: r for r in V.ladder(100)}
    assert (p1(v[100]), p1(rows["rang"].wert), p1(rows["hellsicht"].wert)) == ("98,1 %", "96,4 %", "99,0 %")
    closed = (rows["werte"].wert - rows["rang"].wert) / (rows["hellsicht"].wert - rows["rang"].wert)
    assert d(100 * closed, 0) == "66"
    sweep = V.ladder_sweep("wert")
    gains = {n: sweep[n]["werte"] - sweep[n]["rang"] for n in sweep}
    assert len(gains) == 8 and d(100 * min(gains.values()), 1) == "0,9" and min(gains, key=gains.get) == 200
    assert d(100 * max(gains.values()), 1) == "12,5" and max(gains, key=gains.get) == 2
    has("n = 10: 86,1 %", "n = 100: **98,1 %** gegen 96,4 %", "99,0 %", "schließen 66 %", "in den 8 gerechneten Angebotszahlen (2 bis 200) zwischen 0,9 Prozentpunkten (n = 200) und 12,5 (n = 2)")


def test_ladder_is_ordered_for_the_goals_where_all_stages_exist():
    """Hellsicht ≥ Werte bekannt ≥ Rangordnung ≥ Zufall, für Treffer und Wert in allen gerechneten n (Zählung über alle 8 Angebotszahlen)."""
    for goal in ("treffer", "wert"):
        for n, row in V.ladder_sweep(goal).items():
            assert row["hellsicht"] >= row["werte"] >= row["rang"] >= row["zufall"] - 1e-12, (goal, n)


# ------------------------------------------------------------------ H6
def test_h6_unknown_number_of_offers():
    for m, text, start in ((10, "35,1 %", 2), (100, "27,8 %", 14), (1000, "27,1 %", 136)):
        p, t = UN.optimal_rule(m)
        assert p1(p) == text and t == start, m
    assert p1(2 / math.e ** 2) == "27,1 %"
    known = RK.best_classic_curve(100)
    assert p1(known.mean()) == "38,8 %" and d(100 * (known.mean() - UN.optimal_rule(100)[0]), 1) == "11,0"
    assert p1(UN.win_given_n(14, 100)) == "27,0 %" and p1(known[-1]) == "37,1 %"
    has("m = 10: 35,1 %", "m = 100: **27,8 %**, Regel ab Angebot 14", "m = 1000: 27,1 % ab Angebot 136", "2/e² = 27,1 %", "**38,8 %**", "11,0 Prozentpunkte weniger", "27,0 %")


# ------------------------------------------------------------------ H7
def test_h7_envelopes():
    assert d(100 * EN.win_probability(1, 2, 10), 2) == "54,31" and d(100 * EN.win_probability(5, 15, 10), 2) == "69,17" and d(100 * EN.win_probability(100, 300, 10), 2) == "50,00"
    grid = V.envelope_grid()
    assert len(grid) == 72 and all(r["excess"] > 0 for r in grid)
    assert p1(max(r["p"] for r in grid)) == "84,2 %"
    assert sum(1 for r in grid if r["excess"] < 1e-3) == 21
    sim = EN.simulate(5, 15, 10, 100_000, seed=C.DEFAULT_SEED)
    p = EN.win_probability(5, 15, 10)
    assert abs(sim - p) < 4 * math.sqrt(p * (1 - p) / 100_000)
    has("(1; 2): 54,31 %", "(5; 15): 69,17 %", "(100; 300): 50,00 %", "**72 von 72**", "84,2 %", "in 21 Zellen weniger als 0,1 Prozentpunkte", "0,6944", "0,6917")


# ------------------------------------------------------------------ H8
def test_h8_cayley_rule_in_rank_terms():
    a = V.analyse(S.Settings())
    sim = V.simulate_cross(a)
    mean, se, runs = sim["werte_wert"]["rang"]
    assert runs == C.RUNS == 2000 and d(mean, 2) == "2,35" and d(se, 2) == "0,07"
    assert mean + 2 * se < a.rank_rule[1]
    kind, text = V.verdict_values_rank(mean, se, a.rank_rule[1])
    assert kind == "success" and "2,35" in text
    has("Rang **2,35** ± 0,07", "(3,60)")


# ------------------------------------------------------------------ H9
def test_h9_waiting_time_rules_for_rank_and_value():
    curve = RK.skip_curve(100)
    assert RK.best_skip(curve["rang"], False) == 9 and d(curve["rang"][9], 2) == "9,59"
    assert RK.best_skip(curve["wert"], True) == 9 and p1(curve["wert"][9]) == "90,5 %"
    w, _ = RK.expected_rank_dp(100)
    assert d(w, 2) == "3,60" and p1(RK.value_from_rank(w, 100)) == "96,4 %"
    has("**9** abgelehnte mit Rang 9,59", "für den Wert 9 mit 90,5 %", "3,60 bzw. 96,4 %")


# ------------------------------------------------------------------ H10
def test_h10_two_offers():
    rows = {r.key: r for r in V.ladder(2)}
    assert (p1(rows["rang"].treffer), d(rows["rang"].rang, 2), p1(rows["werte"].treffer), p1(rows["hellsicht"].treffer)) == ("50,0 %", "1,50", "75,0 %", "100,0 %")
    has("Rangordnung 50,0 % (erwarteter Rang 1,50), Werte bekannt 75,0 %, Hellsicht 100 %")


# ------------------------------------------------------------------ Kernfrage und Presets
def test_kernfrage_numbers():
    a = V.analyse(S.Settings())
    kind, text = V.verdict_ladder("treffer", 100, a.rows, a.best_rule[1])
    assert "37,1 %" in text and "58,3 %" in text and "21,2 Prozentpunkte" in text
    has("37,1 %, mit bekannten Werten mit 58,3 %: 21,2 Prozentpunkte mehr", "Rang 20,01", "26,3 %", "Rang 3,60", "11,0 Prozentpunkte weniger auf denselben Tagen", "Rang 2,35")


def test_preset_help_claims():
    """Die Texte der Presets nennen Wartezeiten und Werte: jede wird gegen die Rechnung gehalten."""
    assert RK.best_skip(RK.skip_curve(100)["treffer"], True) == 37 and S.skip_count(100, 37) == 37
    assert RK.best_skip(RK.skip_curve(10)["treffer"], True) == 3 and S.skip_count(10, 30) == 3
    assert S.skip_count(100, 9) == 9 and S.skip_count(200, 37) == 74
    assert RK.best_skip(RK.skip_curve(100)["wert"], True) == 9 and RK.best_skip(RK.skip_curve(100)["rang"], False) == 9
    two = {r.key: r for r in V.ladder(2)}
    assert p1(two["rang"].treffer) == "50,0 %" and p1(two["werte"].treffer) == "75,0 %"
    for name, text in (("Standardtag (100 Angebote)", "37"), ("Kleiner Tag (10 Angebote)", "3 abgelehnt"), ("Guter Preis statt bester", "9 von 100 abgelehnt"),
                       ("Nur zwei Angebote", "50 Prozent, mit bekannter Verteilung 75 Prozent"), ("Großer Tag (200 Angebote)", "74 abgelehnt")):
        assert text in C.PRESET_HELP[name], name


def test_the_waiting_time_help_in_the_sidebar_names_the_right_default():
    assert "37 Prozent ist bei 100 Angeboten die beste Wartezeit" in Path(__file__).resolve().parent.parent.joinpath("app.py").read_text(encoding="utf-8")
    assert RK.best_skip(RK.skip_curve(100)["treffer"], True) == 37


def test_threshold_euro_values_named_in_the_plan_and_the_verdict():
    """Treffer-Schwelle beim ersten von 100 Angeboten 1.197 €, beim vorletzten 1.000 €; Cayley beim ersten 1.192 € (Einzelplan, Verdikt in E4)."""
    b, v = VA.win_thresholds(100), VA.cayley_values(100)
    assert (V.euro(b[99]), V.euro(b[1]), V.euro(v[99])) == ("1.197 €", "1.000 €", "1.192 €")
    assert "nur einen Rekord ab 1.197 €" in V.verdict_thresholds(100, b, v)[1]
