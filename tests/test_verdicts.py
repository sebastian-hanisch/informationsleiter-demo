"""Verdikte als reine Funktionen (DEMO-PLAYBOOK §12): Schwellen und jede Zahl im Text an handgesetzten Zeilen."""

import numpy as np

import inl_evaluation as V
from inl_evaluation import LadderRow


def rows(treffer_r=0.4, treffer_w=0.6, rang_r=3.0, wert_r=0.9, wert_w=0.95, n=10):
    return (
        LadderRow("zufall", "Zufall", 1 / n, (n + 1) / 2, 0.5),
        LadderRow("unbekannt", "Anzahl unbekannt", 0.3, None, None),
        LadderRow("rang", "Rang", treffer_r, rang_r, wert_r),
        LadderRow("werte", "Werte", treffer_w, None, wert_w),
        LadderRow("hellsicht", "Hellsicht", 1.0, 1.0, 0.99),
    )


def test_formatting_helpers():
    assert V.de(3.14159, 2) == "3,14" and V.thousands(1234567) == "1.234.567" and V.pct(0.3714, 1) == "37,1 %" and V.pp(0.212) == "21,2 Prozentpunkte"
    assert V.euro(0.0) == "800 €" and V.euro(1.0) == "1.200 €" and V.euro(0.5) == "1.000 €"


def test_ladder_verdict_for_the_hit_goal():
    kind, text = V.verdict_ladder("treffer", 10, rows(), 5.0)
    assert kind == "success" and "40,0 %" in text and "60,0 %" in text and "20,0 Prozentpunkte" in text and "10,0 %" in text
    kind, _ = V.verdict_ladder("treffer", 10, rows(treffer_r=0.58, treffer_w=0.6), 5.0)        # nur 2 Prozentpunkte: Schwelle 5
    assert kind == "info"
    assert V.verdict_ladder("treffer", 10, rows(treffer_r=0.50, treffer_w=0.56), 5.0)[0] == "success"      # 6 Prozentpunkte über der Schwelle


def test_ladder_verdict_for_the_rank_goal():
    kind, text = V.verdict_ladder("rang", 10, rows(rang_r=3.0), 7.5)
    assert kind == "success" and "**3,00**" in text and "5,5" in text and "**7,50**" in text and "nicht gerechnet" in text
    kind, text = V.verdict_ladder("rang", 10, rows(rang_r=3.0), 3.02)               # Unterschied unter 0,05: dieselbe Regel
    assert kind == "info" and "denselben Wert" in text and "3,00" in text


def test_ladder_verdict_for_the_value_goal_with_the_closed_share():
    kind, text = V.verdict_ladder("wert", 10, rows(wert_r=0.90, wert_w=0.95), 5.0)
    assert kind == "success" and "90,0 %" in text and "95,0 %" in text and "99,0 %" in text and "56 %" in text       # (0,95 − 0,90)/(0,99 − 0,90) = 55,6 %
    assert "1.160 €" in text and "1.180 €" in text                                  # 800 + 400 · 0,90 und 800 + 400 · 0,95


def test_own_waiting_time_verdict():
    kind, text = V.verdict_own("treffer", 100, 37, 0.371, 37, 0.371)
    assert kind == "success" and "37 von 100" in text and "beste Wartezeit" in text
    kind, text = V.verdict_own("treffer", 100, 10, 0.2, 37, 0.37)
    assert kind == "info" and "20,0 %" in text and "37,0 %" in text and "17,0 Prozentpunkte weniger" in text
    kind, text = V.verdict_own("rang", 100, 50, 14.0, 9, 9.5)
    assert kind == "info" and "14,00" in text and "9,50" in text and "4,50 Rangplätze mehr" in text
    assert V.verdict_own("rang", 100, 9, 9.5, 9, 9.5)[0] == "success"
    kind, text = V.verdict_own("wert", 100, 9, 0.9, 9, 0.9)
    assert kind == "success" and "erwarteten Wert 90,0 %" in text


def test_goal_changes_rule_verdict():
    kind, text = V.verdict_goal_changes_rule(100, (0.371, 20.01, 0.80), (0.263, 3.60, 0.96), 37, 28)
    assert kind == "success" and "37,1 %" in text and "20,01" in text and "26,3 %" in text and "3,60" in text and "37 ablehnen" in text and "ab Angebot 28" in text
    kind, text = V.verdict_goal_changes_rule(2, (0.5, 1.5, 0.5), (0.5, 1.5, 0.5), 1, 2)
    assert kind == "info" and "gleich" in text and "50,0 %" in text and "1,50" in text


def test_unknown_number_verdict():
    kind, text = V.verdict_unknown(100, 0.278, 0.388, 14, 0.270, 0.371)
    assert kind == "info" and "27,8 %" in text and "38,8 %" in text and "11,0 Prozentpunkte" in text and "ab Angebot 14" in text and "27,0 %" in text and "37,1 %" in text


def test_envelope_verdict_counts_every_cell():
    grid = [{"p": 0.6, "excess": 0.1}, {"p": 0.5, "excess": 1e-30}, {"p": 0.8, "excess": 0.3}]
    kind, text = V.verdict_envelopes(grid)
    assert kind == "success" and "In 3 von 3 Zellen" in text and "80,0 %" in text and "In 1 Zellen" in text
    kind, text = V.verdict_envelopes(grid + [{"p": 0.5, "excess": 0.0}])
    assert kind == "warning" and "In 3 von 4 Zellen" in text


def test_sweep_verdict_for_every_goal():
    sweep = {2: {"zufall": 0.5, "rang": 0.5, "werte": 0.75}, 10: {"zufall": 0.1, "rang": 0.4, "werte": 0.61}, 100: {"zufall": 0.01, "rang": 0.37, "werte": 0.57}}
    kind, text = V.verdict_sweep("treffer", sweep)
    assert "3 gerechneten Angebotszahlen" in text and "20,0 Prozentpunkten (n = 100)" in text and "25,0 Prozentpunkten (n = 2)" in text and "höchsten Angebot" in text
    assert "erwarteten Wert" in V.verdict_sweep("wert", sweep)[1]
    rank_sweep = {2: {"rang": 1.5, "zufall": 1.5}, 100: {"rang": 3.6, "zufall": 50.5}}
    kind, text = V.verdict_sweep("rang", rank_sweep)
    assert "1,50 (n = 2)" in text and "3,60 (n = 100)" in text and "50,5 bei n = 100" in text


def test_values_rank_verdict_has_both_branches():
    kind, text = V.verdict_values_rank(2.35, 0.07, 3.60)
    assert kind == "success" and "2,35" in text and "0,07" in text and "3,60" in text and "Robbins" in text
    kind, text = V.verdict_values_rank(3.50, 0.20, 3.60)                       # der Vorsprung liegt innerhalb von zwei Standardfehlern
    assert kind == "info" and "nicht zu sehen" in text


def test_threshold_verdict_in_euro_for_small_days():
    b = np.array([0.0, 0.5, 0.69])
    v = np.array([0.0, 0.5, 0.625])
    kind, text = V.verdict_thresholds(3, b, v)
    assert "nur einen Rekord ab 1.076 €" in text and "ein Angebot ab 1.050 €" in text and "1.000 € und 1.000 €" in text and "Schwelle 800 €" in text
    kind, text = V.verdict_thresholds(2, np.array([0.0, 0.5]), np.array([0.0, 0.5]))
    assert "Beim vorletzten" not in text and text.count("Beim letzten") == 1 and "1.000 €" in text
    assert V.verdict_thresholds(1, np.array([0.0]), np.array([0.0]))[1].startswith("Mit einem Angebot")


def test_ladder_check_rows_distance_in_standard_errors():
    a = V.analyse(V.Settings(n=10, goal="treffer"))
    sim = V.simulate_cross(a)
    out = V.ladder_check_rows(a, sim, (0.3, 0.01))
    assert [r["stage"] for r in out] == ["Nur Rangordnung", "Werte bekannt", "Anzahl unbekannt (nur Rangordnung)"]
    last = out[-1]
    assert abs(last["z"] - (0.3 - last["exact"]) / 0.01) < 1e-12
    out_rank = V.ladder_check_rows(V.analyse(V.Settings(n=10, goal="rang")), V.simulate_cross(V.analyse(V.Settings(n=10, goal="rang"))), (0.0, 0.0))
    assert [r["stage"] for r in out_rank] == ["Nur Rangordnung"]
