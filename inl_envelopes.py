"""Unterste Sprosse: zwei Umschläge (Cover). Eine andere Aufgabenform als der Rest der Leiter (n = 2, zwei feste, unbekannte Beträge, keine Verteilung der Beträge).

Man öffnet einen von zwei Umschlägen (zufällig gewählt) mit den Beträgen a < b, sieht den Betrag x und darf bei diesem bleiben oder zum anderen wechseln. Ohne Schwelle
gewinnt man mit 1/2. Mit einer Zufallsschwelle T (aus einer Verteilung mit Dichte > 0 auf allen positiven Zahlen), Wechsel genau dann, wenn x < T, gewinnt man mit
1/2 + 1/2 · P(a < T < b) für JEDES Paar a < b."""

import math

import numpy as np

from inl_rng import uniform_block


def win_probability(a: float, b: float, scale: float) -> float:
    """Exakt für eine exponentialverteilte Schwelle mit Mittel `scale`: 1/2 + 1/2 · (e^(−a/scale) − e^(−b/scale))."""
    return 0.5 + excess(a, b, scale)


def excess(a: float, b: float, scale: float) -> float:
    """Gewinnwahrscheinlichkeit minus 1/2 = 1/2 · P(a < T < b) = 1/2 · e^(−a/scale) · (1 − e^(−(b−a)/scale)), numerisch stabil (kein Verlust bei kleinen Werten)."""
    if not 0 < a <= b:
        raise ValueError("erwartet 0 < a <= b")
    return 0.5 * math.exp(-a / scale) * -math.expm1(-(b - a) / scale)


def simulate(a: float, b: float, scale: float, runs: int, seed: int) -> float:
    """Anteil gewonnener Läufe: Umschlag zufällig, Schwelle T = −scale · ln(1 − u) (Inversion), wechseln wenn der gesehene Betrag < T ist."""
    u = uniform_block(seed, 2 * runs)
    opened_small = u[:runs] < 0.5
    seen = np.where(opened_small, a, b)
    threshold = -scale * np.log(1.0 - u[runs:])
    swap = seen < threshold
    won = np.where(opened_small, swap, ~swap)               # kleiner geöffnet: wechseln gewinnt; größer geöffnet: bleiben gewinnt
    return float(won.mean())
