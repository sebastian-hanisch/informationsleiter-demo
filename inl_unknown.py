"""Stufe „Anzahl unbekannt“: die Zahl N der Angebote ist gleichverteilt auf 1..m (bekannt ist nur m), sonst nur Rangordnung, Ziel: das beste Angebot.

Kommt der Tag zu Ende, ohne dass genommen wurde, gibt es kein Angebot: kein Treffer. Die optimale Regel ist eine Schwelle: ab dem Zeitpunkt `stop_from`
wird der erste Rekord genommen (hier durch Aufzählung aller Stopp-Mengen für m <= 7 und durch eine Prüfung der Form für m <= 200 belegt, nicht bewiesen).

Einheiten: `harmonic` (Hilfsgröße), `optimal_rule` (Rückwärts-DP), `win_given_n` (exakte Trefferwahrscheinlichkeit der Schwellenregel bei festem N)."""

import numpy as np


def harmonic(m: int) -> np.ndarray:
    """H[i] = 1 + 1/2 + … + 1/i, H[0] = 0."""
    return np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, m + 1))])


def stop_set(m: int) -> tuple[float, list[int]]:
    """Rückwärts-DP: (optimale Trefferwahrscheinlichkeit über N ~ U{1..m}, Zeiten, zu denen ein Rekord genommen wird).

    Mit Rekord zur Zeit k (bedingt N >= k) gewinnt Nehmen mit E[k / N | N >= k]; C[k] ist der Wert des Weitermachens nach Zeit k, bedingt N >= k."""
    H = harmonic(m)

    def stop_win(k: int) -> float:
        return k * (H[m] - H[k - 1]) / (m - k + 1)

    C = np.zeros(m + 2)
    stops = []
    for k in range(m - 1, 0, -1):
        sw = stop_win(k + 1)
        p_next = (m - k) / (m - k + 1)                       # P(N >= k + 1 | N >= k)
        C[k] = p_next * (max(sw, C[k + 1]) / (k + 1) + k / (k + 1) * C[k + 1])
        if sw >= C[k + 1]:
            stops.append(k + 1)
    if stop_win(1) >= C[1]:
        stops.append(1)
    return float(max(stop_win(1), C[1])), sorted(stops)


def optimal_rule(m: int) -> tuple[float, int]:
    """(Trefferwahrscheinlichkeit der optimalen Regel über N ~ U{1..m}, Zeitpunkt `stop_from`, ab dem der erste Rekord genommen wird).

    Die Stopp-Menge ist ein Endstück {stop_from, …, m} (in den Tests für alle m bis 200 geprüft). Bei Gleichstand der Werte (nur sehr kleine m) gilt der kleinste Zeitpunkt."""
    p, stops = stop_set(m)
    return p, stops[0]


def win_given_n(stop_from: int, n: int) -> float:
    """Exakte Trefferwahrscheinlichkeit der Regel „ab stop_from den ersten Rekord nehmen“ bei genau N = n Angeboten (0 für n < stop_from).

    Der Rekord zur Zeit t ist mit Wahrscheinlichkeit t / n der beste; noch nicht genommen vor t ist nach stop_from mit Wahrscheinlichkeit (stop_from − 1) / (t − 1)
    (stop_from = 1: das erste Angebot wird immer genommen)."""
    if n < stop_from:
        return 0.0
    if stop_from == 1:
        return 1.0 / n
    t = np.arange(stop_from, n + 1, dtype=float)
    alive = (stop_from - 1.0) / (t - 1.0)
    return float(np.sum(alive) / n)


def win_curve(stop_from: int, m: int) -> np.ndarray:
    """Die Regel bei festem N = 1..m (Index N - 1)."""
    return np.array([win_given_n(stop_from, n) for n in range(1, m + 1)])
