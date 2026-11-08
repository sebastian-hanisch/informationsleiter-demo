"""Stufe „nur Rangordnung“: nach jedem Angebot steht nur fest, auf welchem Platz es unter den bisherigen liegt (1 = am höchsten).

Eine Regel ist hier eine Schwellenliste `s` (Länge n + 1, Index 0 ungenutzt): zum Zeitpunkt k wird das Angebot genommen, wenn sein Relativrang
höchstens `s[k]` ist; `s[n] = n` erzwingt das letzte Angebot. Alle Kennzahlen sind exakt (Brüche von Wahrscheinlichkeiten, keine Stichprobe).

Einheiten (DEMO-PLAYBOOK §1b): `skip_policy` (Regel „erst ablehnen, dann den ersten Rekord“), `expected_rank_dp` (Rückwärts-DP für den erwarteten Rang),
`evaluate_policy` (exakte Bewertung irgendeiner Schwellenliste), `skip_curve` (Bewertung aller Wartezeiten)."""

import numpy as np


def skip_policy(n: int, skip: int) -> np.ndarray:
    """Die ersten `skip` Angebote ablehnen, dann das erste nehmen, das alle bisherigen übertrifft (sonst das letzte). skip = 0 nimmt das erste."""
    if not 0 <= skip <= n - 1:
        raise ValueError(f"skip = {skip} liegt außerhalb von 0 bis {n - 1}")
    s = np.zeros(n + 1, dtype=np.int64)
    s[skip + 1:] = 1
    s[n] = n
    return s


def evaluate_policy(n: int, s: np.ndarray) -> tuple[float, float]:
    """Exakt: (Wahrscheinlichkeit, das beste Angebot zu nehmen, erwarteter absoluter Rang des genommenen Angebots) der Schwellenliste `s`.

    Relativränge der Zeitpunkte sind unabhängig und gleichverteilt auf 1..k. Genommen wird bei Relativrang j <= s[k]; der absolute Rang eines
    Angebots mit Relativrang j zur Zeit k ist im Mittel j (n + 1) / (k + 1), und der Relativbeste ist mit Wahrscheinlichkeit k / n der absolut beste."""
    k = np.arange(1, n + 1, dtype=float)
    sk = np.minimum(np.asarray(s[1:], dtype=float), k)       # ein Relativrang ist höchstens k
    take = sk / k                                            # Wahrscheinlichkeit, zur Zeit k zu nehmen
    alive = np.concatenate([[1.0], np.cumprod(1.0 - take)[:-1]])   # noch nichts genommen vor Zeitpunkt k
    p_best = float(np.sum(alive * (sk >= 1)) / n)
    e_rank = float(np.sum(alive * (sk * (sk + 1) / 2) / k * (n + 1) / (k + 1)))
    return p_best, e_rank


def expected_rank_dp(n: int) -> tuple[float, np.ndarray]:
    """Rückwärts-DP (Chow, Moriguti, Robbins, Samuels): minimaler erwarteter absoluter Rang und die Schwellenliste der optimalen Regel.

    W[k] = erwarteter Endrang, wenn die ersten k Angebote abgelehnt sind und optimal weitergemacht wird; W[n] = unendlich (kein Angebot mehr)."""
    W = np.full(n + 1, np.inf)
    s = np.zeros(n + 1, dtype=np.int64)
    s[n] = n
    for k in range(n - 1, -1, -1):
        t = k + 1                                            # nächstes Angebot
        stop = np.arange(1, t + 1) * (n + 1) / (t + 1)       # erwarteter Endrang bei Relativrang j
        W[k] = np.minimum(stop, W[t]).mean()
        s[t] = int(np.sum(stop <= W[t]))
    s[n] = n
    return float(W[0]), s


def classic_win(n: int, skip: int) -> float:
    """Geschlossene Formel für „skip ablehnen, dann den ersten Rekord“: P(bestes) = (skip / n) · Σ_{k=skip+1..n} 1 / (k − 1); skip = 0: 1 / n."""
    if skip == 0:
        return 1.0 / n
    k = np.arange(skip + 1, n + 1, dtype=float)
    return float(skip / n * np.sum(1.0 / (k - 1.0)))


def skip_curve(n: int) -> dict[str, np.ndarray]:
    """Alle Wartezeiten 0..n-1: exakte Trefferwahrscheinlichkeit, erwarteter Rang und erwarteter Wert (1 − Rang / (n + 1)) der Regel `skip_policy`."""
    skips = np.arange(n)
    p = np.empty(n)
    r = np.empty(n)
    for skip in skips:
        p[skip], r[skip] = evaluate_policy(n, skip_policy(n, int(skip)))
    return {"skip": skips, "treffer": p, "rang": r, "wert": value_from_rank(r, n)}


def best_skip(curve_values: np.ndarray, maximize: bool) -> int:
    """Wartezeit mit dem besten Wert der Kurve; bei Gleichstand (Abstand höchstens 1e-12) die längere Wartezeit, weil die Regel dann wenigstens Information sammelt (n = 2: 1 statt 0)."""
    best = curve_values.max() if maximize else curve_values.min()
    return int(np.flatnonzero(np.abs(curve_values - best) <= 1e-12)[-1])


def value_from_rank(e_rank, n: int):
    """Mit unabhängigen gleichverteilten Werten ist der Erwartungswert des Angebots mit absolutem Rang r gleich (n + 1 − r) / (n + 1) (Ordnungsstatistik);
    der erwartete Wert eines Rangordnungs-Verfahrens ist daher 1 − E[Rang] / (n + 1). Ziel „Wert“ und Ziel „Rang“ haben bei bekannter Rangordnung dieselbe optimale Regel."""
    return 1.0 - e_rank / (n + 1.0)


def best_classic_curve(m: int) -> np.ndarray:
    """Beste Trefferwahrscheinlichkeit der Regel „erst ablehnen, dann Rekord“ für n = 1..m (Index n − 1), in O(m²) mit Harmonischen Zahlen:
    P(n, s) = (s / n) · (H_{n−1} − H_{s−1}) für s >= 1, P(n, 0) = 1 / n."""
    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, m + 1))])
    out = np.empty(m)
    for n in range(1, m + 1):
        s = np.arange(1, n)
        best = 1.0 / n
        if n > 1:
            best = max(best, float(np.max(s / n * (H[n - 1] - H[s - 1]))))
        out[n - 1] = best
    return out
