"""Stufe „Werte bekannt“: die Werte der Angebote sind unabhängig und gleichverteilt auf [0, 1] (bekannt), die Zahl n der Angebote ebenfalls.

Zwei Ziele: Treffer (das größte Angebot) und Wert (der erwartete Wert des genommenen Angebots, Cayley). Alle Zahlen sind exakt bis auf den Rasterfehler
der Raster-DP für den Treffer (Fehler bei n = 200: rund 4 · 10⁻⁶, Vergleich Raster 16 000 gegen 64 000).

Einheiten: `cayley_values` (Rekursion), `win_thresholds` (Schwellen für Rekorde), `full_info_win` (Raster-DP)."""

import numpy as np


def cayley_values(n: int) -> np.ndarray:
    """v[j] = erwarteter Wert des genommenen Angebots mit j noch kommenden Angeboten (optimal); v[0] = 0, v[j] = (1 + v[j-1]²) / 2.

    Ein Angebot x wird bei j noch kommenden genommen, wenn x >= v[j] (der Erwartungswert des Weitermachens)."""
    v = np.zeros(n + 1)
    for j in range(1, n + 1):
        v[j] = (1.0 + v[j - 1] ** 2) / 2.0
    return v


def win_thresholds(n: int) -> np.ndarray:
    """b[k] für k = 0..n-1 noch kommende Angebote: Ein Rekord x wird genommen, wenn x >= b[k]. b[0] = 0 (das letzte Angebot nimmt man, wenn es ein Rekord ist).

    b[k] ist die Wurzel von Σ_{i=1..k} (x^(-i) − 1) / i = 1, hier durch gleichzeitige Bisektion für alle k bestimmt. Herleitung: Bei Rekord x und k noch kommenden gewinnt Nehmen
    mit x^k; Weitermachen und danach jeden Rekord nehmen gewinnt mit Σ_{i=1..k} (x^(k−i) − x^k) / i. Gleichheit beider ist die Gleichung nach Division durch x^k."""
    k = np.arange(n, dtype=float)
    lo = np.full(n, 1e-9)
    hi = np.ones(n)
    idx = np.arange(1, n + 1, dtype=float)                   # i = 1..n
    mask = idx[None, :] <= k[:, None]                        # i <= k
    for _ in range(80):
        mid = (lo + hi) / 2
        with np.errstate(over="ignore"):
            terms = (mid[:, None] ** (-idx[None, :]) - 1.0) / idx[None, :]
        lhs = np.where(mask, terms, 0.0).sum(axis=1)
        larger = lhs > 1.0                                   # links zu groß: x ist zu klein
        lo = np.where(larger, mid, lo)
        hi = np.where(larger, hi, mid)
    b = (lo + hi) / 2
    b[0] = 0.0
    return b


def full_info_win(n: int, grid: int) -> float:
    """Wahrscheinlichkeit, bei bekannter Gleichverteilung und optimalem Stoppen das größte der n Angebote zu nehmen (Raster-DP).

    Zustand: k noch kommende Angebote und bisheriges Maximum m (ein Angebot unter m kann nie das beste werden). f_k(m) = m · f_{k-1}(m) + ∫_m^1 max(x^(k−1), f_{k−1}(x)) dx:
    das nächste Angebot liegt mit Wahrscheinlichkeit m unter dem bisherigen Maximum (weiter mit f_{k−1}(m)), sonst ist es ein Rekord x (nehmen gewinnt mit x^(k−1), weitermachen mit f_{k−1}(x))."""
    x = np.linspace(0.0, 1.0, grid + 1)
    f = np.zeros(grid + 1)
    for k in range(1, n + 1):
        gain = np.maximum(x ** (k - 1), f)
        seg = (gain[:-1] + gain[1:]) / 2 / grid
        suffix = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        f = x * f + suffix
    return float(f[0])
