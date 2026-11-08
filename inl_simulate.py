"""Regeln auf konkreten Folgen: jede Regel wählt aus jeder Zeile einer (runs, n)-Matrix von Werten ein Angebot (Index 0..n-1, −1 = keines gewählt).

Dieselben Folgen gehen durch alle Regeln, damit der Vergleich nicht am Zufall der Folgen hängt. Alle Funktionen sind vektorisiert über die Zeilen; die
Schleifen laufen nur über den Zeitpunkt (jeder Schritt hängt vom Vorgänger ab)."""

import numpy as np


def records(vals: np.ndarray) -> np.ndarray:
    """True, wo ein Angebot alle früheren Angebote der Zeile übertrifft (das erste immer)."""
    runmax = np.maximum.accumulate(vals, axis=1)
    prev = np.concatenate([np.full((vals.shape[0], 1), -np.inf), runmax[:, :-1]], axis=1)
    return vals > prev


def first_true(mask: np.ndarray, fallback: int) -> np.ndarray:
    """Index des ersten True je Zeile; Zeilen ohne True erhalten `fallback`."""
    has = mask.any(axis=1)
    return np.where(has, mask.argmax(axis=1), fallback)


def pick_skip(vals: np.ndarray, skip: int) -> np.ndarray:
    """Die ersten `skip` ablehnen, dann den ersten Rekord nehmen, sonst das letzte Angebot."""
    n = vals.shape[1]
    mask = records(vals)
    mask[:, :skip] = False
    return first_true(mask, n - 1)


def relative_ranks(vals: np.ndarray) -> np.ndarray:
    """(runs, n): Relativrang des Angebots t unter den Angeboten 1..t (1 = höchstes)."""
    runs, n = vals.shape
    rel = np.empty((runs, n), dtype=np.int64)
    for t in range(n):
        rel[:, t] = 1 + (vals[:, :t] > vals[:, t:t + 1]).sum(axis=1)
    return rel


def pick_rank_policy(vals: np.ndarray, s: np.ndarray, rel: np.ndarray | None = None) -> np.ndarray:
    """Das erste Angebot nehmen, dessen Relativrang höchstens s[t] ist (t = 1..n); s[n] = n erzwingt eine Wahl."""
    n = vals.shape[1]
    rel = relative_ranks(vals) if rel is None else rel
    return first_true(rel <= s[1:][None, :], n - 1)


def pick_value_win(vals: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ziel Treffer bei bekannten Werten: der erste Rekord x mit x >= b[k], k = Zahl der danach noch kommenden Angebote."""
    n = vals.shape[1]
    thr = b[n - 1 - np.arange(n)]
    return first_true(records(vals) & (vals >= thr[None, :]), n - 1)


def pick_cayley(vals: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Ziel Wert: das erste Angebot x mit x >= v[k] (k danach noch kommend, v[0] = 0 nimmt das letzte)."""
    n = vals.shape[1]
    thr = v[n - 1 - np.arange(n)]
    return first_true(vals >= thr[None, :], n - 1)


def pick_unknown(vals: np.ndarray, stop_from: int) -> np.ndarray:
    """Anzahl unbekannt: ab Zeitpunkt `stop_from` den ersten Rekord nehmen; ohne Rekord ab dann wird nichts gewählt (−1)."""
    mask = records(vals)
    mask[:, :stop_from - 1] = False
    return first_true(mask, -1)


def pick_hindsight(vals: np.ndarray) -> np.ndarray:
    """Obere Schranke: im Nachhinein das größte Angebot (keine Regel, die man im Betrieb umsetzen kann)."""
    return vals.argmax(axis=1)


def pick_random(n: int, u: np.ndarray) -> np.ndarray:
    """Zufall ohne Wissen: ein gleichverteilter Index aus gleichverteilten Zahlen u in [0, 1)."""
    return np.minimum((u * n).astype(np.int64), n - 1)


def score(vals: np.ndarray, chosen: np.ndarray) -> dict[str, np.ndarray]:
    """Je Zeile: Treffer (1/0), absoluter Rang und Wert des gewählten Angebots; ohne Wahl (−1) Treffer 0 und Rang und Wert NaN."""
    runs = vals.shape[0]
    ok = chosen >= 0
    picked = np.where(ok, vals[np.arange(runs), np.maximum(chosen, 0)], np.nan)
    rank = np.where(ok, 1 + (vals > np.nan_to_num(picked, nan=np.inf)[:, None]).sum(axis=1), np.nan)
    return {"treffer": (rank == 1).astype(float), "rang": rank, "wert": picked}


def summarize(scores: dict[str, np.ndarray]) -> dict[str, tuple[float, float, int]]:
    """Je Kennzahl (Mittel, Standardfehler des Mittels, Zahl der Läufe mit Wahl). Treffer zählt „keine Wahl“ als 0; Rang und Wert nur über Läufe mit Wahl."""
    out = {}
    for key, arr in scores.items():
        good = arr[~np.isnan(arr)]
        se = float(good.std(ddof=1) / np.sqrt(len(good))) if len(good) > 1 else 0.0
        out[key] = (float(good.mean()) if len(good) else float("nan"), se, int(len(good)))
    return out


def unknown_n_days(vals: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Tage verschiedener Länge N_i in einer Matrix fester Breite: Angebote hinter N_i werden zu −∞ (kommen nie an). Für die Simulation der unbekannten Anzahl."""
    mask = np.arange(vals.shape[1])[None, :] < lengths[:, None]
    return np.where(mask, vals, -np.inf)


def simulate_unknown_days(vals: np.ndarray, lengths: np.ndarray, stop_from: int) -> float:
    """Anteil Treffer der Regel „ab stop_from den ersten Rekord“ über Tage mit zufälliger Länge (Zeilen von `vals` sind die Tage, `lengths` ihre Längen)."""
    days = unknown_n_days(vals, lengths)
    chosen = pick_unknown(days, stop_from)
    runs = vals.shape[0]
    ok = chosen >= 0
    return float(np.mean(ok & (days[np.arange(runs), np.maximum(chosen, 0)] == days.max(axis=1))))

