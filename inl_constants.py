"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenbudget (DEMO-PLAYBOOK §1c), alles am Slider-Maximum N_MAX = 200:
* Rangordnung, Ziel Rang: Rückwärts-DP O(n²) in numpy.
* Werte bekannt, Ziel Treffer: Raster-DP, n Schritte über GRID + 1 Punkte, O(n · GRID) = 3,2 Mio. Operationen.
* Schwellen b_k: gleichzeitige Bisektion für alle k, O(60 · n²).
* Simulation: RUNS Folgen, die Regel nach Rangordnung braucht je Zeitpunkt einen Vergleich mit allen früheren, O(RUNS · n²) = 80 Mio. Vergleiche.
* Unbekannte Anzahl: O(n) mit Harmonischen Zahlen."""

N_MIN, N_MAX, DEFAULT_N = 2, 200, 100
WAIT_MIN, WAIT_MAX, DEFAULT_WAIT = 0, 100, 37       # eigene Wartezeit: Anteil der abgelehnten Angebote in Prozent
SEED_MAX, DEFAULT_SEED = 999999, 7
GOALS = ("treffer", "rang", "wert")
GOAL_LABELS = {
    "treffer": "Treffer: das höchste Angebot erwischen",
    "rang": "Rang: kleiner erwarteter Rang",
    "wert": "Wert: hohe erwartete Rate",
}
DEFAULT_GOAL = "treffer"

RUNS = 2000                    # Folgen der Simulation (die gezeigte Folge ist die erste davon)
GRID = 16000                   # Rasterpunkte der Raster-DP bei bekannten Werten (Fehler bei n = 200 gegen Raster 64 000: rund 4 · 10⁻⁶)
SWEEP_NS = (2, 3, 5, 10, 20, 50, 100, 200)       # Angebotszahlen der Vergleiche über n
SWEEP_MS = (10, 20, 50, 100, 200)                # Obergrenzen der Messreihe bei unbekannter Anzahl

# Kulisse: Spotfracht-Angebote. Die Rate ist nur der Maßstab der gleichverteilten Werte (0 = RATE_LOW, 1 = RATE_HIGH).
RATE_LOW, RATE_HIGH = 800.0, 1200.0

# Zwei Umschläge (Einstieg): kleiner Betrag, Verhältnis, Skala der Zufallsschwelle (exponentialverteilt)
ENV_AMOUNTS = (1, 5, 10, 50, 100, 300)
ENV_RATIOS = (1.2, 2, 3, 10)
ENV_SCALES = (5, 10, 50)
ENV_RUNS = 100000

COLOR_CHOSEN = "#2F6B65"
COLOR_OTHER = "rgba(150,150,150,0.55)"
COLOR_LINE = "#4c78a8"
COLORS = {
    "zufall": "#9b9b9b",
    "unbekannt": "#b279a2",
    "rang": "#4c78a8",
    "werte": "#e45756",
    "hellsicht": "#2F6B65",
    "eigene": "#f58518",
}

PRESETS = {
    "Standardtag (100 Angebote)": {"n": DEFAULT_N, "goal": DEFAULT_GOAL, "wait": DEFAULT_WAIT, "seed": DEFAULT_SEED},
    "Kleiner Tag (10 Angebote)": {"n": 10, "goal": DEFAULT_GOAL, "wait": 30, "seed": DEFAULT_SEED},
    "Guter Preis statt bester": {"n": DEFAULT_N, "goal": "wert", "wait": 9, "seed": DEFAULT_SEED},
    "Kleiner erwarteter Rang": {"n": DEFAULT_N, "goal": "rang", "wait": 9, "seed": DEFAULT_SEED},
    "Nur zwei Angebote": {"n": 2, "goal": DEFAULT_GOAL, "wait": 50, "seed": DEFAULT_SEED},
    "Großer Tag (200 Angebote)": {"n": N_MAX, "goal": DEFAULT_GOAL, "wait": DEFAULT_WAIT, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardtag (100 Angebote)": "100 Spotfracht-Angebote nacheinander, Ziel das höchste Angebot. Die ersten 37 ablehnen, dann das erste nehmen, das alle bisherigen übertrifft.",
    "Kleiner Tag (10 Angebote)": "Zehn Angebote, Ziel das höchste. Wartezeit 30 Prozent heißt 3 abgelehnt.",
    "Guter Preis statt bester": "Ziel ist eine hohe erwartete Rate statt der höchsten (Cayley). Wartezeit 9 Prozent heißt 9 von 100 abgelehnt, bei n = 100 die beste Wartezeit der Regelform „ablehnen, dann Rekord“ für dieses Ziel.",
    "Kleiner erwarteter Rang": "Ziel ist ein kleiner erwarteter Rang des gewählten Angebots (Chow, Robbins, Samuels). Wartezeit 9 Prozent ist bei n = 100 die beste Wartezeit der Regelform „ablehnen, dann Rekord“ für dieses Ziel; die Rang-optimale Regel selbst ist keine reine Wartezeit-Regel.",
    "Nur zwei Angebote": "Der kleinste Fall: ohne Werte 50 Prozent, mit bekannter Verteilung 75 Prozent. Zugleich der Einstieg zu den zwei Umschlägen.",
    "Großer Tag (200 Angebote)": "Das Maximum des Reglers: 200 Angebote, Wartezeit 37 Prozent heißt 74 abgelehnt.",
}
