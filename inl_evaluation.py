"""Kennzahlen, Verdikt-Logik und Messreihen. Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)` (DEMO-PLAYBOOK §12)."""

from dataclasses import dataclass

import numpy as np

import inl_constants as C
import inl_envelopes as EN
import inl_rank as RK
import inl_scenario as S
import inl_simulate as SIM
import inl_unknown as UN
import inl_values as VA
from inl_scenario import Settings

STAGES = ("zufall", "unbekannt", "rang", "werte", "hellsicht")
STAGE_NAMES = {
    "zufall": "Zufall (ohne Wissen)",
    "unbekannt": "Anzahl unbekannt (nur Rangordnung)",
    "rang": "Nur Rangordnung",
    "werte": "Werte bekannt",
    "hellsicht": "Hellsicht (obere Schranke)",
}
GOAL_UNITS = {"treffer": "Trefferwahrscheinlichkeit", "rang": "erwarteter Rang", "wert": "erwarteter Wert"}


# --------------------------------------------------------------------------------------------------- Formatierung (deutsches Dezimalkomma)
def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma und festen Nachkommastellen."""
    return f"{x:.{nd}f}".replace(".", ",")


def thousands(x: float) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{int(round(x)):,}".replace(",", ".")


def pct(x: float, nd: int = 1) -> str:
    """Anteil in [0, 1] als Prozentangabe mit Komma."""
    return de(100.0 * x, nd) + " %"


def pp(x: float, nd: int = 1) -> str:
    """Differenz zweier Anteile in Prozentpunkten."""
    return de(100.0 * x, nd) + " Prozentpunkte"


def euro(value: float, nd: int = 0) -> str:
    """Wert in [0, 1] als Frachtrate in Euro."""
    return thousands(S.to_rate(value)) + " €" if nd == 0 else de(float(S.to_rate(value)), nd) + " €"


# --------------------------------------------------------------------------------------------------- die Leiter (exakt)
@dataclass(frozen=True)
class LadderRow:
    key: str
    name: str
    treffer: float | None      # Wahrscheinlichkeit, das höchste Angebot zu nehmen (optimale Regel für dieses Ziel)
    rang: float | None         # erwarteter absoluter Rang des genommenen Angebots (1 = bestes), optimale Regel für dieses Ziel
    wert: float | None         # erwarteter Wert des genommenen Angebots in [0, 1], optimale Regel für dieses Ziel

    def get(self, goal: str):
        return getattr(self, goal)


def ladder(n: int) -> tuple[LadderRow, ...]:
    """Die Wissensstufen für n Angebote, je Ziel die optimale Regel der Stufe; „None“ = für diese Kombination nicht gerechnet (siehe README).

    Anzahl unbekannt: Obergrenze m = n, die Zahl der Angebote ist gleichverteilt auf 1..n, nur das Ziel Treffer ist gerechnet.
    Werte bekannt mit Ziel Rang ist das Robbins-Problem und hier nicht gerechnet."""
    curve = RK.skip_curve(n)
    s_best = RK.best_skip(curve["treffer"], True)
    w_rank, _ = RK.expected_rank_dp(n)
    rows = [
        LadderRow("zufall", STAGE_NAMES["zufall"], 1.0 / n, (n + 1) / 2.0, 0.5),
        LadderRow("unbekannt", STAGE_NAMES["unbekannt"], UN.optimal_rule(n)[0], None, None),
        LadderRow("rang", STAGE_NAMES["rang"], float(curve["treffer"][s_best]), w_rank, float(RK.value_from_rank(w_rank, n))),
        LadderRow("werte", STAGE_NAMES["werte"], VA.full_info_win(n, C.GRID), None, float(VA.cayley_values(n)[n])),
        LadderRow("hellsicht", STAGE_NAMES["hellsicht"], 1.0, 1.0, n / (n + 1.0)),
    ]
    return tuple(rows)


def ladder_sweep(goal: str, ns=C.SWEEP_NS) -> dict[int, dict[str, float | None]]:
    """Die Leiter für mehrere n: {n: {Stufe: Wert für das Ziel}}."""
    return {n: {r.key: r.get(goal) for r in ladder(n)} for n in ns}


# --------------------------------------------------------------------------------------------------- Analyse für die Hauptansicht
@dataclass(frozen=True)
class Analysis:
    settings: Settings
    n: int
    rows: tuple[LadderRow, ...]
    curve: dict                      # skip_curve(n) (Arrays)
    skip_own: int
    skip_opt: dict                   # beste Wartezeit der Regel „ablehnen, dann Rekord“ je Ziel
    own: tuple[float, float, float]  # exakte Kennzahlen der eigenen Wartezeit (Treffer, Rang, Wert)
    s_rank: np.ndarray               # Schwellenliste der Rang-optimalen Regel
    best_rule: tuple[float, float, float]   # Kennzahlen der Regel fürs höchste Angebot (Treffer, Rang, Wert), exakt
    rank_rule: tuple[float, float, float]   # Kennzahlen der Rang-optimalen Regel, exakt
    t_unknown: int                   # ab diesem Zeitpunkt nimmt die Regel bei unbekannter Anzahl den ersten Rekord
    b: np.ndarray                    # Schwellen für Rekorde (Ziel Treffer, Werte bekannt)
    v: np.ndarray                    # Schwellen nach Cayley (Ziel Wert)
    values: np.ndarray               # die gezeigte Folge (Zeile 0 der Simulation)
    picks: dict                      # Stufe -> gewählter Index (0-basiert) oder None, für das gewählte Ziel


def _metrics_of_policy(n: int, s: np.ndarray) -> tuple[float, float, float]:
    p, r = RK.evaluate_policy(n, s)
    return p, r, float(RK.value_from_rank(r, n))


def analyse(settings: Settings) -> Analysis:
    n = settings.n
    rows = ladder(n)
    curve = RK.skip_curve(n)
    skip_opt = {"treffer": RK.best_skip(curve["treffer"], True), "rang": RK.best_skip(curve["rang"], False), "wert": RK.best_skip(curve["wert"], True)}
    skip_own = S.skip_count(n, settings.wait)
    w_rank, s_rank = RK.expected_rank_dp(n)
    p_u, t_u = UN.optimal_rule(n)
    b, v = VA.win_thresholds(n), VA.cayley_values(n)
    vals = S.make_values(settings.seed, n, 1)
    picks = {
        "rang": int(SIM.pick_skip(vals, skip_opt["treffer"])[0]) if settings.goal == "treffer" else int(SIM.pick_rank_policy(vals, s_rank)[0]),
        "werte": {"treffer": lambda: int(SIM.pick_value_win(vals, b)[0]), "wert": lambda: int(SIM.pick_cayley(vals, v)[0]), "rang": lambda: None}[settings.goal](),
        "unbekannt": int(SIM.pick_unknown(vals, t_u)[0]) if settings.goal == "treffer" else None,
        "eigene": int(SIM.pick_skip(vals, skip_own)[0]),
        "hellsicht": int(SIM.pick_hindsight(vals)[0]),
    }
    if picks["unbekannt"] == -1:
        picks["unbekannt"] = None
    return Analysis(
        settings, n, rows, curve, skip_own, skip_opt,
        (float(curve["treffer"][skip_own]), float(curve["rang"][skip_own]), float(curve["wert"][skip_own])),
        s_rank, _metrics_of_policy(n, RK.skip_policy(n, skip_opt["treffer"])), _metrics_of_policy(n, s_rank), t_u, b, v, vals[0], picks,
    )


def row_of(a: Analysis, key: str) -> LadderRow:
    return next(r for r in a.rows if r.key == key)


# --------------------------------------------------------------------------------------------------- Simulation auf denselben Folgen
CROSS_RULES = (
    ("zufall", "Zufall (ohne Wissen)", "–"),
    ("rang_treffer", "Regel für Treffer", "Rangordnung"),
    ("rang_rang", "Regel für Rang", "Rangordnung"),
    ("rang_eigene", "Eigene Wartezeit", "Rangordnung"),
    ("werte_treffer", "Regel für Treffer", "Werte bekannt"),
    ("werte_wert", "Regel für Wert (Cayley)", "Werte bekannt"),
    ("unbekannt", "Regel für Treffer bei genau n Angeboten", "Anzahl unbekannt"),
    ("hellsicht", "Hellsicht", "–"),
)


def cross_exact(a: Analysis) -> dict[str, tuple[float | None, float | None, float | None]]:
    """Kennzahlen (Treffer, Rang, Wert) je Regel, soweit exakt bekannt, sonst None."""
    n = a.n
    return {
        "zufall": (1.0 / n, (n + 1) / 2.0, 0.5),
        "rang_treffer": a.best_rule,
        "rang_rang": a.rank_rule,
        "rang_eigene": a.own,
        "werte_treffer": (row_of(a, "werte").treffer, None, None),
        "werte_wert": (None, None, row_of(a, "werte").wert),
        "unbekannt": (UN.win_given_n(a.t_unknown, n), None, None),
        "hellsicht": (1.0, 1.0, n / (n + 1.0)),
    }


def simulate_cross(a: Analysis) -> dict[str, dict[str, tuple[float, float, int]]]:
    """Alle Regeln auf denselben RUNS Folgen (Seed und n der Einstellungen): {Regel: {Kennzahl: (Mittel, Standardfehler, Läufe mit Wahl)}}."""
    s = a.settings
    n = a.n
    vals = S.make_values(s.seed, n, C.RUNS)
    rel = SIM.relative_ranks(vals)
    from inl_rng import uniform_block
    chosen = {
        "zufall": SIM.pick_random(n, uniform_block(s.seed + 1, C.RUNS)),
        "rang_treffer": SIM.pick_skip(vals, a.skip_opt["treffer"]),
        "rang_rang": SIM.pick_rank_policy(vals, a.s_rank, rel),
        "rang_eigene": SIM.pick_skip(vals, a.skip_own),
        "werte_treffer": SIM.pick_value_win(vals, a.b),
        "werte_wert": SIM.pick_cayley(vals, a.v),
        "unbekannt": SIM.pick_unknown(vals, a.t_unknown),
        "hellsicht": SIM.pick_hindsight(vals),
    }
    return {key: SIM.summarize(SIM.score(vals, idx)) for key, idx in chosen.items()}


def simulate_ladder_unknown(a: Analysis) -> tuple[float, float]:
    """Anzahl unbekannt, Tage zufälliger Länge N ~ U{1..n}: (Anteil Treffer der Regel, Standardfehler)."""
    from inl_rng import uniform_block
    n = a.n
    vals = S.make_values(a.settings.seed, n, C.RUNS)
    lengths = 1 + np.minimum((uniform_block(a.settings.seed + 2, C.RUNS) * n).astype(np.int64), n - 1)
    p = SIM.simulate_unknown_days(vals, lengths, a.t_unknown)
    return p, float(np.sqrt(p * (1 - p) / C.RUNS))


# --------------------------------------------------------------------------------------------------- Messreihen der Experimente
def unknown_sweep(ms=C.SWEEP_MS) -> list[dict]:
    """Je Obergrenze m: optimale Trefferwahrscheinlichkeit bei unbekannter Anzahl, Einsatzzeitpunkt, dieselbe Regel bei genau N = m, das Optimum bei bekannter Anzahl m
    und das Optimum bei bekannter Anzahl über Tage mit denselben Längen N ~ U{1..m} (der faire Vergleich zur unbekannten Anzahl)."""
    rows = []
    for m in ms:
        p, t = UN.optimal_rule(m)
        known = RK.best_classic_curve(m)
        rows.append({"m": m, "p_unknown": p, "t_from": t, "p_at_m": UN.win_given_n(t, m), "p_known": float(known[-1]), "p_known_mixed": float(known.mean())})
    return rows


def envelope_grid() -> list[dict]:
    """Zwei Umschläge: Gewinnwahrscheinlichkeit für alle Beträge, Verhältnisse und Skalen der Zufallsschwelle (exakt)."""
    return [{"a": a, "ratio": r, "scale": sc, "p": EN.win_probability(a, a * r, sc), "excess": EN.excess(a, a * r, sc)}
            for sc in C.ENV_SCALES for a in C.ENV_AMOUNTS for r in C.ENV_RATIOS]


# --------------------------------------------------------------------------------------------------- Verdikte (reine Funktionen)
def verdict_ladder(goal: str, n: int, rows: tuple[LadderRow, ...], er_best_rule: float) -> tuple[str, str]:
    """Kernfrage: Was bringt mehr Wissen beim gewählten Ziel? Zahlen aus den exakten Zeilen der Leiter; `er_best_rule` = erwarteter Rang der Regel fürs höchste Angebot."""
    by = {r.key: r for r in rows}
    if goal == "treffer":
        p_r, p_w, p_z = by["rang"].treffer, by["werte"].treffer, by["zufall"].treffer
        gain = p_w - p_r
        kind = "success" if gain >= 0.05 else "info"
        return kind, (f"Bei {n} Angeboten findet die beste Regel mit bloßer Rangordnung das höchste Angebot mit **{pct(p_r)}**, mit bekannten Werten mit **{pct(p_w)}**: "
                      f"{pp(gain)} mehr. Ohne Wissen (Zufall) sind es {pct(p_z)}; nur die Hellsicht erreicht 100 %.")
    if goal == "rang":
        er = by["rang"].rang
        if er_best_rule - er < 0.05:
            return "info", (f"Bei {n} Angeboten liegt der erwartete Rang mit der besten Regel bei **{de(er, 2)}**; die Regel fürs höchste Angebot ergibt hier denselben Wert. "
                            "Für „Werte bekannt“ ist das Ziel Rang nicht gerechnet.")
        return "success", (f"Bei {n} Angeboten liegt der erwartete Rang des gewählten Angebots mit der besten Rangordnungs-Regel bei **{de(er, 2)}** (Zufall: {de(by['zufall'].rang, 1)}). "
                           f"Dieselbe Information, aber die Regel fürs höchste Angebot: Rang **{de(er_best_rule, 2)}**. Für „Werte bekannt“ ist das Ziel Rang nicht gerechnet.")
    w_r, w_w, w_h = by["rang"].wert, by["werte"].wert, by["hellsicht"].wert
    closed = (w_w - w_r) / (w_h - w_r) if w_h > w_r else 0.0
    return "success", (f"Bei {n} Angeboten erwartet man mit der Rangordnung allein **{pct(w_r)}** des möglichen Wertebereichs (Rate {euro(w_r)}), mit bekannten Werten **{pct(w_w)}** "
                       f"({euro(w_w)}). Hellsicht läge bei {pct(w_h)} ({euro(w_h)}); die bekannten Werte schließen {pct(closed, 0)} der Lücke zwischen Rangordnung und Hellsicht.")


def verdict_own(goal: str, n: int, skip_own: int, own: float, skip_opt: int, opt: float) -> tuple[str, str]:
    """Die eigene Wartezeit gegen die beste Wartezeit der Regel „ablehnen, dann Rekord“ für das Ziel."""
    if goal == "rang":
        gap = own - opt
        same = abs(gap) < 5e-4
        text = f"Ihre Wartezeit ({skip_own} von {n} abgelehnt) ergibt den erwarteten Rang {de(own, 2)}; die beste Wartezeit dieser Regelform ({skip_opt} abgelehnt) {de(opt, 2)}."
        return ("success", text + " Das ist die beste Wartezeit.") if same else ("info", text + f" Das sind {de(gap, 2)} Rangplätze mehr.")
    gap = opt - own
    article, name = ("die", "Trefferwahrscheinlichkeit") if goal == "treffer" else ("den", "erwarteten Wert")
    text = f"Ihre Wartezeit ({skip_own} von {n} abgelehnt) ergibt {article} {name} {pct(own)}; die beste Wartezeit dieser Regelform ({skip_opt} abgelehnt) {pct(opt)}."
    if gap < 5e-5:
        return "success", text + " Das ist die beste Wartezeit."
    return "info", text + f" Das sind {pp(gap)} weniger."


def verdict_goal_changes_rule(n: int, best_rule: tuple, rank_rule: tuple, skip_best: int, first_take_rank: int) -> tuple[str, str]:
    """Das Ziel ändert die Regel: die Regel fürs höchste Angebot (Wartezeit skip_best) gegen die Rang-optimale Regel, beide mit denselben Rangordnungs-Informationen.
    `first_take_rank`: ab diesem Zeitpunkt nimmt die Rang-optimale Regel überhaupt ein Angebot."""
    p1, r1, _ = best_rule
    p2, r2, _ = rank_rule
    if abs(p1 - p2) < 5e-4 and abs(r1 - r2) < 5e-3:
        return "info", f"Bei {n} Angeboten sind beide Regeln gleich: Treffer {pct(p1)}, erwarteter Rang {de(r1, 2)}."
    return "success", (f"Die Regel fürs höchste Angebot (erst {skip_best} ablehnen) trifft mit {pct(p1)}, das gewählte Angebot hat aber im Mittel Rang **{de(r1, 2)}**. "
                       f"Die Regel für den Rang trifft nur mit {pct(p2)}, liegt aber im Mittel bei Rang **{de(r2, 2)}**: sie nimmt erst ab Angebot {first_take_rank} überhaupt eines und später auch Angebote, die nur zweit- oder drittbeste bisher sind.")


def verdict_unknown(m: int, p_unknown: float, p_known_mixed: float, t_from: int, p_at_m: float, p_known_at_m: float) -> tuple[str, str]:
    """Nicht zu wissen, wie viele Angebote kommen. Fairer Vergleich: dieselben Tage (Länge N gleichverteilt auf 1..m), einmal mit unbekanntem, einmal mit bekanntem N (`p_known_mixed`, Mittel der besten Regel je N)."""
    return "info", (f"Weiß man nur, dass es höchstens {m} Angebote sind (jede Zahl von 1 bis {m} gleich wahrscheinlich), trifft die beste Regel mit **{pct(p_unknown)}**; kennt man die Zahl N jedes Tages, "
                    f"sind es auf denselben Tagen {pct(p_known_mixed)}: {pp(p_known_mixed - p_unknown)} weniger durch das Nichtwissen. Die Regel nimmt den ersten Rekord ab Angebot {t_from}; hat der Tag genau {m} Angebote, "
                    f"trifft sie mit {pct(p_at_m)} (die Regel, die N = {m} kennt: {pct(p_known_at_m)}).")


def verdict_envelopes(rows: list[dict]) -> tuple[str, str]:
    """Zwei Umschläge: Zählung über alle Zellen des Rasters (Beträge × Verhältnisse × Skalen). `excess` = Gewinnwahrscheinlichkeit minus 1/2, ohne Rundung auf 1/2 gerechnet."""
    ps = np.array([r["p"] for r in rows])
    positive = int(np.sum(np.array([r["excess"] for r in rows]) > 0))
    near = int(np.sum(np.array([r["excess"] for r in rows]) < 1e-3))
    text = (f"In {positive} von {len(ps)} Zellen (Beträge × Verhältnisse × Skalen der Schwelle) liegt die Gewinnwahrscheinlichkeit über 50 %; der größte Wert ist {pct(float(ps.max()))}. "
            f"In {near} Zellen ist der Abstand zu 50 % kleiner als 0,1 Prozentpunkte, weil beide Beträge weit oberhalb der Schwellenskala liegen: dort bleibt es praktisch beim Münzwurf.")
    return ("success" if positive == len(ps) else "warning"), text


def verdict_sweep(goal: str, sweep: dict[int, dict[str, float | None]]) -> tuple[str, str]:
    """Die Leiter über n (Zählung über die Zeilen der Messreihe). Treffer und Wert: Vorsprung der bekannten Werte vor der Rangordnung; Rang: Verlauf des Rang-Optimums."""
    ns = list(sweep)
    if goal == "rang":
        er = [sweep[n]["rang"] for n in ns]
        return "info", (f"Der kleinste erwartete Rang, den eine Regel mit bloßer Rangordnung erreichen kann, wächst mit n von {de(er[0], 2)} (n = {ns[0]}) auf {de(er[-1], 2)} (n = {ns[-1]}) und bleibt "
                        f"weit unter dem Zufall ({de(sweep[ns[-1]]['zufall'], 1)} bei n = {ns[-1]}). Für „Werte bekannt“ ist das Ziel Rang nicht gerechnet.")
    gains = {n: sweep[n]["werte"] - sweep[n]["rang"] for n in ns}
    n_max = max(gains, key=gains.get)
    n_min = min(gains, key=gains.get)
    what = "beim höchsten Angebot (Ziel Treffer)" if goal == "treffer" else "beim erwarteten Wert (Ziel Wert)"
    return "info", (f"Der Vorsprung der bekannten Werte vor der bloßen Rangordnung {what} liegt in den {len(ns)} gerechneten Angebotszahlen zwischen {de(100 * gains[n_min], 1)} Prozentpunkten "
                    f"(n = {n_min}) und {de(100 * gains[n_max], 1)} Prozentpunkten (n = {n_max}).")


def ladder_check_rows(a: Analysis, sim: dict, unknown_sim: tuple[float, float]) -> list[dict]:
    """Exakte Werte der Leiter (optimale Regel je Stufe für das Ziel des Analysis) gegen die Simulation auf RUNS Folgen: Zeilen {Stufe, exakt, Simulation, Standardfehler, Abstand in Standardfehlern}.

    Anzahl unbekannt: der exakte Wert gilt für N ~ U{1..n}, die Simulation zieht Tage dieser Länge (`unknown_sim`); Treffer ist die einzige gerechnete Kennzahl."""
    goal = a.settings.goal
    rule_of = {"treffer": {"rang": "rang_treffer", "werte": "werte_treffer"}, "rang": {"rang": "rang_rang"}, "wert": {"rang": "rang_rang", "werte": "werte_wert"}}[goal]
    out = []
    for stage, rule in rule_of.items():
        exact = row_of(a, stage).get(goal)
        mean, se, _ = sim[rule][goal]
        out.append({"stage": STAGE_NAMES[stage], "exact": exact, "sim": mean, "se": se, "z": (mean - exact) / se if se > 0 else 0.0})
    if goal == "treffer":
        exact = row_of(a, "unbekannt").treffer
        mean, se = unknown_sim
        out.append({"stage": STAGE_NAMES["unbekannt"], "exact": exact, "sim": mean, "se": se, "z": (mean - exact) / se if se > 0 else 0.0})
    return out


def verdict_values_rank(er_values_rule: float, se: float, er_rank_opt: float) -> tuple[str, str]:
    """Ziel Rang bei bekannten Werten: die Regel für das Ziel Wert (Cayley) gegen das Optimum der Rangordnungs-Stufe. Der Wert der Cayley-Regel ist eine Simulation (Mittel ± Standardfehler)."""
    if er_values_rule + 2 * se < er_rank_opt:
        return "success", (f"Die Regel für das Ziel Wert (bekannte Werte) erreicht in der Simulation im Mittel Rang **{de(er_values_rule, 2)}** ± {de(se, 2)}, besser als das Optimum der bloßen Rangordnung "
                           f"(**{de(er_rank_opt, 2)}**, exakt). Die beste Regel für Rang bei bekannten Werten ist hier nicht bestimmt (offen, laut Suchtreffern als Robbins-Problem bekannt); diese Zahl gehört zu einer Regel, nicht zum Optimum.")
    return "info", (f"Die Regel für das Ziel Wert (bekannte Werte) erreicht in der Simulation Rang {de(er_values_rule, 2)} ± {de(se, 2)}, das Optimum der bloßen Rangordnung liegt bei {de(er_rank_opt, 2)} (exakt); "
                    "ein Vorsprung der bekannten Werte ist in dieser Stichprobe nicht zu sehen.")


def verdict_thresholds(n: int, b: np.ndarray, v: np.ndarray) -> tuple[str, str]:
    """Die Schwellen der Regeln für bekannte Werte in Euro: erstes und letztes Angebot für beide Ziele (b[k], v[k] mit k = Zahl der danach noch kommenden Angebote)."""
    if n < 2:
        return "info", "Mit einem Angebot gibt es nichts zu entscheiden."
    first = (f"Beim ersten von {n} Angeboten nimmt die Regel für Treffer nur einen Rekord ab {euro(b[n - 1])} und die Regel für Wert ein Angebot ab {euro(v[n - 1])}. ")
    last = f"beim letzten wird jedes Angebot genommen (Schwelle {euro(0.0)}), falls vorher keines genommen wurde."
    if n > 2:
        return "info", first + f"Beim vorletzten Angebot sinken die Schwellen auf {euro(b[1])} und {euro(v[1])}, " + last
    return "info", first + "B" + last[1:]
