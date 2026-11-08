"""Informationsleiter: Wie viel ist Wissen beim Auswählen wert? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 4 der Rätsel-Reihe (Arbeitstitel „Teamrätsel“), das Sekretärinnenproblem ausgebaut. Ein Disponent sieht Spotfracht-Angebote nacheinander und muss jedes sofort annehmen oder
ablehnen. Dieselbe zufällige Folge geht durch mehrere Wissensstufen (nur Rangordnung, Werte bekannt, Anzahl unbekannt) und wird an drei Zielen gemessen (höchstes Angebot,
kleiner erwarteter Rang, hoher erwarteter Wert). Unterste Sprosse: das Umschlag-Spiel mit Zufallsschwelle.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import inl_constants as C
import inl_envelopes as EN
import inl_evaluation as V
import inl_rank as RK
import inl_scenario as S
import inl_unknown as UN
import inl_visualization as Z
from inl_layout import dunkles_design, kopf_css
from inl_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Informationsleiter: Wie viel ist Wissen beim Auswählen wert? – Sebastian Hanisch", layout="wide")


# --- Zwischenspeicher (Schlüssel = alle Eingaben inklusive Zufalls-Seed) --------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _analyse(s: S.Settings) -> V.Analysis:
    return V.analyse(s)


@st.cache_data(show_spinner=False)
def _sweep(goal: str):
    return V.ladder_sweep(goal)


@st.cache_data(show_spinner=False)
def _sim_cross(s: S.Settings):
    return V.simulate_cross(_analyse(s))


@st.cache_data(show_spinner=False)
def _sim_unknown(s: S.Settings):
    return V.simulate_ladder_unknown(_analyse(s))


@st.cache_data(show_spinner=False)
def _unknown_sweep():
    return V.unknown_sweep()


@st.cache_data(show_spinner=False)
def _known_curve(m: int):
    return RK.best_classic_curve(m)


@st.cache_data(show_spinner=False)
def _envelope_grid():
    return V.envelope_grid()


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


def table(rows):
    st.dataframe(rows, hide_index=True, width="stretch")


def fmt(goal: str, x, nd: int = 1) -> str:
    """Kennzahl für das Ziel: Rang als Zahl mit Komma, Treffer und Wert als Prozent; None = nicht gerechnet."""
    if x is None:
        return "–"
    return V.de(x, 2) if goal == "rang" else V.pct(x, nd)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in inl_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🪜 Informationsleiter: Wie viel ist Wissen beim Auswählen wert?")
st.markdown(
    """
**Stück 4 der Rätsel-Reihe** (Arbeitstitel „Teamrätsel“: Aufgaben mit abgesprochener Strategie unter beschränkter Information). Ein Disponent sieht **Spotfracht-Angebote** eines Tages **nacheinander** und muss jedes sofort
annehmen oder ablehnen; ein abgelehntes Angebot ist weg. **Dieselbe zufällige Folge** geht durch mehrere **Wissensstufen**: Er kennt nur die **Rangordnung** (jedes Angebot nur im Vergleich mit den bisherigen),
er kennt die **Werte** (gleichverteilte Raten, bekannt), oder er weiß nicht einmal, **wie viele** Angebote kommen. Gemessen wird an einem wählbaren **Ziel**: das **höchste** Angebot (Treffer), ein
**kleiner erwarteter Rang** oder ein **hoher erwarteter Wert**. Die Frage: **Wie viel bringt zusätzliches Wissen, und ändert das Ziel die Regel?** Eine Stufe tiefer steht das Umschlag-Spiel (n = 2).

Abgrenzung: `routenresilienz-demo` entscheidet mit bekannter Risikodynamik per Bellman über Periode und Risikostufe, `online-matching-demo` weist viele Aufträge unwiderruflich zu und vergleicht mit dem Offline-Optimum; hier wird **eine** Wahl
getroffen, und verglichen wird, **was man weiß**. Die Einzelresultate sind Lehrbuchwissen; Neues ist der Vergleich der Stufen auf einer Folge. **Alle Regeln sind Modellannahmen**, keine Aussage über Menschen oder echte Spotmärkte.

Weiter unten: fünf Tabs mit je einem **🔬 Experiment**, die Annahmen-Tabelle **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Das Spiel in einem Absatz", expanded=True):
    st.markdown(
        r"""
1. **Angebote.** An einem Tag treffen $n$ Angebote in zufälliger Reihenfolge ein. Jedes ist eine Frachtrate; die Werte sind unabhängig und gleichverteilt zwischen 800 € und 1.200 € (0 bis 1 im Rechenmodell).
   Pro Angebot gibt es genau eine Entscheidung: nehmen (dann ist der Tag zu Ende) oder ablehnen (dann ist es weg). Das letzte Angebot muss genommen werden, falls vorher nichts genommen wurde.
2. **Nur Rangordnung.** Nach dem $k$-ten Angebot weiß man nur, auf welchem Platz es unter den ersten $k$ liegt (1 = am höchsten). Die klassische Regel: die ersten $s$ ablehnen, dann das erste nehmen,
   das alle bisherigen übertrifft („Rekord“). Für das Ziel Treffer ist $s \approx n/e$ (37 von 100) optimal.
3. **Werte bekannt.** Man kennt die Rate jedes Angebots und ihre Verteilung. Dann entscheidet eine **Schwelle**, die mit den noch kommenden Angeboten sinkt.
4. **Anzahl unbekannt.** Man weiß nur, dass es höchstens $m$ Angebote sind (jede Zahl von 1 bis $m$ gleich wahrscheinlich). Endet der Tag, ohne dass man genommen hat, gibt es keinen Treffer.
5. **Ziele.** *Treffer*: das höchste Angebot nehmen. *Rang*: der erwartete Platz des genommenen Angebots, kleiner ist besser. *Wert*: die erwartete Rate des genommenen Angebots. Hellsicht (im Nachhinein das höchste nehmen)
   ist die obere Schranke, keine umsetzbare Regel.
        """
    )

with st.expander("Erklärtabelle: was die Regler und Auswahlen tun"):
    st.markdown(
        """
| Regler / Auswahl | Was er tut | Wirkung auf die Frage |
|---|---|---|
| **Zahl der Angebote n** | Länge des Tages. | Wie sich der Abstand zwischen den Stufen mit n ändert, zeigt E2. |
| **Ziel** | Treffer, Rang oder Wert: woran die Regeln gemessen werden. | Ändert die optimale Regel und die Zahl, die man erreicht (E3). Nicht jede Kombination von Stufe und Ziel ist gerechnet: „–“. |
| **Eigene Wartezeit (Prozent)** | Anteil der Angebote, die Sie ablehnen, bevor Sie den ersten Rekord nehmen (abgerundet auf ganze Angebote). | Zeigt die Kurve der Regel „ablehnen, dann Rekord“ und wie nah Ihre Wahl am Optimum liegt. |
| **Zufalls-Seed** | Welche Folge gezeigt wird (die erste von 2.000 Folgen der Simulation). | Ein Tag ist ein Einzelfall: die Zahlen darunter sind Erwartungswerte. |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🚛 Angebote und Ziel")
    n = st.slider("Zahl der Angebote n", *bounds("n_slider"), key="n_slider", step=1,
                  help="Wie viele Spotfracht-Angebote der Tag hat. Bei unbekannter Anzahl ist n die Obergrenze m.")
    goal = st.selectbox("Ziel", C.GOALS, key="goal_select", format_func=lambda g: C.GOAL_LABELS[g],
                        help="Woran die Regeln gemessen werden: das höchste Angebot nehmen, kleiner erwarteter Rang des genommenen Angebots oder hohe erwartete Rate.")
    st.header("🪜 Eigene Regel")
    wait = st.slider("Wartezeit (Prozent der Angebote ablehnen)", *bounds("wait_slider"), key="wait_slider", step=1,
                     help="Die ersten so vielen Prozent der Angebote ablehnen (abgerundet), dann das erste nehmen, das alle bisherigen übertrifft. 37 Prozent ist bei 100 Angeboten die beste Wartezeit für das Ziel Treffer.")
    st.header("🎲 Folge")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "goal_select": goal, "wait_slider": int(wait), "seed_input": int(seed)})

settings = S.Settings(n=int(n), goal=goal, wait=int(wait), seed=int(seed))
a = _analyse(settings)
verdicts = []   # Verdikte für den PDF-Export

# --- Hauptansicht --------------------------------------------------------------------------------------------------------------------------------
st.markdown("## 📊 Ein Tag, mehrere Wissensstufen")
st.caption(f"Ziel: **{C.GOAL_LABELS[goal]}**. Die Kennzahlen sind **Erwartungswerte über alle Folgen** (exakt gerechnet), nicht das Ergebnis des gezeigten Tages.")
m1, m2, m3, m4 = st.columns(4)
by = {r.key: r for r in a.rows}
unit_help = {"treffer": "Wahrscheinlichkeit, das höchste der Angebote zu nehmen.", "rang": "Erwarteter Platz des genommenen Angebots unter allen (1 = das höchste).",
             "wert": "Erwartete Rate des genommenen Angebots als Anteil des Wertebereichs (0 = 800 €, 1 = 1.200 €)."}[goal]
m1.metric("Zufall (ohne Wissen)", fmt(goal, by["zufall"].get(goal)), help=unit_help + " Ein gleichverteilt gezogenes Angebot.")
m2.metric("Anzahl unbekannt", fmt(goal, by["unbekannt"].get(goal)), help=unit_help + f" Nur Rangordnung, die Zahl der Angebote ist gleichverteilt auf 1 bis {a.n}. Die Tage sind im Mittel kürzer als n (E5 vergleicht fair). Nur für das Ziel Treffer gerechnet.")
m3.metric("Nur Rangordnung", fmt(goal, by["rang"].get(goal)), help=unit_help + " Beste Regel, die nur Relativränge benutzt.")
m4.metric("Werte bekannt", fmt(goal, by["werte"].get(goal)), help=unit_help + " Beste Regel bei bekannter Gleichverteilung. Für das Ziel Rang nicht gerechnet (Robbins-Problem).")

st.markdown("#### ❓ Kernfrage: Was bringt mehr Wissen für dieses Ziel?")
v_main = V.verdict_ladder(goal, a.n, a.rows, a.best_rule[1])
show(v_main)
verdicts.append(v_main)

st.plotly_chart(Z.build_ladder_bars(a.rows, goal), width="stretch", key="ladder_chart")
st.caption("Die Stufen für das gewählte Ziel, jeweils mit der besten Regel der Stufe. Stufen ohne Säule sind für dieses Ziel nicht gerechnet. Die Hellsicht ist die obere Schranke. "
           "„Anzahl unbekannt“ hat Tage der Länge 1 bis n und ist darum nicht Schritt für Schritt mit n bekannten Angeboten vergleichbar: der faire Vergleich steht in E5.")

st.markdown("#### 🚛 Der gezeigte Tag")
st.plotly_chart(Z.build_sequence(a.values, a.picks, wait_boundary=a.skip_own if a.skip_own > 0 else None), width="stretch", key="sequence_chart")
vals = a.values
rows_day = []
labels = {"rang": "Nur Rangordnung", "werte": "Werte bekannt", "unbekannt": "Anzahl unbekannt", "eigene": "Eigene Wartezeit", "hellsicht": "Hellsicht (höchstes Angebot)"}
for key, label in labels.items():
    idx = a.picks.get(key)
    if idx is None:
        rows_day.append({"Stufe": label, "Gewähltes Angebot": "–", "Rate": "–", "Rang an diesem Tag": "–"})
    else:
        rows_day.append({"Stufe": label, "Gewähltes Angebot": f"{idx + 1} von {a.n}", "Rate": V.euro(vals[idx]), "Rang an diesem Tag": int(1 + np.sum(vals > vals[idx]))})
table(rows_day)
st.caption("Die gestrichelte senkrechte Linie trennt die abgelehnten Angebote Ihrer Wartezeit. Die Regeln der Stufen gehören zum gewählten Ziel; „–“ heißt: für diese Kombination keine Regel gerechnet "
           "(bei Anzahl unbekannt nur Ziel Treffer, bei Werte bekannt nicht das Ziel Rang) oder die Regel hat keinen Rekord gefunden. Ein einzelner Tag zeigt nur, wie sich die Regeln hier verhalten haben.")

st.markdown("#### 🪜 Ihre Wartezeit gegen die beste")
refs = {}
if goal == "treffer":
    refs = {f"Werte bekannt {V.pct(by['werte'].treffer)}": by["werte"].treffer}
elif goal == "rang":
    refs = {f"beste Regel für Rang {V.de(by['rang'].rang, 2)}": by["rang"].rang}
else:
    refs = {f"Werte bekannt {V.pct(by['werte'].wert)}": by["werte"].wert, f"beste Rangordnungs-Regel {V.pct(by['rang'].wert)}": by["rang"].wert}
st.plotly_chart(Z.build_skip_curve(a.curve, goal, a.skip_own, a.skip_opt[goal], refs), width="stretch", key="skip_chart")
own_val = a.own[C.GOALS.index(goal)]
opt_val = float(a.curve[goal][a.skip_opt[goal]])
v_own = V.verdict_own(goal, a.n, a.skip_own, own_val, a.skip_opt[goal], opt_val)
show(v_own)
verdicts.append(v_own)
st.caption("Exakt für die Regel „erst ablehnen, dann Rekord“. Für die Ziele Rang und Wert ist die beste Regel der Stufe mindestens so gut wie jede Wartezeit-Regel und bei n = 100 deutlich besser (Bezugslinie im Diagramm): sie nimmt später auch Angebote, die nur zweit- oder drittbeste bisher sind (E3).")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1 Einstieg: zwei Umschläge", "2 Die Leiter über n", "3 Das Ziel ändert die Regel", "4 Die Schwellen", "5 Anzahl unbekannt"])

# --- Tab 1 ---------------------------------------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("🔬 Experiment E1: Zwei Umschläge, die unterste Sprosse")
    st.caption("Eine andere Aufgabenform als der Rest: zwei Umschläge mit festen, unbekannten Beträgen a < b, kein Wissen über deren Verteilung. Man öffnet einen (zufällig), sieht den Betrag und darf wechseln. "
               "Mit einer **Zufallsschwelle** T (hier exponentialverteilt, also mit überall positiver Dichte) und der Regel „wechseln, wenn der gesehene Betrag unter T liegt“ gewinnt man mit **1/2 + 1/2 · P(a < T < b)**: für jedes Paar mehr als die Hälfte (Cover). "
               "Der Gewinn über 1/2 ist umso kleiner, je weiter beide Beträge außerhalb dessen liegen, was die Schwelle trifft.")
    e1a, e1b, e1c = st.columns(3)
    amount = e1a.selectbox("Kleinerer Betrag a", C.ENV_AMOUNTS, index=1, key="e1_amount")
    ratio = e1b.selectbox("Verhältnis b / a", C.ENV_RATIOS, index=1, key="e1_ratio")
    scale = e1c.selectbox("Mittel der Schwelle", C.ENV_SCALES, index=1, key="e1_scale")
    b_amount = amount * ratio
    p_env = EN.win_probability(amount, b_amount, scale)
    st.metric(f"Gewinnwahrscheinlichkeit bei a = {amount}, b = {V.de(b_amount, 1) if ratio % 1 else int(b_amount)}", V.pct(p_env, 2),
              help="1/2 + 1/2 · (e^(−a/Mittel) − e^(−b/Mittel)), exakt für eine exponentialverteilte Schwelle.")
    st.plotly_chart(Z.build_envelopes(float(ratio)), width="stretch", key="e1_chart")
    grid = _envelope_grid()
    def ratio_label(r):
        return f"b = {V.de(r, 1) if r % 1 else int(r)} · a"
    table([{"Mittel der Schwelle": sc, "kleinerer Betrag a": av,
            **{ratio_label(r): V.pct(next(x["p"] for x in grid if x["a"] == av and x["ratio"] == r and x["scale"] == sc), 2) for r in C.ENV_RATIOS}} for sc in C.ENV_SCALES for av in C.ENV_AMOUNTS])
    v1 = V.verdict_envelopes(grid)
    show(v1)
    verdicts.append(v1)
    r2 = V.ladder(2)
    by2 = {r.key: r for r in r2}
    st.markdown(f"**Zum Vergleich auf der Leiter bei n = 2:** nur Rangordnung **{V.pct(by2['rang'].treffer)}**, Werte bekannt (gleichverteilt, bekannt) **{V.pct(by2['werte'].treffer)}**. Das Umschlag-Spiel kennt keine Verteilung der Beträge, "
                "setzt aber eine Zufallsschwelle ein; je nach Betragspaar liegt sein Gewinn zwischen 50 % und über 80 % (siehe Tabelle oben), seine Garantie gilt für jedes feste Paar, die Leiter-Zahlen sind Mittel über die angenommene Verteilung. Die Zahlen sind nicht mit den Leiterwerten vergleichbar.")
    st.markdown("**Nachrechnen** (Betrag, Verhältnis und Schwelle von oben; 100.000 Läufe mit dem Zufalls-Seed der Seitenleiste)")
    if st.button("▶️ Umschläge simulieren", key="e1_button", help=f"{V.thousands(C.ENV_RUNS)} Läufe; rechnet rund eine Sekunde."):
        st.session_state["e1_signature"] = (settings.seed, amount, ratio, scale)
    if st.session_state.get("e1_signature") == (settings.seed, amount, ratio, scale):
        with st.spinner(f"Simuliere {V.thousands(C.ENV_RUNS)} Läufe …"):
            sim_p = EN.simulate(float(amount), float(b_amount), float(scale), C.ENV_RUNS, settings.seed)
        se = (sim_p * (1 - sim_p) / C.ENV_RUNS) ** 0.5
        table([{"exakt": V.pct(p_env, 2), "Simulation": V.pct(sim_p, 2), "Standardfehler": V.pct(se, 2), "Abstand in Standardfehlern": V.de((sim_p - p_env) / se, 1)}])
        st.caption("Bei kleinem Abstand zu 50 % kann die Simulation den Gewinn nicht von einem Münzwurf trennen: dort ist der Standardfehler größer als der Gewinn.")

# --- Tab 2 ---------------------------------------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("🔬 Experiment E2: Die Leiter über die Zahl der Angebote")
    st.caption("Alle Stufen für alle Ziele bei der Zahl n der Seitenleiste (exakt), und das gewählte Ziel über n. Nicht gerechnet („–“): Anzahl unbekannt für Rang und Wert, Werte bekannt für den Rang.")
    table([{"Stufe": r.name, "Treffer": fmt("treffer", r.treffer), "erwarteter Rang": fmt("rang", r.rang), "erwarteter Wert": fmt("wert", r.wert)} for r in a.rows])
    sweep = _sweep(goal)
    st.plotly_chart(Z.build_sweep(sweep, goal), width="stretch", key="e2_chart")
    table([{"n": nn, **{V.STAGE_NAMES[k]: fmt(goal, x) for k, x in vals_n.items()}} for nn, vals_n in sweep.items()])
    v2 = V.verdict_sweep(goal, sweep)
    show(v2)
    verdicts.append(v2)
    st.markdown(f"**Nachrechnen** (Zahl der Angebote, Ziel und Zufalls-Seed der Seitenleiste; {V.thousands(C.RUNS)} Folgen, die gezeigte Folge ist die erste davon)")
    if st.button("▶️ Leiter auf 2.000 Folgen nachmessen", key="e2_button", help=f"{V.thousands(C.RUNS)} Folgen durch die Regeln der Stufen; rechnet etwa eine Sekunde."):
        st.session_state["e2_signature"] = settings
    if st.session_state.get("e2_signature") == settings:
        with st.spinner(f"Rechne {V.thousands(C.RUNS)} Folgen …"):
            sim = _sim_cross(settings)
            sim_u = _sim_unknown(settings)
        check = V.ladder_check_rows(a, sim, sim_u)
        table([{"Stufe": r["stage"], "exakt": fmt(goal, r["exact"], 2), "Simulation": fmt(goal, r["sim"], 2), "Standardfehler": fmt(goal, r["se"], 2), "Abstand in Standardfehlern": V.de(r["z"], 1)} for r in check])
        st.caption(f"{V.thousands(C.RUNS)} Folgen je Zeile. Abweichungen von ein bis zwei Standardfehlern sind bei dieser Stichprobe zu erwarten, die exakte Zahl ist die Aussage. "
                   "Anzahl unbekannt: Tage der zufälligen Länge 1 bis n.")

# --- Tab 3 ---------------------------------------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("🔬 Experiment E3: Das Ziel ändert die Regel")
    st.caption("Dieselbe Information, zwei Ziele: Die Regel, die das höchste Angebot sucht, und die Regel für einen kleinen erwarteten Rang unterscheiden sich ab n = 4 (bei n = 2 und n = 3 sind beide gleich), und keine schneidet am Ziel der anderen besser ab (bei n = 100 deutlich schlechter, siehe Verdikt). "
               "Zahlen für die Rangordnungs-Regeln sind exakt; bei den Regeln für bekannte Werte ist nur die Diagonale exakt, die übrigen Zellen kommen aus der Simulation (Knopf).")
    exact = V.cross_exact(a)
    sim3 = _sim_cross(settings) if st.session_state.get("e3_signature") == settings else None
    cross_rows = []
    for key, rule, stage in V.CROSS_RULES:
        ex = exact[key]
        row = {"Stufe": stage, "Regel": rule, "Treffer": fmt("treffer", ex[0]), "erwarteter Rang": fmt("rang", ex[1]), "erwarteter Wert": fmt("wert", ex[2])}
        if sim3 is not None:
            for g, label in (("treffer", "Treffer (Sim.)"), ("rang", "Rang (Sim.)"), ("wert", "Wert (Sim.)")):
                mean, se, cnt = sim3[key][g]
                row[label] = "–" if np.isnan(mean) else fmt(g, mean, 1) + f" ± {fmt(g, se, 1) if g == 'rang' else V.de(100 * se, 1)}"
        cross_rows.append(row)
    table(cross_rows)
    st.caption("Die Zeile „Eigene Wartezeit“ folgt der Seitenleiste. Anzahl unbekannt: die Regel für Obergrenze n an einem Tag mit genau n Angeboten; Rang und Wert sind dort nicht gerechnet. "
               "Erwarteter Rang des Zufalls: (n + 1)/2; erwarteter Wert des Zufalls: 1/2.")
    first_take = int(next((k for k in range(1, a.n + 1) if a.s_rank[k] >= 1), a.n))
    v3 = V.verdict_goal_changes_rule(a.n, a.best_rule, a.rank_rule, a.skip_opt["treffer"], first_take)
    show(v3)
    verdicts.append(v3)
    st.markdown("**Nachrechnen** (Zahl der Angebote, Ziel, Wartezeit und Zufalls-Seed der Seitenleiste; alle Regeln auf denselben 2.000 Folgen, die Zellen der Tabelle oben kommen hinzu)")
    if st.button("▶️ Alle Regeln auf 2.000 Folgen messen", key="e3_button", help=f"{V.thousands(C.RUNS)} Folgen durch alle acht Regeln; rechnet etwa eine Sekunde."):
        st.session_state["e3_signature"] = settings
        st.rerun()
    if sim3 is not None:
        mean_r, se_r, _ = sim3["werte_wert"]["rang"]
        v3b = V.verdict_values_rank(mean_r, se_r, a.rank_rule[1])
        show(v3b)
        verdicts.append(v3b)
        st.caption(f"{V.thousands(C.RUNS)} Folgen, Mittel ± Standardfehler des Mittels; Treffer in Prozent mit Prozentpunkten als Fehler. Rang und Wert nur über Läufe, in denen eine Wahl getroffen wurde (bei Anzahl unbekannt nicht immer).")

# --- Tab 4 ---------------------------------------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("🔬 Experiment E4: Wie sehen die Regeln aus?")
    st.caption("Die Regeln als Schwellen, für die Zahl n der Seitenleiste. Links nur Rangordnung: bis zu welchem Relativrang wird zum Zeitpunkt k genommen. Rechts Werte bekannt: ab welcher Rate wird genommen. "
               "Alle Schwellen sind exakt gerechnet (Rückwärts-DP bzw. Wurzel einer Gleichung).")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(Z.build_rank_thresholds(a.s_rank, a.skip_opt["treffer"]), width="stretch", key="e4_rank_chart")
        st.caption("Regel für Rang: je später, desto großzügiger nimmt sie auch Angebote mit Relativrang 2, 3 und mehr. Regel für Treffer: nur Rekorde, erst nach der Wartezeit.")
    with c2:
        st.plotly_chart(Z.build_value_thresholds(a.b, a.v), width="stretch", key="e4_value_chart")
        st.caption("Die Schwelle sinkt mit den noch kommenden Angeboten. Treffer: nur Rekorde ab dieser Rate. Wert: jedes Angebot ab dieser Rate.")
    pos = sorted({1, max(1, a.n // 4), max(1, a.n // 2), max(1, (3 * a.n) // 4), max(1, a.n - 1), a.n})
    table([{"Angebot t": t, "Rang-Regel: genommen bei Relativrang bis": int(a.s_rank[t]), "Treffer-Regel: Rekord ab": V.euro(a.b[a.n - t]), "Wert-Regel: Angebot ab": V.euro(a.v[a.n - t])} for t in pos])
    v4 = V.verdict_thresholds(a.n, a.b, a.v)
    show(v4)
    verdicts.append(v4)

# --- Tab 5 ---------------------------------------------------------------------------------------------------------------------------------
with tab5:
    st.subheader("🔬 Experiment E5: Die Anzahl der Angebote ist unbekannt")
    st.caption("Man weiß nur: höchstens m Angebote, jede Zahl von 1 bis m gleich wahrscheinlich (m ist hier die Zahl n der Seitenleiste). Nur Rangordnung, Ziel Treffer. Die beste Regel nimmt den ersten Rekord "
               "ab einem festen Zeitpunkt; Tage, die vor einem Rekord enden, sind verloren.")
    p_u, t_u = UN.optimal_rule(a.n)
    known_curve = _known_curve(a.n)
    unknown_curve = UN.win_curve(t_u, a.n)
    st.plotly_chart(Z.build_unknown_curve(unknown_curve, known_curve, t_u), width="stretch", key="e5_chart")
    sw = _unknown_sweep()
    table([{"Obergrenze m": r["m"], "Anzahl unbekannt: Treffer": V.pct(r["p_unknown"]), "Anzahl bekannt, dieselben Tage": V.pct(r["p_known_mixed"]), "nimmt Rekord ab Angebot": r["t_from"],
            "dieselbe Regel bei genau N = m": V.pct(r["p_at_m"]), "Regel mit bekanntem N = m": V.pct(r["p_known"])} for r in sw])
    v5 = V.verdict_unknown(a.n, p_u, float(known_curve.mean()), t_u, UN.win_given_n(t_u, a.n), float(known_curve[-1]))
    show(v5)
    verdicts.append(v5)
    st.markdown("**Nachrechnen** (m = n und Zufalls-Seed der Seitenleiste; 2.000 Tage zufälliger Länge)")
    if st.button("▶️ Tage zufälliger Länge simulieren", key="e5_button", help=f"{V.thousands(C.RUNS)} Tage; rechnet unter einer Sekunde."):
        st.session_state["e5_signature"] = settings
    if st.session_state.get("e5_signature") == settings:
        with st.spinner(f"Simuliere {V.thousands(C.RUNS)} Tage …"):
            p_sim, se_sim = _sim_unknown(settings)
        table([{"exakt": V.pct(p_u), "Simulation": V.pct(p_sim), "Standardfehler": V.pct(se_sim), "Abstand in Standardfehlern": V.de((p_sim - p_u) / se_sim, 1) if se_sim > 0 else "–"}])
        st.caption(f"{V.thousands(C.RUNS)} Tage mit gleichverteilter Länge 1 bis {a.n}.")

# --- Verwandte Demos -----------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 Verwandte Demos")
st.markdown(
    """
- **`routenresilienz-demo`**: optimales Stoppen mit **bekannter** Risikodynamik (Bellman über Periode und Risikostufe); hier gibt es keine Dynamik, nur die Frage, was man über die Angebote weiß.
- **`online-matching-demo`**: Aufträge kommen nacheinander in Zufallsreihenfolge, die Zusage ist unwiderruflich, viele Paare statt einer Wahl, Vergleich mit dem Offline-Optimum; hier eine einzige Wahl.
- **`revenue-management-demo`**: das Hindsight-Orakel als obere Schranke, dasselbe Muster wie die Hellsicht hier.
"""
)

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Raten sind unabhängig und gleichverteilt**, die Verteilung ist bekannt (Stufe „Werte bekannt“ und Ziel Wert) | Die Zahlen der Stufen „Werte bekannt“ und der Zielmaße Wert gelten für diese Verteilung. Eine falsch angenommene Verteilung ist **nicht gemessen**. Die Stufe „Nur Rangordnung“ braucht nur unabhängige Werte aus derselben Verteilung. | – |
| **Zufällige Reihenfolge**: jede Reihenfolge der Angebote ist gleich wahrscheinlich | Steigen oder fallen die Raten im Lauf des Tages, gelten die Zahlen dieser Demo nicht mehr; wie stark sie abweichen, ist nicht gemessen. | – |
| **Keine Rückfrage**: ein abgelehntes Angebot ist weg | Mit Rückruf (das abgelehnte Angebot gilt noch mit einer Wahrscheinlichkeit) ändern sich Regeln und Zahlen. Nur als **Ausblick**, nicht gerechnet. | – |
| **Genau ein Angebot wird gewählt**, ohne Kosten je Ansehen | Mehrere Plätze und Kosten je Ansehen verschieben die Regeln. Nur als **Ausblick**, nicht gerechnet. | `online-matching-demo` (viele Zuordnungen) |
| **Ziel „guter Preis statt bester“** (Ziel Wert) | Das Ziel Treffer („das beste oder nichts“) ist das Lehrbuchziel, das Ziel Wert bewertet auch gute, aber nicht beste Angebote. Beides ist ein Modell, keine Aussage darüber, wie Disponenten entscheiden. | – |
| **Die Zahl der Angebote ist bekannt** (außer der Stufe „Anzahl unbekannt“) | Bei unbekannter Zahl sinkt die Trefferwahrscheinlichkeit (E5); die Gleichverteilung der Zahl ist eine Modellannahme. | – |
| **Ziel Rang bei bekannten Werten** | Die beste Regel für den erwarteten Rang bei bekannten Werten ist ein offenes Problem (laut Titeln der Suchtreffer als Robbins-Problem bekannt, nicht im Volltext gelesen), hier nicht gerechnet. Die Simulation in E3 zeigt nur, dass eine Regel für das Ziel Wert dort im Mittel besser abschneidet als das Optimum der Rangordnung. | – |
| **Umschlag-Spiel** | Die Garantie über 50 % gilt für jedes feste Paar, wenn die Zufallsschwelle jede Zahl treffen kann; die Höhe hängt an Schwelle und Beträgen. Es ist eine andere Aufgabenform als die Leiter. | – |
| **Lehrbuchresultate** | Die Einzelergebnisse sind bekannt (siehe README). Hier wurden sie neu gerechnet und mit einem zweiten Rechenweg geprüft (ausgenommen die Simulation zum Rang bei bekannten Werten in E3 und die Literatur-Grenzwerte); das Neue ist der Vergleich der Stufen auf einer Folge. | – |
    """
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Angebote in zufälliger Reihenfolge, die nacheinander angenommen oder abgelehnt werden; gesucht sind die Regeln und ihre Kennzahlen je Wissensstufe und Ziel.
Die Herleitungen sind Lehrbuchwissen, hier mit eigener Rechnung, Aufzählung und Simulation geprüft, nicht gegen die Originalarbeiten gelesen.

| Symbol | Bedeutung |
|---|---|
| $n,\ m$ | Zahl der Angebote; Obergrenze bei unbekannter Zahl |
| $k,\ t$ | Zeitpunkt des Angebots |
| $j$ | Relativrang unter den ersten $k$ (1 = höchstes) |
| $s$ | Zahl der abgelehnten Angebote (Wartezeit) |
| $R$ | absoluter Rang des genommenen Angebots (1 = höchstes) |
| $x$ | Wert eines Angebots, gleichverteilt auf $[0, 1]$ |
| $b_k,\ v_k$ | Schwellen bei $k$ noch kommenden Angeboten (Ziel Treffer bzw. Wert) |
| $W_k$ | erwarteter Endrang beim Weitermachen nach $k$ abgelehnten Angeboten |
| $N$ | tatsächliche Zahl der Angebote bei unbekannter Zahl |

**Nur Rangordnung, Ziel Treffer.** Erst $s$ ablehnen, dann den ersten Rekord nehmen; der Rekord zur Zeit $k$ ist mit Wahrscheinlichkeit $k/n$ das höchste Angebot insgesamt:

$$P(\text{Treffer}) = \frac{s}{n} \sum_{k=s+1}^{n} \frac{1}{k-1}$$

**Nur Rangordnung, Ziel Rang.** Ein Angebot mit Relativrang $j$ zur Zeit $k$ hat im Mittel den absoluten Rang

$$E[R \mid j, k] = \frac{j\,(n+1)}{k+1}$$

und die Rückwärts-DP (Chow, Moriguti, Robbins, Samuels) wählt zwischen Nehmen und Weitermachen, mit $t = k+1$:

$$W_k = \frac{1}{t} \sum_{j=1}^{t} \min\!\Bigl(\frac{j\,(n+1)}{t+1},\ W_t\Bigr), \qquad W_n = \infty$$

**Ziel Wert aus der Rangordnung.** Bei unabhängigen gleichverteilten Werten hat das Angebot mit absolutem Rang $r$ den Erwartungswert $(n+1-r)/(n+1)$, also ist der erwartete Wert eines Rangordnungs-Verfahrens

$$E[\text{Wert}] = 1 - \frac{E[R]}{n+1}$$

**Werte bekannt, Ziel Wert (Cayley).** Mit $j$ noch kommenden Angeboten wird ein Angebot $x$ genommen, wenn $x \ge v_j$, und der erwartete Wert ist $v_n$:

$$v_j = \frac{1 + v_{j-1}^2}{2}, \qquad v_0 = 0$$

**Werte bekannt, Ziel Treffer.** Mit $k$ noch kommenden Angeboten und bisherigem Maximum $m$ gilt für die Gewinnwahrscheinlichkeit $f_k(m)$ des optimalen Weitermachens

$$f_k(m) = m\, f_{k-1}(m) + \int_m^1 \max\bigl(x^{k-1},\ f_{k-1}(x)\bigr)\,dx, \qquad f_0 = 0$$

und ein Rekord $x$ wird genommen, wenn $x \ge b_k$, wobei $b_k$ die Wurzel von

$$\sum_{i=1}^{k} \frac{x^{-i} - 1}{i} = 1$$

ist (Nehmen gewinnt mit $x^k$; Weitermachen und danach jeden Rekord nehmen gewinnt mit $\sum_{i=1}^k (x^{k-i} - x^k)/i$). Die Demo rechnet $f$ auf einem Raster; der Rasterfehler steht in der README.

**Anzahl unbekannt.** Ist $N$ gleichverteilt auf $1, \dots, m$, gewinnt ein Rekord zur Zeit $k$ mit

$$E\!\left[\frac{k}{N} \,\Big|\, N \ge k\right] = \frac{k\,(H_m - H_{k-1})}{m-k+1}$$

und die Regel „ab $t_0$ den ersten Rekord nehmen“ gewinnt bei genau $N$ Angeboten mit

$$P(\text{Treffer} \mid N) = \frac{1}{N} \sum_{t=t_0}^{N} \frac{t_0 - 1}{t - 1} \qquad (t_0 \ge 2,\ N \ge t_0)$$

**Zwei Umschläge.** Mit einer Zufallsschwelle $T$ (hier exponentialverteilt mit Mittel $\sigma$), Wechsel bei $x < T$, und den Beträgen $a < b$:

$$P(\text{Gewinn}) = \frac{1}{2} + \frac{1}{2}\,P(a < T < b) = \frac{1}{2} + \frac{1}{2}\bigl(e^{-a/\sigma} - e^{-b/\sigma}\bigr)$$

**Literatur.** Die Quellen stehen in der README mit Stand der Prüfung; Formeln und Zahlen dieser Demo sind eigene Rechnung.
        """
    )

st.markdown("---")
st.subheader("📄 Ergebnisse als PDF")
if st.button("PDF erstellen", key="pdf_button"):
    from inl_pdf_export import build_pdf
    st.session_state["pdf_bytes"] = build_pdf(a, verdicts)
if st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="informationsleiter.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
