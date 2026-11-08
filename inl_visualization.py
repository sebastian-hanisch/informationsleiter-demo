"""Plotly-Figuren: alle Achsen fest (`fixedrange`), damit Touch-Scrollen auf der Seite nicht am Chart hängen bleibt."""

import numpy as np
import plotly.graph_objects as go

import inl_constants as C
import inl_envelopes as EN
import inl_scenario as S

PICK_STYLE = {
    "rang": ("Nur Rangordnung", C.COLORS["rang"], "circle"),
    "werte": ("Werte bekannt", C.COLORS["werte"], "square"),
    "unbekannt": ("Anzahl unbekannt", C.COLORS["unbekannt"], "triangle-up"),
    "eigene": ("Eigene Wartezeit", C.COLORS["eigene"], "diamond"),
    "hellsicht": ("Höchstes Angebot (Hellsicht)", C.COLORS["hellsicht"], "star"),
}
PICK_OFFSET = {"hellsicht": 14, "rang": 14, "werte": 32, "unbekannt": 50, "eigene": 68}   # Markierungen übereinander, wenn mehrere Stufen dasselbe Angebot wählen
GOAL_AXIS = {"treffer": "Trefferwahrscheinlichkeit", "rang": "erwarteter Rang (kleiner ist besser)", "wert": "erwarteter Wert (Anteil des Wertebereichs)"}


def _layout(fig, title=None, xtitle=None, ytitle=None, height=380, legend_bottom=True):
    fig.update_layout(
        title=title, height=height, margin=dict(l=55, r=15, t=45 if title else 15, b=130 if legend_bottom else 50),
        legend=dict(orientation="h", yanchor="top", y=-0.3, x=0) if legend_bottom else None,
        xaxis=dict(title=xtitle, fixedrange=True), yaxis=dict(title=ytitle, fixedrange=True),
    )
    return fig


def build_sequence(values: np.ndarray, picks: dict, wait_boundary: int | None = None) -> go.Figure:
    """Die gezeigte Folge als Säulen (Rate in Euro), darüber die Wahl jeder Stufe. `wait_boundary`: Zahl der abgelehnten Angebote der eigenen Wartezeit (gestrichelte Linie)."""
    n = len(values)
    rates = S.to_rate(values)
    x = np.arange(1, n + 1)
    fig = go.Figure()
    colors = [C.COLOR_OTHER] * n
    fig.add_trace(go.Bar(x=x, y=rates - C.RATE_LOW, base=C.RATE_LOW, marker=dict(color=colors), name="Angebote", hovertemplate="Angebot %{x}: %{customdata:.0f} €<extra></extra>",
                         customdata=rates, showlegend=False))
    for key, idx in picks.items():
        if idx is None:
            continue
        name, color, symbol = PICK_STYLE[key]
        fig.add_trace(go.Scatter(x=[idx + 1], y=[rates[idx] + PICK_OFFSET[key]], mode="markers", name=f"{name}: Angebot {idx + 1}", marker=dict(color=color, symbol=symbol, size=13, line=dict(width=1, color="white")),
                                 hovertemplate=f"{name}<br>Angebot {idx + 1}: {rates[idx]:.0f} €<extra></extra>"))
    if wait_boundary:
        fig.add_vline(x=wait_boundary + 0.5, line=dict(color=C.COLORS["eigene"], dash="dash", width=1))
    _layout(fig, xtitle="Angebot (Reihenfolge des Eintreffens)", ytitle="Frachtrate in €")
    fig.update_xaxes(range=[0, n + 1])
    fig.update_yaxes(range=[C.RATE_LOW - 5, C.RATE_HIGH + 90])
    return fig


def build_skip_curve(curve: dict, goal: str, skip_own: int, skip_opt: int, refs: dict[str, float]) -> go.Figure:
    """Das Ziel als Funktion der Wartezeit (Regel „erst ablehnen, dann Rekord“), exakt. `refs`: Bezugslinien {Beschriftung: Wert}."""
    y = curve[goal]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["skip"], y=y, mode="lines", name="Regel „erst ablehnen, dann Rekord“", line=dict(color=C.COLORS["rang"], width=2)))
    fig.add_trace(go.Scatter(x=[skip_opt], y=[y[skip_opt]], mode="markers", name=f"Beste Wartezeit: {skip_opt}", marker=dict(color=C.COLORS["hellsicht"], size=12, symbol="star")))
    fig.add_trace(go.Scatter(x=[skip_own], y=[y[skip_own]], mode="markers", name=f"Ihre Wartezeit: {skip_own}", marker=dict(color=C.COLORS["eigene"], size=11, symbol="diamond")))
    palette = [C.COLORS["werte"], C.COLORS["hellsicht"], C.COLORS["zufall"]]
    for (label, value), color in zip(refs.items(), palette):
        fig.add_hline(y=value, line=dict(color=color, dash="dot", width=1), annotation_text=label, annotation_position="top right", annotation_font_size=11)
    _layout(fig, xtitle="Zahl der abgelehnten Angebote (Wartezeit)", ytitle=GOAL_AXIS[goal])
    n = len(y)
    fig.update_xaxes(range=[-1, n])
    allv = list(y) + list(refs.values())
    lo, hi = min(allv), max(allv)
    pad = 0.06 * (hi - lo if hi > lo else 1.0)
    fig.update_yaxes(range=[(0 if goal == "treffer" else lo - pad), (1.0 if goal == "treffer" else hi + pad)])
    return fig


def build_ladder_bars(rows, goal: str) -> go.Figure:
    """Die Stufen der Leiter für ein Ziel als waagerechte Säulen; Stufen ohne gerechneten Wert fehlen."""
    keys = {"zufall": "zufall", "unbekannt": "unbekannt", "rang": "rang", "werte": "werte", "hellsicht": "hellsicht"}
    have = [r for r in rows if r.get(goal) is not None]
    fig = go.Figure(go.Bar(
        y=[r.name for r in have][::-1], x=[r.get(goal) for r in have][::-1], orientation="h",
        marker=dict(color=[C.COLORS[keys[r.key]] for r in have][::-1]),
        text=[(f"{r.get(goal):.2f}" if goal == "rang" else f"{100 * r.get(goal):.1f} %").replace(".", ",") for r in have][::-1], textposition="auto",
    ))
    _layout(fig, xtitle=GOAL_AXIS[goal], height=300, legend_bottom=False)
    fig.update_layout(margin=dict(l=230, r=15, t=15, b=50))
    top = max(r.get(goal) for r in have)
    fig.update_xaxes(range=[0, (1.0 if goal != "rang" else top * 1.1)])
    fig.update_yaxes(fixedrange=True)
    return fig


def build_sweep(sweep: dict, goal: str) -> go.Figure:
    """Die Stufen über die Zahl der Angebote n (logarithmische Achse) für ein Ziel; fehlende Werte lassen die Linie weg."""
    ns = list(sweep)
    fig = go.Figure()
    names = {"zufall": "Zufall", "unbekannt": "Anzahl unbekannt", "rang": "Nur Rangordnung", "werte": "Werte bekannt", "hellsicht": "Hellsicht"}
    for key in names:
        ys = [sweep[n][key] for n in ns]
        if all(y is None for y in ys):
            continue
        fig.add_trace(go.Scatter(x=ns, y=ys, mode="lines+markers", name=names[key], line=dict(color=C.COLORS[key], width=2), connectgaps=False))
    _layout(fig, xtitle="Zahl der Angebote n", ytitle=GOAL_AXIS[goal])
    fig.update_xaxes(type="log", tickvals=ns, ticktext=[str(n) for n in ns])
    if goal == "rang":
        fig.update_yaxes(type="log")
    elif goal == "treffer":
        fig.update_yaxes(range=[0, 1.02])
    else:
        fig.update_yaxes(range=[0.45, 1.01])
    return fig


def build_envelopes(ratio: float, scale_values=C.ENV_SCALES) -> go.Figure:
    """Gewinnwahrscheinlichkeit beim Umschlag-Spiel über dem kleineren Betrag a (logarithmisch) für das Verhältnis b = ratio · a, je Skala der Zufallsschwelle."""
    a = np.logspace(-1, 3, 120)
    fig = go.Figure()
    palette = [C.COLORS["rang"], C.COLORS["werte"], C.COLORS["hellsicht"]]
    for sc, color in zip(scale_values, palette):
        p = [EN.win_probability(float(x), float(x) * ratio, sc) for x in a]
        fig.add_trace(go.Scatter(x=a, y=p, mode="lines", name=f"Schwelle exponentialverteilt, Mittel {sc}", line=dict(color=color, width=2)))
    fig.add_hline(y=0.5, line=dict(color=C.COLORS["zufall"], dash="dot"), annotation_text="Münzwurf: 50 %", annotation_position="bottom right")
    _layout(fig, xtitle="Kleinerer Betrag a (logarithmisch)", ytitle="Gewinnwahrscheinlichkeit")
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[0.49, 0.8])
    return fig


def build_rank_thresholds(s: np.ndarray, skip_best: int) -> go.Figure:
    """Rangordnung, Ziel Rang: bis zu welchem Relativrang wird zum Zeitpunkt k genommen? Die Regel fürs höchste Angebot: nie vor skip_best + 1, dann nur Relativrang 1."""
    n = len(s) - 1
    k = np.arange(1, n + 1)
    fig = go.Figure()
    best = np.where(k > skip_best, 1, 0)
    fig.add_trace(go.Scatter(x=k, y=s[1:], mode="lines", line=dict(color=C.COLORS["rang"], width=2, shape="hv"), name="Regel für Rang: größter genommener Relativrang"))
    fig.add_trace(go.Scatter(x=k, y=best, mode="lines", line=dict(color=C.COLORS["werte"], width=2, dash="dash", shape="hv"), name=f"Regel für Treffer (erst {skip_best} ablehnen)"))
    _layout(fig, xtitle="Zeitpunkt k (Angebot)", ytitle="genommen bei Relativrang bis")
    fig.update_xaxes(range=[0, n + 1])
    cap = max(6, int(np.max(s[1:n])) + 2) if n > 1 else 3
    fig.update_yaxes(range=[-0.3, cap], dtick=1 if cap <= 15 else None)
    return fig


def build_value_thresholds(b: np.ndarray, v: np.ndarray) -> go.Figure:
    """Werte bekannt: ab welcher Rate wird beim t-ten von n Angeboten genommen? (b: nur Rekorde, Ziel Treffer; v: jedes Angebot, Ziel Wert.)"""
    n = len(b)
    t = np.arange(1, n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S.to_rate(v[n - t]), mode="lines", line=dict(color=C.COLORS["hellsicht"], width=2), name="Ziel Wert: jedes Angebot ab dieser Rate"))
    fig.add_trace(go.Scatter(x=t, y=S.to_rate(b[n - t]), mode="lines", line=dict(color=C.COLORS["werte"], width=2, dash="dash"), name="Ziel Treffer: Rekord ab dieser Rate"))
    _layout(fig, xtitle="Angebot t", ytitle="Schwellenrate in €")
    fig.update_xaxes(range=[0, n + 1])
    fig.update_yaxes(range=[C.RATE_LOW - 10, C.RATE_HIGH + 10])
    return fig


def build_unknown_curve(win_unknown: np.ndarray, win_known: np.ndarray, t_from: int) -> go.Figure:
    """Trefferwahrscheinlichkeit bei Tagen der Länge N = 1..m: die Regel für unbekannte Anzahl gegen die beste Regel, die N kennt."""
    N = np.arange(1, len(win_unknown) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=N, y=win_known, mode="lines", name="Regel, die N kennt", line=dict(color=C.COLORS["rang"], width=2)))
    fig.add_trace(go.Scatter(x=N, y=win_unknown, mode="lines", name=f"Regel für unbekanntes N (ab Angebot {t_from})", line=dict(color=C.COLORS["unbekannt"], width=2)))
    _layout(fig, xtitle="tatsächliche Zahl N der Angebote des Tages", ytitle="Trefferwahrscheinlichkeit")
    fig.update_xaxes(range=[0, len(N) + 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig
