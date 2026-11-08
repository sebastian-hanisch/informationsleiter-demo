"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Knopf, fährt jeden Regler an seine Grenzen und prüft, dass
kein Python-Fehler auftritt - insbesondere `StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=). Die ▶️-Knöpfe laufen einzeln (die Schleife lässt sie aus)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import inl_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer
AUTOPLAY_MARKER = "▶"  # ▶: Knöpfe mit Rechenlast, einzeln getestet
PRESET_LABELS = set(C.PRESETS)


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([w.value for w in at.warning] + [s.value for s in at.success] + [i.value for i in at.info] + [e.value for e in at.error])


def _tables(at, column):
    return [df for df in at.dataframe if column in df.value.columns.tolist()]


def test_app_loads_without_exception_and_shows_the_headline_numbers():
    at = _fresh_app()
    assert [m.value for m in at.metric][:4] == ["1,0 %", "27,8 %", "37,1 %", "58,3 %"]       # Zufall, Anzahl unbekannt, Rangordnung, Werte bekannt bei n = 100
    assert "21,2 Prozentpunkte mehr" in _texts(at)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label or label in PRESET_LABELS:
            continue                                                 # Presets und Rechenknöpfe laufen einzeln
        at = _fresh_app()
        at.button[index].click().run()
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run()
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("goal", C.GOALS)
@pytest.mark.parametrize("edge", ["min", "max"])
def test_corner_matrix_all_regulators_at_their_minima_or_maxima(goal, edge):
    """Eckfall-Matrix (DEMO-PLAYBOOK §1d): je Ziel alle Regler zugleich am Minimum (n = 2, Wartezeit 0) bzw. Maximum (n = 200, Wartezeit 100)."""
    at = _fresh_app()
    at.selectbox(key="goal_select").set_value(goal).run()
    at.slider(key="n_slider").set_value(C.N_MIN if edge == "min" else C.N_MAX)
    at.slider(key="wait_slider").set_value(C.WAIT_MIN if edge == "min" else C.WAIT_MAX)
    at.run()
    assert not at.exception, (goal, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_loads_and_shows_a_verdict(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert _texts(at)
    p = C.PRESETS[name]
    assert at.session_state["n_slider"] == p["n"] and at.session_state["goal_select"] == p["goal"] and at.session_state["wait_slider"] == p["wait"]


def test_goal_rank_shows_no_value_stage_and_says_so():
    at = _fresh_app()
    at.selectbox(key="goal_select").set_value("rang").run()
    assert not at.exception
    assert [m.value for m in at.metric][:4] == ["50,50", "–", "3,60", "–"]                    # Zufall, Anzahl unbekannt, Rangordnung, Werte bekannt: nicht gerechnet
    assert "nicht gerechnet" in _texts(at)


def test_goal_value_shows_the_value_stage():
    at = _fresh_app()
    at.selectbox(key="goal_select").set_value("wert").run()
    assert [m.value for m in at.metric][2:4] == ["96,4 %", "98,1 %"]


def test_two_offers_show_the_smallest_case():
    at = _fresh_app()
    at.sidebar.button(key="preset_Nur zwei Angebote").click().run()
    assert [m.value for m in at.metric][2:4] == ["50,0 %", "75,0 %"]


def test_the_play_buttons_work_individually():
    at = _fresh_app()
    at.button(key="e1_button").click().run()
    assert not at.exception and len(_tables(at, "Abstand in Standardfehlern")) == 1

    at = _fresh_app()
    at.button(key="e2_button").click().run()
    assert not at.exception and len(_tables(at, "Abstand in Standardfehlern")) == 1
    assert _tables(at, "Abstand in Standardfehlern")[0].value["Stufe"].tolist() == ["Nur Rangordnung", "Werte bekannt", "Anzahl unbekannt (nur Rangordnung)"]

    at = _fresh_app()
    at.button(key="e3_button").click().run()
    assert not at.exception and len(_tables(at, "Treffer (Sim.)")) == 1
    assert any("Simulation" in s.value and "Rang **2,35**" in s.value for s in at.success)       # Verdikt zur Cayley-Regel beim Ziel Rang

    at = _fresh_app()
    at.button(key="e5_button").click().run()
    assert not at.exception and len(_tables(at, "Abstand in Standardfehlern")) == 1


def test_play_results_are_hidden_when_the_sidebar_settings_change():
    at = _fresh_app()
    at.button(key="e2_button").click().run()
    assert len(_tables(at, "Abstand in Standardfehlern")) == 1
    at.slider(key="n_slider").set_value(50).run()
    assert not at.exception and _tables(at, "Abstand in Standardfehlern") == []


def test_experiment_tabs_show_their_results_without_a_button():
    at = _fresh_app()
    texts = _texts(at)
    assert "In 72 von 72 Zellen" in texts                                                    # E1
    assert "Der Vorsprung der bekannten Werte vor der bloßen Rangordnung" in texts           # E2
    assert "Die Regel fürs höchste Angebot (erst 37 ablehnen) trifft mit 37,1 %" in texts    # E3
    assert "Beim ersten von 100 Angeboten" in texts                                          # E4
    assert "11,0 Prozentpunkte weniger durch das Nichtwissen" in texts                       # E5


def test_selectboxes_of_the_umschlag_experiment_work_for_every_value():
    for key, values in (("e1_amount", C.ENV_AMOUNTS), ("e1_ratio", C.ENV_RATIOS), ("e1_scale", C.ENV_SCALES)):
        for v in values:
            at = _fresh_app()
            at.selectbox(key=key).set_value(v).run()
            assert not at.exception, (key, v, [str(e) for e in at.exception])


def test_pdf_button_creates_the_download():
    at = _fresh_app()
    at.button(key="pdf_button").click().run()
    assert not at.exception and at.session_state["pdf_bytes"][:4] == b"%PDF"
    for goal in ("rang", "wert"):
        at = _fresh_app()
        at.selectbox(key="goal_select").set_value(goal).run()
        at.button(key="pdf_button").click().run()
        assert not at.exception and at.session_state["pdf_bytes"][:4] == b"%PDF"


def test_math_expander_has_one_display_block_per_equation():
    at = _fresh_app()
    exp = next(e for e in at.expander if e.label == "📐 Mathematische Formulierung")
    text = "\n".join(m.value for m in exp.markdown)
    assert text.count("$$") % 2 == 0 and text.count("$$") // 2 == 10
    chunks = text.split("$$")[1::2]
    assert all("\n" not in c.strip() and "\\\\" not in c for c in chunks)        # je Gleichung ein einzeiliger Block, kein Zeilenumbruch \\ darin (KaTeX-Fehler im Browser)


def test_permalink_sets_the_regulators_and_clamps_to_the_bounds():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "999"                          # über der Grenze: wird auf 200 begrenzt
    at.query_params["ziel"] = "wert"
    at.query_params["wartezeit"] = "-5"                   # unter der Grenze: wird auf 0 begrenzt
    at.query_params["seed"] = "123"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["goal_select"] == "wert" and at.session_state["wait_slider"] == 0 and at.session_state["seed_input"] == 123


def test_permalink_ignores_values_outside_the_allowed_choices_and_garbage():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["ziel"] = "Quatsch"
    at.query_params["n"] = "viele"
    at.run()
    assert not at.exception
    assert at.session_state["goal_select"] == C.DEFAULT_GOAL and at.session_state["n_slider"] == C.DEFAULT_N


def test_the_address_bar_mirrors_the_configuration():
    at = _fresh_app()
    at.slider(key="n_slider").set_value(40).run()
    assert at.query_params["n"] == "40" and at.query_params["ziel"] == C.DEFAULT_GOAL


def test_the_shown_day_is_the_same_for_the_same_seed():
    a1, a2 = _fresh_app(), _fresh_app()
    assert [d.value.to_dict("list") for d in a1.dataframe][0] == [d.value.to_dict("list") for d in a2.dataframe][0]
