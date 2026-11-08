"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Verdikt-Zeile je Preset."""

import inl_constants as C
import inl_evaluation as V
import inl_presets as P
from inl_scenario import Settings


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regulator_bounds():
    for name, preset in C.PRESETS.items():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.WAIT_MIN <= preset["wait"] <= C.WAIT_MAX and 0 <= preset["seed"] <= C.SEED_MAX, name
        assert preset["goal"] in C.GOALS, name


def test_default_preset_equals_the_default_settings():
    assert Settings(**C.PRESETS["Standardtag (100 Angebote)"]) == Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_differ_from_each_other():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)


def test_every_preset_gives_the_expected_verdict_line():
    """DEMO-PLAYBOOK §11a (7): jedes Preset läuft durch, und die Verdikt-Zeile sagt, was zum Ziel des Presets gehört."""
    expected = {
        "treffer": "das höchste Angebot mit",
        "rang": "erwartete Rang",
        "wert": "des möglichen Wertebereichs",
    }
    for name, preset in C.PRESETS.items():
        a = V.analyse(Settings(**preset))
        kind, text = V.verdict_ladder(preset["goal"], a.n, a.rows, a.best_rule[1])
        assert expected[preset["goal"]] in text, (name, text)
        assert kind in ("success", "info")
