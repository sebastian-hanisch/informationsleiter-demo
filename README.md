# Informationsleiter: Wie viel ist Wissen beim Auswählen wert? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-informationsleiter-demo.streamlit.app/)**

Stück 4 der Rätsel-Reihe (Arbeitstitel „Teamrätsel“: Aufgaben, bei denen eine Strategie vorab feststeht und danach mit beschränkter Information gehandelt wird) für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, das **Sekretärinnenproblem ausgebaut**. Vehikel: **Spotfracht-Angebote**, die ein Disponent eines Tages **nacheinander** sieht und von denen er jedes sofort annehmen oder ablehnen muss; ein abgelehntes Angebot ist weg. **Dieselbe zufällige Folge** geht durch mehrere **Wissensstufen** (nur die Rangordnung, Werte bekannt, Anzahl unbekannt) und wird an einem wählbaren **Ziel** gemessen: das höchste Angebot (Treffer), kleiner erwarteter Rang oder hoher erwarteter Wert. Die unterste Sprosse der Leiter ist das Umschlag-Spiel mit Zufallsschwelle (n = 2, eine andere Aufgabenform).

Abgrenzung: `sekretaerproblem-demo` wählt genau ein Angebot aus einer Folge (klassische Stichprobenregel); diese Demo lässt dieselbe Aufgabe durch Wissensstufen und Ziele laufen. `routenresilienz-demo` entscheidet mit **bekannter** Risikodynamik per Bellman über Periode und Risikostufe (Kosten, kein Beobachten-dann-Wählen), `online-matching-demo` weist viele Aufträge unwiderruflich zu und vergleicht mit dem Offline-Optimum (viele Paare statt einer Wahl), `revenue-management-demo` liefert das Hindsight-Muster, das hier als „Hellsicht“ die obere Schranke ist. Die Einzelresultate dieser Demo sind **Lehrbuchwissen** (Gilbert und Mosteller; Chow, Moriguti, Robbins und Samuels; Cayley; Presman und Sonin; Cover, siehe Quellenstand); **Eigenes dieser Demo** ist der Vergleich der Stufen und Ziele auf derselben Folge, die Kreuzbewertung der Regeln und die Messungen unten.

## Kernfrage

*Mit bloßer Rangordnung findet die beste Regel bei 100 Angeboten das höchste mit 37,1 %, mit bekannten Werten mit 58,3 %: 21,2 Prozentpunkte mehr. Ändert man das Ziel, ändert sich die Regel: die Regel fürs höchste Angebot trifft mit 37,1 %, das gewählte Angebot hat aber im Mittel Rang 20,01; die Regel für den kleinen erwarteten Rang trifft nur mit 26,3 %, liegt aber bei Rang 3,60.* Eigenes dieser Demo ist außerdem der faire Vergleich bei unbekannter Anzahl (11,0 Prozentpunkte weniger auf denselben Tagen) und die Kreuzbewertung: die Regel für das Ziel Wert bei bekannten Werten erreicht in der Simulation (2.000 Folgen, Zufalls-Seed 7) Rang 2,35, in längeren Läufen (200.000 Folgen) rund 2,2, jeweils besser als das exakte Optimum der Rangordnung (3,60).

## Modell

Voraussetzungen, wo nichts anderes steht: **n bekannte Angebote**, Werte **unabhängig gleichverteilt** auf [0, 1] (in Euro: 800 bis 1.200, nur als Maßstab), **zufällige Reihenfolge**, **genau ein Angebot wird genommen**, **keine Rückfrage**, keine Kosten je Ansehen.

- **Nur Rangordnung:** nach Angebot k ist nur der Relativrang j (1 = höchstes unter den ersten k) bekannt; Relativränge verschiedener Zeitpunkte sind unabhängig und gleichverteilt. Ziel Treffer: erst s ablehnen, dann den ersten Rekord nehmen, Trefferwahrscheinlichkeit (s/n)·Σₖ₌ₛ₊₁ⁿ 1/(k−1). Ziel Rang: Rückwärts-DP, ein Angebot mit Relativrang j zur Zeit k hat im Mittel den absoluten Rang j(n+1)/(k+1). Ziel Wert: bei gleichverteilten Werten ist der Erwartungswert des Angebots mit Rang r gleich (n+1−r)/(n+1), also hat das Ziel Wert bei Rangordnung dieselbe Regel wie das Ziel Rang und E[Wert] = 1 − E[Rang]/(n+1).
- **Werte bekannt:** Ziel Wert (Cayley): v₀ = 0, vⱼ = (1 + vⱼ₋₁²)/2, ein Angebot x wird bei j noch kommenden genommen, wenn x ≥ vⱼ. Ziel Treffer: Raster-DP über (k noch kommend, bisheriges Maximum), ein Rekord x wird bei k noch kommenden genommen, wenn x ≥ b_k, b_k die Wurzel von Σᵢ₌₁ᵏ (x⁻ⁱ − 1)/i = 1. Das Ziel Rang bei bekannten Werten ist **nicht gerechnet**.
- **Anzahl unbekannt:** N gleichverteilt auf 1 bis m (bekannt ist nur m), nur Rangordnung, Ziel Treffer; endet der Tag ohne Wahl, gibt es keinen Treffer. Die beste Regel nimmt den ersten Rekord ab einem festen Zeitpunkt (Form für m ≤ 200 geprüft, nicht bewiesen). Der faire Vergleich zur bekannten Anzahl nimmt dieselben Tage (N gleichverteilt) und lässt die Regel N kennen: Mittel der besten Regel je N.
- **Umschlag-Spiel:** zwei Umschläge mit festen Beträgen a < b, einer wird zufällig geöffnet, bei gesehenem Betrag x wird gewechselt, wenn x < T für eine exponentialverteilte Zufallsschwelle T: Gewinn 1/2 + 1/2·(e^(−a/σ) − e^(−b/σ)) für jedes Paar, kein Wissen über die Beträge nötig.
- **Hellsicht** (im Nachhinein das höchste Angebot) ist die obere Schranke, keine Regel für den Betrieb; **Zufall** ist ein gleichverteilt gezogenes Angebot.
- Alle Zufallsfolgen kommen aus SplitMix64 (feste Zufalls-Seeds); die gezeigte Folge ist die erste von 2.000 Folgen der Simulation.

Formeln stehen im Expander „📐 Mathematische Formulierung“ der App. Alle Herleitungen sind Lehrbuchwissen, hier mit eigener Rechnung, Aufzählung und Simulation geprüft, nicht gegen die Originalarbeiten gelesen.

## Befunde und Korrekturen gegenüber dem Plan

Alle Zahlen sind in dieser Demo neu gerechnet (`tests/test_claims.py`), jede (außer der Cayley-Simulation zum Rang in H8 und den Literatur-Grenzwerten) mit einem zweiten Rechenweg gegengeprüft (`tests/test_oracle_informationsleiter.py`, siehe Verifikation). Exakte Zahlen sind Rückwärts-DP, Formeln oder Raster-DP; Simulationen haben feste Zufalls-Seeds und eine Marge von vier Standardfehlern.

| Nr. | Frage | Befund | Beleg |
|---|---|---|---|
| H1 | Rangordnung, Ziel Treffer: beste Wartezeit und Trefferwahrscheinlichkeit | n = 10: 39,9 % bei 3 abgelehnten; n = 100: **37,1 % bei 37 abgelehnten**; n = 200: 36,9 % bei 73; n = 1000: 36,8 % (Grenzwert 1/e = 36,8 %) | `test_h1_*`; Formel gegen Aufzählung aller Reihenfolgen bis n = 7 und gegen das Optimum über alle Stopp-Mengen |
| H2 | Rangordnung, Ziel Rang: erwarteter Rang | n = 100: **3,60** (Zufall 50,5); n = 1000: 3,83; der Grenzwert für sehr viele Angebote ist laut Suchtreffern 3,8695 (Quelle nicht im Volltext gelesen). Wert statt Rang: 96,4 % des Wertebereichs | `test_h2_*`; DP gegen Entscheidungsbaum über alle Reihenfolgen bis n = 7 |
| H3 | Ändert das Ziel die Regel? (n = 100, nur Rangordnung) | **Ja.** Regel fürs höchste Angebot: Treffer 37,1 %, Rang **20,01**, Wert 80,2 %. Regel für den Rang: Treffer 26,3 %, Rang **3,60**, Wert 96,4 %. Die Rang-Regel nimmt erst ab Angebot 28 überhaupt eines; zum Ende nimmt sie auch zweit- und drittbeste bisherige | `test_h3_*` (exakte Bewertung beider Regeln, gegen Aufzählung bis n = 6) |
| H4 | Werte bekannt, Ziel Treffer | n = 2: 75,0 %; n = 3: 68,4 %; n = 10: 60,9 %; n = 100: **58,3 %**; n = 200: 58,2 %; n = 1000: 58,0 % (Raster 64 000 Punkte; Grenzwert laut Suchtreffern 0,5801). Bei n = 100 Vorsprung vor der Rangordnung **21,2 Prozentpunkte** | `test_h4_*`; Raster-DP gegen exakte Fälle n = 2, 3 und gegen die Simulation der Schwellenregel |
| H5 | Ziel Wert (Cayley) | n = 10: 86,1 %; n = 100: **98,1 %** gegen 96,4 % bei bloßer Rangordnung und 99,0 % bei Hellsicht; die bekannten Werte schließen 66 % dieser Lücke. Der Vorsprung vor der Rangordnung liegt in den 8 gerechneten Angebotszahlen (2 bis 200) zwischen 0,9 Prozentpunkten (n = 200) und 12,5 (n = 2) | `test_h5_*`; Rekursion gegen Simulation |
| H6 | Anzahl unbekannt (N gleichverteilt 1 bis m) | m = 10: 35,1 %; m = 100: **27,8 %**, Regel ab Angebot 14; m = 1000: 27,1 % ab Angebot 136 (Grenzwert 2/e² = 27,1 %). Auf denselben Tagen mit bekannter Anzahl: m = 100 **38,8 %**, also 11,0 Prozentpunkte weniger; hat der Tag genau 100 Angebote, trifft dieselbe Regel mit 27,0 % | `test_h6_*`; DP gegen Aufzählung aller Stopp-Mengen und Reihenfolgen bis m = 6 und gegen Simulation |
| H7 | Umschlag-Spiel mit exponentialverteilter Schwelle (Mittel 10) | (1; 2): 54,31 %; (5; 15): 69,17 %; (100; 300): 50,00 %. In **72 von 72** Zellen (Beträge × Verhältnisse × Schwellenskalen) über 50 %, größter Wert 84,2 %; in 21 Zellen weniger als 0,1 Prozentpunkte darüber | `test_h7_*`; Formel gegen Simulation |
| H8 | Hilft das Wissen um die Werte auch beim Ziel Rang? (n = 100, Simulation 2.000 Folgen, Zufalls-Seed 7) | Die Cayley-Regel erreicht Rang **2,35** ± 0,07 (eine Stichprobe; in längeren Läufen mit 200.000 Folgen rund 2,2), besser als das exakte Optimum der Rangordnung (3,60). Das beste Verfahren für den Rang bei bekannten Werten ist damit **nicht** bestimmt | `test_h8_*` |
| H9 | Wartezeit-Regel für die Ziele Rang und Wert (n = 100) | Beste Wartezeit der Form „ablehnen, dann Rekord“: für den Rang **9** abgelehnte mit Rang 9,59, für den Wert 9 mit 90,5 %; die DP-Regel erreicht 3,60 bzw. 96,4 % | `test_h9_*` |
| H10 | Kleinster Fall n = 2 | Rangordnung 50,0 % (erwarteter Rang 1,50), Werte bekannt 75,0 %, Hellsicht 100 % | `test_h10_*` |

**Korrekturen gegenüber dem Plan:** (1) Der Plan nannte für volle Information bei n = 1000 den Wert 0,5824; das war ein **Rasterfehler** des ersten Entwurfs (4 000 Rasterpunkte, +0,002); mit 64 000 Punkten sind es 0,5804. Auch n = 100: 0,5829 statt 0,5830. (2) Für das Umschlag-Paar (5; 15) nannte der Plan 0,6944 aus einer Simulation; die exakte Formel gibt 0,6917, eine neue Simulation mit 100.000 Läufen bleibt innerhalb von vier Standardfehlern. (3) Beim unbekannten n waren nur 0,2778 und 0,2714 genannt; neu sind der faire Vergleich auf denselben Tagen und die Form der Regel.

## Quellenstand

Die folgenden Resultate sind Lehrbuchwissen. Die Angaben unten stammen **aus Suchtreffern (Titel und Zusammenfassungen), die Originalarbeiten wurden nicht im Volltext gelesen**: Gilbert und Mosteller, „Recognizing the Maximum of a Sequence“ (Journal of the American Statistical Association 61, 1966): Grenzwert 0,5801 bei bekannter Verteilung. Chow, Moriguti, Robbins und Samuels, „Optimal selection based on relative rank“ (Israel Journal of Mathematics 2, 1964): Grenzwert 3,8695 beim erwarteten Rang. Presman und Sonin, „The best choice problem for a random number of objects“ (1972): für die Gleichverteilung Einsatzzeitpunkt e⁻² und Trefferwahrscheinlichkeit 2e⁻² (nach einer späteren Arbeit, die dies mit dem Ergebnis von Presman und Sonin gleichsetzt). Cover, „Pick the largest number“ (1987): Zufallsschwelle; die Formel 1/2 + 1/2·P(Schwelle zwischen den Beträgen) nach Suchtreffern, die sie Cover zuschreiben. Das Cayley-Problem (Wert statt Treffer) ist nicht per Suche geprüft; die Rekursion ist hier gegen Simulation gesichert. Das Robbins-Problem (Rang bei bekannten Werten) wird nur wegen seines Namens in den Suchtreffern erwähnt.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist |
|---|---|
| Raten unabhängig gleichverteilt, Verteilung bekannt | Gilt für „Werte bekannt“ und das Ziel Wert. Eine falsch angenommene Verteilung ist **nicht gemessen**. Die Stufe „Nur Rangordnung“ braucht nur unabhängige Werte aus derselben Verteilung. |
| Zufällige Reihenfolge | Bei Trend im Tagesverlauf gelten die Zahlen dieser Demo nicht mehr; wie stark sie abweichen, ist nicht gemessen. |
| Keine Rückfrage, ein Angebot, keine Kosten je Ansehen | Rückruf, mehrere Plätze und Kosten je Ansehen sind **nicht gerechnet**. |
| Anzahl bekannt (außer der Stufe „Anzahl unbekannt“) | Die Gleichverteilung der Anzahl ist eine Modellannahme; die Stufe ist nicht Schritt für Schritt mit n bekannten Angeboten vergleichbar, weil ihre Tage im Mittel kürzer sind (der faire Vergleich steht in E5). |
| Nicht jede Kombination von Stufe und Ziel | Anzahl unbekannt nur für das Ziel Treffer; Werte bekannt nicht für das Ziel Rang (offenes Problem, hier nur die Simulation einer Regel). |
| Beste Regel für unbekannte Anzahl | Die Form „ab einem Zeitpunkt den ersten Rekord“ ist für m ≤ 200 geprüft, nicht bewiesen. |
| Raster-DP bei bekannten Werten | Rasterfehler bei n = 200: rund 4 · 10⁻⁶ (Raster 16 000 gegen 64 000 Punkte); bei n = 1000 und 4 000 Punkten +0,002 (Korrektur 1 oben). |
| Modell | Keine Aussage über Menschen, Disponenten oder echte Spotmärkte. |

## Bewusst nicht umgesetzt

Als **Ausblick, nicht gemessen:** Kosten pro Ansehen, Rückruf, mehrere Plätze, eine falsch angenommene Verteilung der Raten, andere Verteilungen als die Gleichverteilung, der erwartete Rang bei bekannten Werten (offen) und andere Schwellenverteilungen im Umschlag-Spiel.

## Verifikation

- **Kerneinheiten** (`tests/test_rank.py`, `test_values.py`, `test_unknown.py`, `test_envelopes.py`, `test_simulate.py`): jede Einheit an Handrechnungen und exakten kleinen Fällen (Mini-Folge in `tests/conftest.py`).
- **Orakel** (`tests/test_oracle_informationsleiter.py`): Rang-Stufe gegen Aufzählung aller Reihenfolgen und das Optimum über alle Stopp-Mengen (n ≤ 7), Rang-DP gegen einen Entscheidungsbaum über alle Reihenfolgen (n ≤ 7) und die exakte Bewertung gegen Aufzählung für Zufalls-Schwellenlisten; Werte bekannt: Raster-DP gegen Simulation der Wurzel-Schwellen und gegen exakte Fälle; Cayley gegen Simulation; unbekannte Anzahl gegen Aufzählung (m ≤ 6) und Simulation; jeweils Nachbar-Schwellen verschlechtern.
- **Aussagen** (`tests/test_claims.py`): jede Zahl dieser README wird nachgerechnet und im Text gesucht.
- **Verdikte, Eigenschaften, Randfälle:** reine Verdikt-Funktionen an handgesetzten Zeilen; Reihenfolge der Stufen über n; n = 1, 2, 200.
- **Oberfläche:** AppTest (jeder Button, jeder Regler am Minimum und Maximum, alle Presets, Permalink, Eckfall-Matrix, jeder ▶️-Knopf, PDF), Layout (Schnellstart, fester Kopf, Hervorhebung), Footer, Laufzeit-Importe, Mathe-Blöcke.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Hauptansicht, fünf Experimente, Annahmen, Formeln, PDF) |
| `inl_constants.py` | Regler-Grenzen, Rechenbudget, Presets, Farben |
| `inl_rank.py` | Stufe „nur Rangordnung“: Schwellenlisten, exakte Bewertung, Rang-DP, Wartezeit-Kurve |
| `inl_values.py` | Stufe „Werte bekannt“: Cayley-Rekursion, Schwellen, Raster-DP |
| `inl_unknown.py` | Stufe „Anzahl unbekannt“: DP, Regel, Trefferwahrscheinlichkeit bei festem N |
| `inl_envelopes.py` | Umschlag-Spiel: Formel und Simulation |
| `inl_simulate.py` | Regeln auf konkreten Folgen (vektorisiert über die Folgen) |
| `inl_scenario.py` | Einstellungen, Folgen aus SplitMix64, Rate in Euro |
| `inl_evaluation.py` | Leiter, Analyse, Kreuzbewertung, Messreihen, Verdikt-Funktionen |
| `inl_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `inl_pdf_export.py` | PDF-Export (erst nach Klick) |
| `inl_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `inl_layout.py` | Fester Kopf (Stylesheet) |
| `inl_rng.py` | SplitMix64, wörtliche Kopie aus `ab-test-demo` |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit-Bibliotheken (ungepinnt), dazu pytest |
| `.github/workflows/tests.yml`, `.streamlit/config.toml`, `.gitignore` | CI (Python 3.12, wöchentlicher Canary), Streamlit-Einstellungen, Ausschlüsse |
| `tests/` | siehe Verifikation |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Python-Version der Streamlit Cloud ist in der Cloud-Einstellung gewählt; lokal 3.14 und in der CI 3.12 laufen alle Tests. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

189 Tests (unter Windows, Python 3.14 im Test-venv; rund 13 Sekunden mit pytest-xdist, `-n 6`).

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Teamrätsel: Strategie ohne Verständigung](https://sebastianhanisch.net/konzepte-teamraetsel.html).
