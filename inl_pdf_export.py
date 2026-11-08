"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Leiter und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import inl_constants as C
import inl_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "≠": "!=", "√": "Wurzel", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich",
            "→": "->", "×": "x", "≈": "~", "½": "1/2", "…": "...", "‚": ",", "‘": "'", "’": "'", "”": '"', "€": "EUR",
            "±": "+-", "π": "pi", "δ": "delta"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1 (Umlaute und ß gehören dazu); übrige Sonderzeichen werden ersetzt, Markdown-Sterne entfernt."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def _cell(value, goal: str) -> str:
    if value is None:
        return "-"
    return V.de(value, 3) if goal == "rang" else V.pct(value, 1)


def build_pdf(analysis: V.Analysis, verdicts: list[tuple[str, str]]) -> bytes:
    s = analysis.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Informationsleiter: Wie viel ist Wissen beim Auswählen wert?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"{s.n} Angebote; Ziel: {C.GOAL_LABELS[s.goal]}; eigene Wartezeit {s.wait} Prozent ({analysis.skip_own} abgelehnt); Zufalls-Seed {s.seed}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Die Leiter (exakt, optimale Regel je Stufe und Ziel)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    widths = (74, 38, 38, 38)
    for text, w in zip(("Stufe", "Treffer", "erwarteter Rang", "erwarteter Wert"), widths):
        pdf.cell(w, 6, text, border=1)
    pdf.ln()
    for row in analysis.rows:
        pdf.cell(widths[0], 6, _latin(row.name), border=1)
        for goal, w in zip(C.GOALS, widths[1:]):
            pdf.cell(w, 6, _latin(_cell(row.get(goal), goal)), border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Alle Regeln sind Modellannahmen (gleichverteilte, unabhängige Raten, zufällige Reihenfolge, keine Rückfrage), keine Aussagen über Menschen oder echte Spotmärkte. "
                                "Die Zahlen sind eigene Rechnung, durch Aufzählung, zweite Rechenwege und Simulation gestützt; die Einzelresultate sind Lehrbuchwissen."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
