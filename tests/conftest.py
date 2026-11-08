"""Gemeinsame Fixtures: Mini-Folgen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_day():
    """Fünf Angebote als Werte: 0,2 / 0,9 / 0,5 / 0,95 / 0,1 (eine Zeile). Von Hand:
    Rekorde an den Positionen 1, 2 und 4 (1-basiert); Relativränge 1, 1, 2, 1, 5; der Gesamtbeste ist Angebot 4 (Rang 1), Angebot 2 hat Rang 2."""
    return np.array([[0.2, 0.9, 0.5, 0.95, 0.1]])
