"""Szenario: Spotfracht-Angebote eines Tages. Die Werte sind unabhängig gleichverteilt auf [0, 1]; die Rate in Euro ist nur ein Maßstab dazu."""

from dataclasses import dataclass

import numpy as np

import inl_constants as C
from inl_rng import uniform_block


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    goal: str = C.DEFAULT_GOAL
    wait: int = C.DEFAULT_WAIT
    seed: int = C.DEFAULT_SEED


def skip_count(n: int, wait_percent: int) -> int:
    """Zahl der abgelehnten Angebote bei einer Wartezeit in Prozent der n Angebote: abgerundet (ganzzahlig gerechnet) und höchstens n − 1."""
    return min(n - 1, (n * wait_percent) // 100)


def make_values(seed: int, n: int, runs: int) -> np.ndarray:
    """(runs, n) unabhängig gleichverteilte Werte aus einem SplitMix64-Block; die Zeile 0 ist die gezeigte Folge. Gleiche Eingabe, gleiche Werte."""
    return uniform_block(seed, runs * n).reshape(runs, n)


def to_rate(value):
    """Wert in [0, 1] -> Frachtrate in Euro."""
    return C.RATE_LOW + np.asarray(value) * (C.RATE_HIGH - C.RATE_LOW)


def with_(s: Settings, **changes) -> Settings:
    return Settings(**{**s.__dict__, **changes})
