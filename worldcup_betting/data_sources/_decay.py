"""Recency weighting helpers.

A match (or season) that happened `age` time units before the reference point
gets weight 0.5 ** (age / half_life): one half-life ago counts half as much,
two half-lives a quarter, and so on. Decay-weighted per-90 rates therefore lean
on a player's most recent form while still using the full history as a base.
"""
from __future__ import annotations


def half_life_weight(age: float, half_life: float | None) -> float:
    """Weight for something `age` units old given a half-life (same units).

    half_life=None (or <=0) disables decay and returns 1.0 (equal weighting).
    """
    if not half_life or half_life <= 0:
        return 1.0
    return 0.5 ** (max(age, 0.0) / half_life)


def geometric_weights(n: int, decay: float) -> list[float]:
    """Weights for n items in newest-first order: 1, decay, decay^2, ...

    decay=0.5 means each older season counts half as much as the next-newer one.
    """
    decay = max(0.0, min(decay, 1.0))
    return [decay ** k for k in range(n)]
