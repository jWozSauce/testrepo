"""Compare model probabilities to market odds and surface +EV bets.

Workflow:
  1. Convert each book price to an implied probability (1 / decimal odds).
  2. Optionally de-vig a complete market so the implied probs sum to 1 and you
     can see the book's "true" estimate without its margin.
  3. Compute expected value per unit staked and a Kelly stake fraction.
  4. Flag selections whose model edge clears a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def implied_prob(decimal_odds: float) -> float:
    return 0.0 if decimal_odds <= 0 else 1.0 / decimal_odds


def devig(decimal_odds: list[float]) -> list[float]:
    """Normalise a market's implied probabilities to remove the bookmaker vig."""
    raw = [implied_prob(o) for o in decimal_odds]
    s = sum(raw)
    return [r / s for r in raw] if s > 0 else raw


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """EV per unit staked: positive means the bet beats its price."""
    return model_prob * decimal_odds - 1.0


def kelly_fraction(model_prob: float, decimal_odds: float, cap: float = 1.0) -> float:
    """Full-Kelly stake as a fraction of bankroll (clipped at 0 and `cap`)."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (model_prob * decimal_odds - 1.0) / b
    return float(min(max(f, 0.0), cap))


@dataclass
class Edge:
    market: str
    selection: str
    model_prob: float
    market_odds: float
    market_implied: float
    ev: float
    kelly: float


def find_edges(
    rows: list[dict],
    min_ev: float = 0.02,
    kelly_multiplier: float = 0.25,
) -> pd.DataFrame:
    """Score a list of {market, selection, model_prob, market_odds} dicts.

    `kelly_multiplier` scales full Kelly (0.25 = quarter-Kelly, a common,
    less-volatile staking choice). Only selections with EV >= `min_ev` are
    returned, sorted by EV.
    """
    out = []
    for r in rows:
        odds = float(r["market_odds"])
        mp = float(r["model_prob"])
        ev = expected_value(mp, odds)
        out.append(
            Edge(
                market=r.get("market", ""),
                selection=r.get("selection", ""),
                model_prob=mp,
                market_odds=odds,
                market_implied=implied_prob(odds),
                ev=ev,
                kelly=kelly_fraction(mp, odds) * kelly_multiplier,
            )
        )
    df = pd.DataFrame([e.__dict__ for e in out])
    if df.empty:
        return df
    df = df[df["ev"] >= min_ev].sort_values("ev", ascending=False).reset_index(drop=True)
    return df
