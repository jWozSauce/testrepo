"""World Cup betting model: player ratings -> lines -> edges.

Typical use:

    from worldcup_betting import load_model, price_match
    model = load_model()
    card = price_match(model, "Brazil", "Argentina")
    card["one_x_two"], card["totals"], card["props_home"], ...
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import edges, markets
from .match_model import build_match
from .ratings import (
    apply_availability,
    get_strength,
    load_players,
    squad,
    team_strengths,
)

__all__ = [
    "load_model", "price_match", "Model",
    "edges", "markets",
]


@dataclass
class Model:
    players: pd.DataFrame
    strengths: pd.DataFrame

    @property
    def teams(self) -> list[str]:
        return sorted(self.strengths["team"].tolist())


def load_model(unavailable: dict[str, list[str]] | None = None) -> Model:
    """Load players, apply any availability changes, and compute team strengths."""
    players = apply_availability(load_players(), unavailable)
    return Model(players=players, strengths=team_strengths(players))


def price_match(
    model: Model,
    home: str,
    away: str,
    home_adv: float = 1.0,
    total_lines: tuple[float, ...] = (2.5,),
    handicap_line: float = -0.5,
) -> dict:
    """Price every supported market for one fixture and return a card dict."""
    hs = get_strength(model.strengths, home)
    as_ = get_strength(model.strengths, away)
    m = build_match(hs, as_, home_adv=home_adv)

    return {
        "fixture": (home, away),
        "expected_goals": {"home": m.lambda_home, "away": m.lambda_away},
        "one_x_two": markets.one_x_two(m),
        "totals": {ln: markets.totals(m, ln) for ln in total_lines},
        "handicap": markets.asian_handicap(m, handicap_line),
        "btts": markets.btts(m),
        "correct_score": markets.correct_score(m),
        "props_home": markets.goalscorer_props(model.players, home, m.lambda_home),
        "props_away": markets.goalscorer_props(model.players, away, m.lambda_away),
        "squad_home": squad(model.players, home),
        "squad_away": squad(model.players, away),
    }
