"""Price betting markets from a scoreline matrix and from player ratings.

Match-level markets read directly off the score matrix (1X2, totals, Asian
handicap, BTTS, correct score). Player props use the per-player xG share and
expected minutes, so a player's market price reflects both how dangerous he is
and how much of the match he is expected to play.

Every probability is returned alongside its fair decimal odds (1 / p), the
price you'd need to beat to have an edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .match_model import MatchModel


def fair_odds(p: float) -> float:
    return float("inf") if p <= 0 else round(1.0 / p, 3)


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(n)[:, None]  # home goals
    j = np.arange(n)[None, :]  # away goals
    return np.broadcast_to(i, (n, n)), np.broadcast_to(j, (n, n))


def one_x_two(m: MatchModel) -> dict:
    M = m.matrix
    p_home = float(np.tril(M, -1).sum())   # home goals > away goals
    p_away = float(np.triu(M, 1).sum())    # away goals > home goals
    p_draw = float(np.trace(M))
    return {
        "home": {"prob": p_home, "fair_odds": fair_odds(p_home)},
        "draw": {"prob": p_draw, "fair_odds": fair_odds(p_draw)},
        "away": {"prob": p_away, "fair_odds": fair_odds(p_away)},
    }


def totals(m: MatchModel, line: float = 2.5) -> dict:
    n = m.matrix.shape[0]
    hi, hj = _grid(n)
    total = hi + hj
    p_over = float(m.matrix[total > line].sum())
    p_under = float(m.matrix[total < line].sum())
    p_push = float(m.matrix[total == line].sum())  # nonzero only for integer lines
    return {
        "line": line,
        "over": {"prob": p_over, "fair_odds": fair_odds(p_over)},
        "under": {"prob": p_under, "fair_odds": fair_odds(p_under)},
        "push": p_push,
    }


def asian_handicap(m: MatchModel, line: float = -0.5) -> dict:
    """Handicap applied to the HOME side (e.g. -0.5 = home must win).

    Returns home/away cover probabilities renormalised to exclude pushes,
    which is how Asian handicap stakes are settled.
    """
    n = m.matrix.shape[0]
    hi, hj = _grid(n)
    margin = hi - hj + line  # home margin after handicap
    p_home = float(m.matrix[margin > 0].sum())
    p_away = float(m.matrix[margin < 0].sum())
    p_push = float(m.matrix[margin == 0].sum())
    denom = p_home + p_away
    p_home_n = p_home / denom if denom else 0.0
    p_away_n = p_away / denom if denom else 0.0
    return {
        "line": line,
        "home": {"prob": p_home_n, "fair_odds": fair_odds(p_home_n)},
        "away": {"prob": p_away_n, "fair_odds": fair_odds(p_away_n)},
        "push": p_push,
    }


def btts(m: MatchModel) -> dict:
    M = m.matrix
    p_no = float(M[0, :].sum() + M[:, 0].sum() - M[0, 0])  # at least one side blanks
    p_yes = 1.0 - p_no
    return {
        "yes": {"prob": p_yes, "fair_odds": fair_odds(p_yes)},
        "no": {"prob": p_no, "fair_odds": fair_odds(p_no)},
    }


def correct_score(m: MatchModel, top_n: int = 8) -> pd.DataFrame:
    M = m.matrix
    rows = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            rows.append((f"{i}-{j}", float(M[i, j])))
    df = pd.DataFrame(rows, columns=["score", "prob"])
    df = df.sort_values("prob", ascending=False).head(top_n).reset_index(drop=True)
    df["fair_odds"] = df["prob"].map(fair_odds)
    return df


def goalscorer_props(
    players: pd.DataFrame, team: str, team_lambda: float
) -> pd.DataFrame:
    """Anytime / 2+ goalscorer probabilities for a team's players.

    The team's match expected goals (`team_lambda`) is distributed across the
    squad in proportion to each player's minutes-weighted attacking output, so
    expected minutes feed directly into every player's price.
    """
    s = players[players["team"] == team].copy()
    weight = s["xg90"] * (s["exp_minutes"] / 90.0)
    total = weight.sum()
    if total <= 0:
        s["exp_goals"] = 0.0
    else:
        # ~75% of team goals come from open-play field contribution we model here
        s["exp_goals"] = team_lambda * (weight / total)

    lam = s["exp_goals"].to_numpy()
    p_any = 1.0 - np.exp(-lam)
    p_two = 1.0 - np.exp(-lam) * (1.0 + lam)

    s = s.assign(
        anytime_prob=p_any,
        anytime_fair_odds=[fair_odds(p) for p in p_any],
        two_plus_prob=p_two,
        two_plus_fair_odds=[fair_odds(p) for p in p_two],
    )
    cols = [
        "player", "position", "exp_minutes", "exp_goals",
        "anytime_prob", "anytime_fair_odds", "two_plus_prob", "two_plus_fair_odds",
    ]
    return s.sort_values("anytime_prob", ascending=False)[cols].reset_index(drop=True)
