"""Player ratings -> minutes-weighted team strength.

The central idea: a player's contribution to his team is his per-90 rating
scaled by the share of a match he is expected to play. Summing those weighted
contributions over the squad gives team-level attacking and defensive numbers
that already account for who actually takes the field.

Because expected minutes sum to 990 (= 11 players x 90) per squad, the
minutes-weighted sum of player xG/90 equals the team's expected goals per 90
against an average opponent. Defensive ratings aggregate the same way into an
expected-goals-conceded figure.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# League-average expected goals scored by one team in one match (neutral).
LEAGUE_AVG_XG = 1.35

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "players.csv")


def load_players(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"team", "player", "position", "xg90", "def90", "exp_minutes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"players file is missing columns: {sorted(missing)}")
    return df


def apply_availability(
    players: pd.DataFrame, unavailable: dict[str, list[str]] | None
) -> pd.DataFrame:
    """Redistribute the minutes of unavailable players across their squad.

    `unavailable` maps team -> list of player names that are out (injury,
    suspension, rotation). Their expected minutes are removed and the squad is
    renormalised back to 990 so the remaining players absorb the load.
    """
    if not unavailable:
        return players
    df = players.copy()
    mask = pd.Series(False, index=df.index)
    for team, names in unavailable.items():
        if not names:
            continue
        mask |= (df["team"] == team) & (df["player"].isin(names))
    df.loc[mask, "exp_minutes"] = 0.0

    team_total = df.groupby("team")["exp_minutes"].transform("sum")
    # avoid divide-by-zero for an (unrealistic) fully-removed squad
    team_total = team_total.replace(0.0, np.nan)
    df["exp_minutes"] = (df["exp_minutes"] * 990.0 / team_total).fillna(0.0)
    return df


@dataclass
class TeamStrength:
    team: str
    attack_xg90: float       # minutes-weighted expected goals scored / 90
    xga90: float             # minutes-weighted expected goals conceded / 90
    attack_index: float      # relative to the field (1.0 = average)
    defense_index: float     # relative to the field (<1.0 = harder to score on)


def team_strengths(players: pd.DataFrame) -> pd.DataFrame:
    """Aggregate players into per-team attack/defense numbers and indices."""
    w = players["exp_minutes"] / 90.0
    contrib = players.assign(
        att_contrib=players["xg90"] * w,
        def_contrib=players["def90"] * w,
    )
    agg = contrib.groupby("team").agg(
        attack_xg90=("att_contrib", "sum"),
        def_rating=("def_contrib", "sum"),
    )

    # Convert the aggregate defensive rating into expected goals conceded:
    # a stronger defensive rating suppresses goals below the league baseline.
    league_def = agg["def_rating"].mean()
    agg["xga90"] = (LEAGUE_AVG_XG - (agg["def_rating"] - league_def)).clip(lower=0.3)

    agg["attack_index"] = agg["attack_xg90"] / agg["attack_xg90"].mean()
    agg["defense_index"] = agg["xga90"] / agg["xga90"].mean()
    return agg.reset_index()


def get_strength(strengths: pd.DataFrame, team: str) -> TeamStrength:
    row = strengths.loc[strengths["team"] == team]
    if row.empty:
        raise KeyError(f"unknown team: {team}")
    r = row.iloc[0]
    return TeamStrength(
        team=team,
        attack_xg90=float(r["attack_xg90"]),
        xga90=float(r["xga90"]),
        attack_index=float(r["attack_index"]),
        defense_index=float(r["defense_index"]),
    )


def squad(players: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return a team's players ordered by expected involvement."""
    s = players[players["team"] == team].copy()
    return s.sort_values("exp_minutes", ascending=False).reset_index(drop=True)
