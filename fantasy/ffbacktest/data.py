"""Build a season-level player dataset with a half-PPR fantasy target.

The unit of analysis is one (player_id, season) row. The target is season-long
half-PPR points, which is what you actually draft for. Half-PPR is computed as:

    half_ppr = fantasy_points (standard)  +  0.5 * receptions
             = fantasy_points_ppr         -  0.5 * receptions

All raw pulls from nfl_data_py are cached to parquet so repeated runs are fast.
"""
from __future__ import annotations

import os
import warnings
import pandas as pd
import numpy as np

warnings.simplefilter("ignore")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Weekly columns we aggregate to the season. Volume stats are summed; rate/share
# stats are averaged (weighted later if needed).
_SUM_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "sacks", "passing_epa", "passing_first_downs", "passing_air_yards",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles_lost",
    "rushing_epa", "rushing_first_downs",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_fumbles_lost", "receiving_epa", "receiving_first_downs",
    "receiving_air_yards", "receiving_yards_after_catch", "special_teams_tds",
    "fantasy_points", "fantasy_points_ppr",
]
_MEAN_COLS = ["target_share", "air_yards_share", "wopr", "racr", "pacr", "dakota"]


def _cache(name: str, builder):
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = builder()
    df.to_parquet(path, index=False)
    return df


# nfl_data_py 0.3.3 points at the retired player_stats release. Recent seasons live
# under the renamed nflverse release, with a few column renames.
_NFLVERSE_WEEKLY = ("https://github.com/nflverse/nflverse-data/releases/download/"
                    "stats_player/stats_player_week_{0}.parquet")
_WEEKLY_RENAME = {"team": "recent_team",
                  "passing_interceptions": "interceptions",
                  "sacks_suffered": "sacks"}
_WEEKLY_ID_COLS = ["player_id", "player_display_name", "player_name", "position",
                   "season", "week", "season_type", "recent_team", "opponent_team"]


def _weekly_year(year: int) -> pd.DataFrame:
    """One season of weekly stats: prefer nfl_data_py, fall back to the current
    nflverse release for seasons the installed library can't reach (e.g. 2025+)."""
    import nfl_data_py as nfl
    try:
        return nfl.import_weekly_data([year])
    except Exception:
        df = pd.read_parquet(_NFLVERSE_WEEKLY.format(year))
        return df.rename(columns={k: v for k, v in _WEEKLY_RENAME.items()
                                  if k in df.columns and v not in df.columns})


def load_weekly(seasons: list[int]) -> pd.DataFrame:
    """Regular-season weekly stat lines for the given seasons (cached).

    Raises if a season is unavailable from BOTH sources, so callers can probe
    availability with a try/except.
    """
    def build():
        frames = [_weekly_year(y) for y in seasons]
        w = pd.concat(frames, ignore_index=True)
        w = w[w["season_type"] == "REG"].copy()
        w["half_ppr"] = w["fantasy_points"].fillna(0) + 0.5 * w["receptions"].fillna(0)
        # keep a stable column subset so old/new schemas cache consistently
        keep = [c for c in _WEEKLY_ID_COLS + _SUM_COLS + _MEAN_COLS if c in w.columns]
        return w[keep + ["half_ppr"]]

    tag = f"weekly_{min(seasons)}_{max(seasons)}.parquet"
    return _cache(tag, build)


def load_rosters(seasons: list[int]) -> pd.DataFrame:
    """Seasonal rosters -> preseason-known player attributes (age, exp, team, position)."""
    import nfl_data_py as nfl

    def build():
        r = nfl.import_seasonal_rosters(list(seasons))
        keep = ["season", "player_id", "player_name", "position", "team",
                "age", "years_exp", "entry_year", "rookie_year", "draft_number"]
        keep = [c for c in keep if c in r.columns]
        r = r[keep].copy()
        # some seasons store these as mixed object dtype -> coerce so caching/modeling is clean
        for c in ["age", "years_exp", "entry_year", "rookie_year", "draft_number"]:
            if c in r.columns:
                r[c] = pd.to_numeric(r[c], errors="coerce")
        return r

    tag = f"rosters_{min(seasons)}_{max(seasons)}.parquet"
    return _cache(tag, build)


def build_player_seasons(seasons: list[int]) -> pd.DataFrame:
    """One row per (player_id, season): position, preseason attrs, season aggregates, target.

    Returns a tidy frame with:
      keys:     player_id, player_name, season, position, team
      attrs:    age, years_exp, draft_number, is_rookie
      volume:   summed weekly stats (see _SUM_COLS)
      rates:    mean weekly share/efficiency stats (see _MEAN_COLS)
      games:    number of regular-season games with a stat line
      targets:  half_ppr (season total, the label), half_ppr_ppg
    """
    weekly = load_weekly(seasons)
    rosters = load_rosters(seasons)

    sum_cols = [c for c in _SUM_COLS if c in weekly.columns]
    mean_cols = [c for c in _MEAN_COLS if c in weekly.columns]

    grp = weekly.groupby(["player_id", "season"])
    agg = grp.agg({**{c: "sum" for c in sum_cols},
                   **{c: "mean" for c in mean_cols}})
    agg["half_ppr"] = grp["half_ppr"].sum()
    agg["games"] = grp["week"].nunique()
    # dominant position from weekly appearances (fallback if roster missing)
    pos_wk = (weekly.groupby(["player_id", "season"])["position"]
              .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan))
    agg["position_wk"] = pos_wk
    agg = agg.reset_index()

    df = agg.merge(rosters, on=["player_id", "season"], how="left", suffixes=("", "_r"))
    df["position"] = df["position"].fillna(df["position_wk"])
    df["team"] = df.get("team")
    df["half_ppr_ppg"] = df["half_ppr"] / df["games"].clip(lower=1)
    # Per-game rate versions of volume stats, so a player who missed games is not
    # penalized on the production side (availability is handled separately). Rate/share
    # stats (_MEAN_COLS: target_share, wopr, ...) are already per-week averages.
    _PG = ["passing_yards", "passing_tds", "interceptions", "completions", "attempts",
           "carries", "rushing_yards", "rushing_tds", "rushing_first_downs",
           "targets", "receptions", "receiving_yards", "receiving_tds",
           "receiving_air_yards", "passing_epa", "rushing_epa", "receiving_epa"]
    g = df["games"].clip(lower=1)
    for c in _PG:
        if c in df.columns:
            df[c + "_pg"] = df[c] / g
    if "rookie_year" in df.columns:
        df["is_rookie"] = (df["rookie_year"] == df["season"]).astype("float")
    else:
        df["is_rookie"] = np.nan

    df = df.drop(columns=[c for c in ["position_wk"] if c in df.columns])
    return df


def build_team_seasons(seasons: list[int]) -> pd.DataFrame:
    """Team-season offensive aggregates (used, lagged, as offense / QB-quality context)."""
    weekly = load_weekly(seasons)
    g = weekly.groupby(["recent_team", "season"])
    team = g.agg(
        team_pass_epa=("passing_epa", "sum"),
        team_pass_yards=("passing_yards", "sum"),
        team_pass_tds=("passing_tds", "sum"),
        team_rush_yards=("rushing_yards", "sum"),
        team_rush_tds=("rushing_tds", "sum"),
        team_rush_epa=("rushing_epa", "sum"),
    ).reset_index().rename(columns={"recent_team": "team"})
    return team


if __name__ == "__main__":  # quick smoke test
    import sys
    yrs = list(range(2019, 2026))
    d = build_player_seasons(yrs)
    print("player-seasons:", d.shape)
    print(d[d.position == "WR"].sort_values("half_ppr", ascending=False)
          [["player_name", "season", "team", "games", "half_ppr", "half_ppr_ppg"]].head(8).to_string(index=False))
