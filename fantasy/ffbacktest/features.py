"""Per-position modeling frames: lagged production + preseason-known attrs + Madden + context.

For a target season Y, every feature is knowable BEFORE Y kicks off:
  * preseason attrs from the season-Y roster: age, years of experience, draft slot
  * lagged production from seasons Y-1 and Y-2 (prefixes lag1_, lag2_)
  * preseason (launch) Madden ratings for season Y
  * the player's team's PRIOR-year offensive context (QB/offense quality proxy)

The label ``y`` is season-Y half-PPR points. No season-Y production leaks in.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from . import data as D
from .sources.madden import load_madden, normalize_name

# union of production columns to build lag tables from (whatever is present)
_LAG_STATS = [
    "half_ppr", "half_ppr_ppg", "games",
    "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
    "target_share", "air_yards_share", "wopr", "racr", "receiving_epa",
    "carries", "rushing_yards", "rushing_tds", "rushing_epa", "rushing_first_downs",
    "attempts", "completions", "passing_yards", "passing_tds", "interceptions",
    "passing_epa", "dakota",
]

# Per-position lag stats: only the production that matters for that position, so a QB
# does not carry (near-zero) receiving columns and a WR does not carry passing columns.
_COMMON_LAG = ["half_ppr", "half_ppr_ppg", "games"]
POSITION_LAG = {
    "QB": _COMMON_LAG + ["attempts", "completions", "passing_yards", "passing_tds",
                         "interceptions", "passing_epa", "dakota",
                         "carries", "rushing_yards", "rushing_tds", "rushing_epa",
                         "rushing_first_downs"],
    "RB": _COMMON_LAG + ["carries", "rushing_yards", "rushing_tds", "rushing_epa",
                         "rushing_first_downs", "targets", "receptions",
                         "receiving_yards", "receiving_tds", "target_share",
                         "receiving_epa"],
    "WR": _COMMON_LAG + ["targets", "receptions", "receiving_yards", "receiving_tds",
                         "receiving_air_yards", "target_share", "air_yards_share",
                         "wopr", "racr", "receiving_epa", "carries", "rushing_yards"],
    "TE": _COMMON_LAG + ["targets", "receptions", "receiving_yards", "receiving_tds",
                         "receiving_air_yards", "target_share", "air_yards_share",
                         "wopr", "racr", "receiving_epa"],
}

# Madden features that matter for each position (harmonized canonical names)
_MADDEN_COMMON = ["madden_overall", "madden_speed", "madden_acceleration",
                  "madden_agility", "madden_awareness", "madden_injury"]
POSITION_MADDEN = {
    "QB": _MADDEN_COMMON + ["madden_throw_power", "madden_throw_acc",
                            "madden_throw_on_the_run", "madden_play_action",
                            "madden_break_sack", "madden_carrying"],
    "RB": _MADDEN_COMMON + ["madden_carrying", "madden_trucking", "madden_elusiveness",
                            "madden_ball_carrier_vision", "madden_stiff_arm",
                            "madden_catching", "madden_route"],
    "WR": _MADDEN_COMMON + ["madden_catching", "madden_route", "madden_release",
                            "madden_catch_in_traffic", "madden_spectacular_catch",
                            "madden_jumping"],
    "TE": _MADDEN_COMMON + ["madden_catching", "madden_route", "madden_release",
                            "madden_catch_in_traffic", "madden_spectacular_catch"],
}

# Within-team positional depth signal (available every season, incl. overall-only years).
# Added to every position: a high rating behind a higher one means a backup role.
_MADDEN_DEPTH = ["madden_pos_rank", "madden_ovr_gap_to_top",
                 "madden_is_top1", "madden_team_pos_n"]
for _p in POSITION_MADDEN:
    POSITION_MADDEN[_p] = POSITION_MADDEN[_p] + _MADDEN_DEPTH

_TEAM_CTX = ["team_pass_epa", "team_pass_yards", "team_pass_tds",
             "team_rush_yards", "team_rush_tds", "team_rush_epa"]

_ATTR = ["age", "years_exp", "draft_number", "is_rookie"]


def _lagged(ps: pd.DataFrame, k: int) -> pd.DataFrame:
    cols = [c for c in _LAG_STATS if c in ps.columns]
    lag = ps[["player_id", "season"] + cols].copy()
    lag["season"] = lag["season"] + k
    return lag.rename(columns={c: f"lag{k}_{c}" for c in cols})


def build_frames(span_start: int, span_end: int,
                 positions=("QB", "RB", "WR", "TE")):
    """Return (frames, feature_cols) where frames[pos] is a modeling DataFrame.

    span covers the seasons for which we build LABELLED rows; lag features reach
    back two more seasons, which are pulled automatically.
    """
    pull = list(range(span_start - 2, span_end + 1))
    # 2025+ weekly may be unpublished; pull only what exists.
    avail = []
    for y in pull:
        try:
            D.load_weekly([y]); avail.append(y)
        except Exception:
            pass
    ps = D.build_player_seasons(avail)
    team = D.build_team_seasons(avail)
    madden = load_madden()

    # lagged team context (prior-year offense), keyed on (team, season)
    team_lag = team.copy()
    team_lag["season"] = team_lag["season"] + 1
    team_lag = team_lag.rename(columns={c: c for c in _TEAM_CTX})

    ps = ps.copy()
    ps["name_key"] = ps["player_name"].map(normalize_name)

    lag1, lag2 = _lagged(ps, 1), _lagged(ps, 2)

    frames, feat_cols = {}, {}
    for pos in positions:
        cur = ps[(ps["position"] == pos) & (ps["season"].between(span_start, span_end))].copy()
        if cur.empty:
            continue
        base_cols = ["player_id", "player_name", "name_key", "season", "position",
                     "team", "half_ppr"] + [c for c in _ATTR if c in cur.columns]
        frame = cur[base_cols].rename(columns={"half_ppr": "y"})
        frame = frame.merge(lag1, on=["player_id", "season"], how="left")
        frame = frame.merge(lag2, on=["player_id", "season"], how="left")
        frame = frame.merge(team_lag, on=["team", "season"], how="left")

        # merge preseason Madden for the target season
        mad_cols = ["season", "name_key"] + [c for c in POSITION_MADDEN[pos]
                                             if c in madden.columns]
        frame = frame.merge(madden[mad_cols].drop_duplicates(["season", "name_key"]),
                            on=["season", "name_key"], how="left")

        lag_feats = [f"lag{k}_{s}" for k in (1, 2) for s in POSITION_LAG[pos]
                     if f"lag{k}_{s}" in frame.columns]
        mad_feats = [c for c in POSITION_MADDEN[pos] if c in frame.columns]
        ctx_feats = [c for c in _TEAM_CTX if c in frame.columns]
        attr_feats = [c for c in _ATTR if c in frame.columns]
        features = attr_feats + lag_feats + ctx_feats + mad_feats

        frame["has_prior"] = frame["lag1_half_ppr"].notna().astype(float)
        features.append("has_prior")

        frames[pos] = frame
        feat_cols[pos] = features
    return frames, feat_cols


def build_projection_frame(target: int, positions=("QB", "RB", "WR", "TE")):
    """Build unlabeled feature rows for a FUTURE season to project.

    Candidates come from the target-season rosters (there is no production data for a
    season that hasn't happened). Features mirror build_frames: prior-year (target-1)
    and two-year (target-2) production, target-season preseason attrs, prior-year team
    context, and target-season Madden launch ratings + depth rank.
    """
    avail = []
    for y in (target - 2, target - 1):
        try:
            D.load_weekly([y]); avail.append(y)
        except Exception:
            pass
    ps = D.build_player_seasons(avail)
    team = D.build_team_seasons(avail)
    madden = load_madden()

    team_lag = team.copy()
    team_lag["season"] = team_lag["season"] + 1
    lag1, lag2 = _lagged(ps, 1), _lagged(ps, 2)

    rosters = D.load_rosters([target]).copy()
    rosters = rosters.drop_duplicates("player_id", keep="first")
    rosters["name_key"] = rosters["player_name"].map(normalize_name)
    if "rookie_year" in rosters.columns:
        rosters["is_rookie"] = (rosters["rookie_year"] == target).astype(float)
    else:
        rosters["is_rookie"] = np.nan

    frames, feat_cols = {}, {}
    for pos in positions:
        cur = rosters[rosters["position"] == pos].copy()
        if cur.empty:
            continue
        cur["season"] = target
        base = ["player_id", "player_name", "name_key", "season", "position",
                "team"] + [c for c in _ATTR if c in cur.columns]
        frame = cur[base].copy()
        frame = frame.merge(lag1, on=["player_id", "season"], how="left")
        frame = frame.merge(lag2, on=["player_id", "season"], how="left")
        frame = frame.merge(team_lag, on=["team", "season"], how="left")
        mad_cols = ["season", "name_key"] + [c for c in POSITION_MADDEN[pos]
                                             if c in madden.columns]
        frame = frame.merge(madden[mad_cols].drop_duplicates(["season", "name_key"]),
                            on=["season", "name_key"], how="left")
        frame["has_prior"] = frame["lag1_half_ppr"].notna().astype(float)

        lag_feats = [f"lag{k}_{s}" for k in (1, 2) for s in POSITION_LAG[pos]
                     if f"lag{k}_{s}" in frame.columns]
        mad_feats = [c for c in POSITION_MADDEN[pos] if c in frame.columns]
        ctx_feats = [c for c in _TEAM_CTX if c in frame.columns]
        attr_feats = [c for c in _ATTR if c in frame.columns]
        frames[pos] = frame
        feat_cols[pos] = attr_feats + lag_feats + ctx_feats + mad_feats + ["has_prior"]
    return frames, feat_cols


def madden_match_rate(span_start: int, span_end: int) -> pd.DataFrame:
    """Diagnostic: share of position players per season matched to a Madden rating."""
    frames, _ = build_frames(span_start, span_end)
    rows = []
    for pos, f in frames.items():
        g = f.groupby("season").apply(
            lambda d: pd.Series({
                "players": len(d),
                "madden_matched": d["madden_overall"].notna().sum() if "madden_overall" in d else 0,
            }))
        g["pos"] = pos
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


if __name__ == "__main__":
    print(madden_match_rate(2023, 2024).to_string(index=False))
