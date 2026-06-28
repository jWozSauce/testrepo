"""Build a real player-ratings CSV from StatsBomb's free open data.

StatsBomb publishes full event + lineup data (no key required) for several
tournaments, including the FIFA World Cup 2018 & 2022, the Euros and Copa
America: https://github.com/statsbomb/open-data

This module turns that raw data into the schema the model expects
(team, player, position, xg90, npxg90, def90, exp_minutes):

  * xg90  - REAL IMPACT: xGChain per 90 -- the total xG of every possession the
            player was involved in. Unlike direct npxG+xAG (which scores
            defenders and deep playmakers at ~0), xGChain credits everyone who
            builds an attack, so it measures a player's contribution to team
            expected goals. (xGBuildup, which also excludes the shot and the
            assist, is computed too and exposed as xgbuildup90.)
  * npxg90 - REAL FINISHING: non-penalty xG from the player's own shots per 90.
            Used for goalscorer props, where only the shooter's own xG matters.
  * def90 - HEURISTIC: per-90 volume of defensive actions (tackles, blocks,
            interceptions, clearances, recoveries, pressures), z-scored within
            position and mapped to a goals-prevented-style scale. Public event
            data has no true "goals prevented" metric; replace this with a
            bespoke defensive value if you have one.
  * exp_minutes - REAL usage: a player's average minutes per team match,
            renormalised so each squad sums to 990 (11 x 90).

Usage:
    python -m worldcup_betting.data_sources.statsbomb --competition 43 --season 106
    # writes worldcup_betting/data/players.csv (back up the synthetic one first)
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from ._decay import half_life_weight

warnings.filterwarnings("ignore")

# StatsBomb detailed position -> coarse group used by the model
POSITION_GROUP = {
    "Goalkeeper": "GK",
    "Right Back": "DEF", "Left Back": "DEF", "Right Center Back": "DEF",
    "Left Center Back": "DEF", "Center Back": "DEF",
    "Right Wing Back": "DEF", "Left Wing Back": "DEF",
    "Right Defensive Midfield": "MID", "Left Defensive Midfield": "MID",
    "Center Defensive Midfield": "MID", "Right Center Midfield": "MID",
    "Left Center Midfield": "MID", "Center Midfield": "MID",
    "Right Midfield": "MID", "Left Midfield": "MID",
    "Right Attacking Midfield": "MID", "Left Attacking Midfield": "MID",
    "Center Attacking Midfield": "MID",
    "Right Wing": "FWD", "Left Wing": "FWD", "Right Center Forward": "FWD",
    "Left Center Forward": "FWD", "Center Forward": "FWD", "Secondary Striker": "FWD",
}

DEF_ACTION_WEIGHTS = {
    "Interception": 1.0, "Block": 1.0, "Clearance": 0.8,
    "Ball Recovery": 0.6, "Duel": 0.7, "50/50": 0.4, "Pressure": 0.25,
}

# target def90 scale by position group (centre of the goals-prevented proxy)
DEF_BASE = {"GK": 0.55, "DEF": 0.34, "MID": 0.12, "FWD": 0.02}

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "players.csv")


def _clock(s) -> float | None:
    if not s or not isinstance(s, str) or ":" not in s:
        return None
    mm, ss = s.split(":")
    return int(mm) + int(ss) / 60.0


def _match_minutes(lineups: dict) -> dict:
    """player_name -> (minutes, primary_group) for one match."""
    out = {}
    for _team, df in lineups.items():
        for _, row in df.iterrows():
            spells = row["positions"] or []
            total = 0.0
            by_group: dict[str, float] = {}
            for p in spells:
                frm = _clock(p.get("from"))
                to = _clock(p.get("to"))
                if frm is None:
                    continue
                if to is None:
                    to = 95.0  # played to the final whistle (+ stoppage)
                dur = max(0.0, min(to, 98.0) - frm)
                total += dur
                grp = POSITION_GROUP.get(p.get("position"), "MID")
                by_group[grp] = by_group.get(grp, 0.0) + dur
            if total <= 0:
                continue
            primary = max(by_group, key=by_group.get) if by_group else "MID"
            out[row["player_name"]] = (min(total, 90.0), primary)
    return out


def build_players(
    windows: list[tuple[int, int]] | None = None,
    competition_id: int = 43,
    season_id: int = 106,
    half_life_days: float | None = None,
    as_of: str | None = None,
    max_matches: int | None = None,
) -> pd.DataFrame:
    """Aggregate player ratings, optionally across several tournaments and with
    recency decay.

    windows        list of (competition_id, season_id) to pool (default: the
                   single competition_id/season_id below).
    half_life_days exponential decay half-life in days; a match this many days
                   before the reference date counts half as much. None = equal
                   weighting.
    as_of          reference date 'YYYY-MM-DD' (default: the most recent match).
    """
    from statsbombpy import sb

    if windows is None:
        windows = [(competition_id, season_id)]

    # collect (match_id, date) across all pooled windows
    matchlist: list[tuple[int, "pd.Timestamp"]] = []
    for comp, seas in windows:
        m = sb.matches(competition_id=comp, season_id=seas)
        for _, r in m.iterrows():
            matchlist.append((int(r["match_id"]), pd.to_datetime(r["match_date"])))
    matchlist.sort(key=lambda x: x[1])
    if max_matches:
        matchlist = matchlist[-max_matches:]

    ref = pd.to_datetime(as_of) if as_of else max(d for _, d in matchlist)

    def match_weight(date: "pd.Timestamp") -> float:
        return half_life_weight((ref - date).days, half_life_days)

    acc: dict[tuple, dict] = {}        # (player, team) -> accumulators
    team_weight: dict[str, float] = {}  # team -> sum of match recency weights

    for n, (mid, mdate) in enumerate(matchlist, 1):
        w = match_weight(mdate)
        ev = sb.events(mid)
        lineups = sb.lineups(mid)
        minutes = _match_minutes(lineups)

        xg_col = "shot_statsbomb_xg"
        all_shots = ev[ev["type"] == "Shot"].copy()
        shots = all_shots
        if "shot_type" in shots.columns:
            shots = shots[shots["shot_type"] != "Penalty"]

        # finishing: a player's own non-penalty shot xG
        npxg = shots.groupby("player")[xg_col].sum().to_dict()

        # xGChain / xGBuildup: credit every player involved in a possession with
        # the xG that possession produced.
        chain: dict = {}
        buildup: dict = {}
        if "possession" in ev.columns and len(all_shots):
            poss_val = all_shots.groupby("possession")[xg_col].sum()
            kp_ids = set(all_shots["shot_key_pass_id"].dropna()) \
                if "shot_key_pass_id" in all_shots.columns else set()
            for poss, val in poss_val.items():
                if not val or val <= 0:
                    continue
                pe = ev[ev["possession"] == poss]
                team = pe["possession_team"].iloc[0]
                inv = pe[pe["team"] == team]
                for p in inv["player"].dropna().unique():
                    chain[p] = chain.get(p, 0.0) + float(val)
                bu = inv[(inv["type"] != "Shot") & (~inv["id"].isin(kp_ids))]
                for p in bu["player"].dropna().unique():
                    buildup[p] = buildup.get(p, 0.0) + float(val)

        defs = ev[ev["type"].isin(DEF_ACTION_WEIGHTS)]
        def_val: dict = {}
        for typ, w in DEF_ACTION_WEIGHTS.items():
            for player, c in defs[defs["type"] == typ]["player"].value_counts().items():
                def_val[player] = def_val.get(player, 0.0) + w * c

        # player -> team for this match (from lineups)
        p2team = {}
        for team, df in lineups.items():
            for nm in df["player_name"]:
                p2team[nm] = team

        seen_teams = set()
        for player, (mins, grp) in minutes.items():
            team = p2team.get(player)
            if team is None:
                continue
            if team not in seen_teams:
                team_weight[team] = team_weight.get(team, 0.0) + w
                seen_teams.add(team)
            key = (player, team)
            a = acc.setdefault(key, {"wmin": 0.0, "raw_min": 0.0, "npxg": 0.0,
                                     "chain": 0.0, "buildup": 0.0, "def": 0.0,
                                     "grp": {}})
            # all per-match contributions are scaled by the match's recency weight
            a["wmin"] += w * mins
            a["raw_min"] += mins
            a["npxg"] += w * float(npxg.get(player, 0.0))
            a["chain"] += w * float(chain.get(player, 0.0))
            a["buildup"] += w * float(buildup.get(player, 0.0))
            a["def"] += w * float(def_val.get(player, 0.0))
            a["grp"][grp] = a["grp"].get(grp, 0.0) + mins
        print(f"  processed {n}/{len(matchlist)} matches", end="\r")
    print()

    rows = []
    for (player, team), a in acc.items():
        if a["raw_min"] < 20 or a["wmin"] <= 0:  # drop cameo-only players
            continue
        per90 = 90.0 / a["wmin"]  # decay-weighted minutes -> decay-weighted rates
        grp = max(a["grp"], key=a["grp"].get)
        rows.append({
            "team": team, "player": player, "position": grp,
            "xg90": round(a["chain"] * per90, 3),        # impact (xGChain/90)
            "xgbuildup90": round(a["buildup"] * per90, 3),
            "npxg90": round(a["npxg"] * per90, 3),       # finishing for props
            "def_rate90": (a["def"]) * per90,
            # recency-weighted average minutes per match this player featured in
            "exp_minutes_raw": a["wmin"] / team_weight[team],
        })
    df = pd.DataFrame(rows)

    # def90: z-score defensive volume within position, map onto a sane scale
    df["def90"] = 0.0
    for grp, base in DEF_BASE.items():
        m = df["position"] == grp
        if m.sum() == 0:
            continue
        x = df.loc[m, "def_rate90"]
        z = (x - x.mean()) / (x.std() or 1.0)
        df.loc[m, "def90"] = (base * (1 + 0.30 * z)).clip(lower=0.0)
    df["def90"] = df["def90"].round(3)

    # exp_minutes: recency-weighted average minutes per match, renorm to 990
    team_total = df.groupby("team")["exp_minutes_raw"].transform("sum")
    df["exp_minutes"] = (df["exp_minutes_raw"] * 990.0 / team_total).round(1)

    return df[["team", "player", "position", "xg90", "xgbuildup90", "npxg90",
               "def90", "exp_minutes"]] \
        .sort_values(["team", "exp_minutes"], ascending=[True, False]) \
        .reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build players.csv from StatsBomb open data")
    ap.add_argument("--seasons", nargs="*", default=None,
                    help="pool several windows as comp:season (e.g. 43:106 223:282 55:282)")
    ap.add_argument("--competition", type=int, default=43, help="competition_id (43 = World Cup)")
    ap.add_argument("--season", type=int, default=106, help="season_id (106 = 2022, 3 = 2018)")
    ap.add_argument("--half-life-days", type=float, default=None, dest="half_life_days",
                    help="recency decay half-life in days (omit = equal weighting)")
    ap.add_argument("--as-of", default=None, dest="as_of", help="reference date YYYY-MM-DD")
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--out", default=os.path.abspath(DATA_PATH))
    args = ap.parse_args()

    windows = None
    if args.seasons:
        windows = [tuple(int(x) for x in s.split(":")) for s in args.seasons]

    df = build_players(windows=windows, competition_id=args.competition,
                       season_id=args.season, half_life_days=args.half_life_days,
                       as_of=args.as_of, max_matches=args.max_matches)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} real players across {df['team'].nunique()} teams -> {args.out}")


if __name__ == "__main__":
    main()
