"""Build a real player-ratings CSV from StatsBomb's free open data.

StatsBomb publishes full event + lineup data (no key required) for several
tournaments, including the FIFA World Cup 2018 & 2022, the Euros and Copa
America: https://github.com/statsbomb/open-data

This module turns that raw data into the schema the model expects
(team, player, position, xg90, def90, exp_minutes):

  * xg90  - REAL: non-penalty expected goals + expected assists per 90,
            summed from per-shot StatsBomb xG (xA credited to the key-pass
            player using each assisted shot's xG).
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
    competition_id: int = 43, season_id: int = 106, max_matches: int | None = None
) -> pd.DataFrame:
    from statsbombpy import sb

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    match_ids = matches["match_id"].astype(int).tolist()
    if max_matches:
        match_ids = match_ids[:max_matches]

    acc: dict[tuple, dict] = {}        # (player, team) -> accumulators
    team_matches: dict[str, set] = {}  # team -> set(match_id)

    for n, mid in enumerate(match_ids, 1):
        ev = sb.events(mid)
        lineups = sb.lineups(mid)
        minutes = _match_minutes(lineups)

        shots = ev[ev["type"] == "Shot"].copy()
        if "shot_type" in shots.columns:
            shots = shots[shots["shot_type"] != "Penalty"]
        xg_col = "shot_statsbomb_xg"

        # xA: credit each assisted shot's xG to its key-pass player
        passer = dict(zip(ev["id"], ev["player"]))
        xa: dict = {}
        if "shot_key_pass_id" in shots.columns:
            for _, s in shots.dropna(subset=["shot_key_pass_id"]).iterrows():
                pid = passer.get(s["shot_key_pass_id"])
                if pid:
                    xa[pid] = xa.get(pid, 0.0) + float(s[xg_col] or 0.0)

        npxg = shots.groupby("player")[xg_col].sum().to_dict()

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

        for player, (mins, grp) in minutes.items():
            team = p2team.get(player)
            if team is None:
                continue
            team_matches.setdefault(team, set()).add(mid)
            key = (player, team)
            a = acc.setdefault(key, {"min": 0.0, "npxg": 0.0, "xa": 0.0,
                                     "def": 0.0, "grp": {}})
            a["min"] += mins
            a["npxg"] += float(npxg.get(player, 0.0))
            a["xa"] += float(xa.get(player, 0.0))
            a["def"] += float(def_val.get(player, 0.0))
            a["grp"][grp] = a["grp"].get(grp, 0.0) + mins
        print(f"  processed {n}/{len(match_ids)} matches", end="\r")
    print()

    rows = []
    for (player, team), a in acc.items():
        if a["min"] < 20:  # drop cameo-only players
            continue
        per90 = 90.0 / a["min"]
        grp = max(a["grp"], key=a["grp"].get)
        rows.append({
            "team": team, "player": player, "position": grp,
            "xg90": round((a["npxg"] + a["xa"]) * per90, 3),
            "def_rate90": (a["def"]) * per90,
            "minutes_total": a["min"],
            "team_matches": len(team_matches[team]),
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

    # exp_minutes: average minutes per team match, renormalised to 990 / squad
    df["exp_minutes"] = df["minutes_total"] / df["team_matches"]
    team_total = df.groupby("team")["exp_minutes"].transform("sum")
    df["exp_minutes"] = (df["exp_minutes"] * 990.0 / team_total).round(1)

    return df[["team", "player", "position", "xg90", "def90", "exp_minutes"]] \
        .sort_values(["team", "exp_minutes"], ascending=[True, False]) \
        .reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build players.csv from StatsBomb open data")
    ap.add_argument("--competition", type=int, default=43, help="competition_id (43 = World Cup)")
    ap.add_argument("--season", type=int, default=106, help="season_id (106 = 2022, 3 = 2018)")
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--out", default=os.path.abspath(DATA_PATH))
    args = ap.parse_args()

    df = build_players(args.competition, args.season, args.max_matches)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} real players across {df['team'].nunique()} teams -> {args.out}")


if __name__ == "__main__":
    main()
