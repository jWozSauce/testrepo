"""Ingest FBref CSV exports into the model's players.csv schema.

FBref lets you export any stats table from the browser:
  open a table -> "Share & Export" -> "Get table as CSV (for Excel)" -> copy the
  text into a .csv file. Drop one or more of those files into
  worldcup_betting/data/fbref/ and run this module.

Required: a *Player Standard Stats* export (Big 5 European Leagues, or any
league). Its columns include Player, Nation, Pos, Squad, Min, xG, npxG, xAG.
Optional: a *Defensive Actions* export (Tkl, Int, Blocks, Clr) to compute a
real def90 instead of a position default.

National "squads" are formed by grouping players by their FBref Nation, then
keeping the most-used players (club minutes are the proxy for who features).
This is club-form-based, not an official call-up list; supply a roster CSV via
--roster (columns: team, player) to restrict to actual squads when you have one.

Usage:
  python -m worldcup_betting.data_sources.fbref_csv \
      --standard data/fbref/big5_standard.csv \
      --defense  data/fbref/big5_defense.csv      # optional
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
FBREF_DIR = os.path.join(DATA_DIR, "fbref")

POS_GROUP = {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"}
DEF_BASE = {"GK": 0.55, "DEF": 0.34, "MID": 0.12, "FWD": 0.02}

# FBref 3-letter nation codes -> country names (extend as needed)
NATION = {
    "ARG": "Argentina", "FRA": "France", "BRA": "Brazil", "ENG": "England",
    "ESP": "Spain", "POR": "Portugal", "NED": "Netherlands", "GER": "Germany",
    "BEL": "Belgium", "ITA": "Italy", "CRO": "Croatia", "URU": "Uruguay",
    "MAR": "Morocco", "COL": "Colombia", "MEX": "Mexico", "USA": "United States",
    "SEN": "Senegal", "JPN": "Japan", "SUI": "Switzerland", "DEN": "Denmark",
    "KOR": "South Korea", "SRB": "Serbia", "POL": "Poland", "ECU": "Ecuador",
    "AUS": "Australia", "NGA": "Nigeria", "CAN": "Canada", "CMR": "Cameroon",
    "GHA": "Ghana", "KSA": "Saudi Arabia", "QAT": "Qatar", "IRN": "Iran",
    "WAL": "Wales", "TUN": "Tunisia", "CRC": "Costa Rica", "SCO": "Scotland",
    "NOR": "Norway", "SWE": "Sweden", "EGY": "Egypt", "CIV": "Ivory Coast",
    "ALG": "Algeria", "UKR": "Ukraine", "TUR": "Turkey", "AUT": "Austria",
}

# 4-3-3 depth chart used to allocate expected minutes within a national pool
FORMATION = {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}


def _read_fbref_csv(path: str) -> pd.DataFrame:
    """Read an FBref export, tolerating the optional over-header / comment row."""
    # FBref CSVs sometimes have a grouping row above the real header; find it.
    raw = pd.read_csv(path, header=0, dtype=str, keep_default_na=False)
    if "Player" not in raw.columns:
        raw = pd.read_csv(path, header=1, dtype=str, keep_default_na=False)
    raw = raw[raw["Player"].astype(str).str.strip().ne("")]
    raw = raw[raw["Player"] != "Player"]  # drop repeated header rows
    return raw


def _nation_name(code: str) -> str:
    code = (code or "").strip().split()[-1].upper() if code else ""
    return NATION.get(code, code or "Unknown")


def _pos_group(pos: str) -> str:
    first = (pos or "").strip().split(",")[0].strip().upper()
    return POS_GROUP.get(first, "MID")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce").fillna(0.0)


def _allocate_minutes(pool: pd.DataFrame) -> pd.Series:
    """Spread 990 match-minutes across a national pool via a depth chart.

    Club minutes rank players within each position; the depth-chart starters get
    a heavy share and the rest rotate in. Output sums to ~990 per national team.
    """
    minutes = pd.Series(0.0, index=pool.index)
    used = set()
    for grp, n_start in FORMATION.items():
        cand = pool[pool["position"] == grp].sort_values("club_min", ascending=False)
        starters = cand.index[:n_start]
        for rank, idx in enumerate(starters):
            minutes[idx] = 84.0 - 2.0 * rank   # 84,82,... slight taper
            used.add(idx)
    # bench: remaining players share rotation minutes by club usage
    bench = pool.index.difference(list(used))
    if len(bench):
        w = pool.loc[bench, "club_min"].clip(lower=1.0)
        minutes[bench] = 14.0 * w / w.sum() * min(len(bench), 12)
    total = minutes.sum()
    if total > 0:
        minutes = minutes * 990.0 / total
    return minutes.round(1)


def build_players(
    standard_paths: list[str],
    defense_paths: list[str] | None = None,
    creation_paths: list[str] | None = None,
    roster_path: str | None = None,
    min_club_minutes: int = 450,
    max_squad: int = 26,
) -> pd.DataFrame:
    frames = [_read_fbref_csv(p) for p in standard_paths]
    std = pd.concat(frames, ignore_index=True)

    df = pd.DataFrame({
        "player": std["Player"].str.strip(),
        "team": std["Nation"].map(_nation_name),
        "position": std["Pos"].map(_pos_group),
        "club_min": _num(std["Min"]),
        "npxg": _num(std.get("npxG", std.get("xG", 0))),
        "xag": _num(std.get("xAG", std.get("xA", 0))),
    })
    # a player can appear once per club spell; keep their largest-minutes row
    df = df.sort_values("club_min", ascending=False).drop_duplicates("player")
    df = df[df["club_min"] >= min_club_minutes]
    df = df[df["team"] != "Unknown"]

    nineties = (df["club_min"] / 90.0).clip(lower=0.1)
    # finishing (own-shot npxG per 90) -> goalscorer props
    df["npxg90"] = (df["npxg"] / nineties).round(3)
    # impact rating (xg90): prefer a Goal & Shot Creation export (SCA90, which
    # credits buildup, not just the finish); else fall back to direct npxG+xAG.
    df["xg90"] = ((df["npxg"] + df["xag"]) / nineties).round(3)
    if creation_paths:
        cre = pd.concat([_read_fbref_csv(p) for p in creation_paths], ignore_index=True)
        sca = pd.DataFrame({
            "player": cre["Player"].str.strip(),
            "sca90": _num(cre.get("SCA90", 0)),
        }).drop_duplicates("player")
        df = df.merge(sca, on="player", how="left")
        # SCA90 is in action units; scale it to sit on the same footing as the
        # direct xg90 it replaces, so the league-average anchor still holds.
        have = df["sca90"].notna() & (df["sca90"] > 0)
        if have.any():
            scale = df.loc[have, "xg90"].mean() / df.loc[have, "sca90"].mean()
            df.loc[have, "xg90"] = (df.loc[have, "sca90"] * scale).round(3)
        df = df.drop(columns=["sca90"])

    # def90: from a defensive-actions export if provided, else a position default
    df["def90"] = df["position"].map(DEF_BASE)
    if defense_paths:
        dfr = pd.concat([_read_fbref_csv(p) for p in defense_paths], ignore_index=True)
        dfn = pd.DataFrame({
            "player": dfr["Player"].str.strip(),
            "def_actions": _num(dfr.get("Tkl", 0)) + _num(dfr.get("Int", 0))
            + _num(dfr.get("Blocks", 0)) + _num(dfr.get("Clr", 0)),
            "d_min": _num(dfr.get("Min", dfr.get("90s", 0))),
        }).drop_duplicates("player")
        merged = df.merge(dfn, on="player", how="left")
        nn = (merged["d_min"].fillna(0) / 90.0).clip(lower=0.1)
        rate = (merged["def_actions"].fillna(0) / nn)
        merged["def_rate90"] = rate.values
        # z-score within position, map onto the DEF_BASE scale
        for grp, base in DEF_BASE.items():
            m = merged["position"] == grp
            if m.sum() > 1 and merged.loc[m, "def_rate90"].std():
                z = (merged.loc[m, "def_rate90"] - merged.loc[m, "def_rate90"].mean()) \
                    / merged.loc[m, "def_rate90"].std()
                merged.loc[m, "def90"] = (base * (1 + 0.30 * z)).clip(lower=0.0)
        df = merged
        df["def90"] = df["def90"].round(3)

    if roster_path:  # restrict to an official squad list if supplied
        roster = pd.read_csv(roster_path)
        keys = set(zip(roster["team"], roster["player"]))
        df = df[[(t, p) in keys for t, p in zip(df["team"], df["player"])]]

    # keep the most-used players per nation, then allocate expected minutes
    df = df.sort_values(["team", "club_min"], ascending=[True, False])
    df = df.groupby("team", group_keys=False).head(max_squad)

    out = []
    for team, pool in df.groupby("team"):
        pool = pool.copy()
        if len(pool) < 7:  # too few players in our coverage to field a side
            continue
        pool["exp_minutes"] = _allocate_minutes(pool)
        out.append(pool)
    result = pd.concat(out, ignore_index=True)

    return result[["team", "player", "position", "xg90", "npxg90", "def90",
                   "exp_minutes"]] \
        .sort_values(["team", "exp_minutes"], ascending=[True, False]) \
        .reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build players.csv from FBref CSV exports")
    ap.add_argument("--standard", nargs="*", help="Player Standard Stats CSV export(s)")
    ap.add_argument("--defense", nargs="*", default=None, help="Defensive Actions CSV export(s)")
    ap.add_argument("--creation", nargs="*", default=None,
                    help="Goal & Shot Creation CSV export(s) -> SCA90 impact rating")
    ap.add_argument("--roster", default=None, help="optional roster CSV (team, player)")
    ap.add_argument("--min-minutes", type=int, default=450, dest="min_minutes")
    ap.add_argument("--out", default=os.path.abspath(os.path.join(DATA_DIR, "players.csv")))
    args = ap.parse_args()

    standard = args.standard or sorted(glob.glob(os.path.join(FBREF_DIR, "*standard*.csv")))
    if not standard:
        raise SystemExit(
            "No standard-stats CSV given. Export 'Player Standard Stats' from FBref "
            f"and pass --standard PATH (or drop it in {FBREF_DIR}/ named *standard*.csv)."
        )
    defense = args.defense
    if defense is None:
        defense = sorted(glob.glob(os.path.join(FBREF_DIR, "*defens*.csv"))) or None
    creation = args.creation
    if creation is None:
        creation = sorted(glob.glob(os.path.join(FBREF_DIR, "*creation*.csv"))
                          + glob.glob(os.path.join(FBREF_DIR, "*gca*.csv"))) or None

    df = build_players(standard, defense, creation, args.roster, args.min_minutes)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} players across {df['team'].nunique()} teams -> {args.out}")
    if defense is None:
        print("note: no defensive-actions export found; def90 uses position defaults.")
    if creation is None:
        print("note: no Goal & Shot Creation export found; xg90 uses direct npxG+xAG "
              "(not an impact metric). Export that table for an SCA-based impact rating.")


if __name__ == "__main__":
    main()
