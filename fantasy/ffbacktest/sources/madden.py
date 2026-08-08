"""Load preseason (launch/day-one) Madden ratings and key them to NFL seasons.

Files live in ``fantasy/data/madden/`` as either ``.xlsx`` or ``.csv``. The Madden
GAME NUMBER is parsed from the filename and mapped to the NFL season it is played
during:

    nfl_season = 2000 + game - 1      # Madden NFL 24 -> 2023 season

Both ``madden24_launch.xlsx`` (2-digit) and ``Madden2024Ratings.csv`` (4-digit,
last two digits = game number) resolve to game 24 -> 2023 season.

We use ONLY launch spreadsheets so the rating is genuinely known before the season
(no mid-season roster-update leakage).

Per-year sheets have inconsistent headers (split vs single route-running / throw
accuracy, typos like "Stength"/"Finnesse", "Full Name" vs "First Name"+"Last Name").
``load_one`` canonicalizes every year into one stable schema of ``madden_*`` features.
"""
from __future__ import annotations

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

MADDEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          "data", "madden")

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation and generational suffixes for joining."""
    if not isinstance(name, str):
        return ""
    n = name.lower().replace(".", " ").replace("-", " ").replace("'", "")
    n = re.sub(r"[^a-z ]", " ", n)
    parts = [p for p in n.split() if p and p not in _SUFFIXES]
    return " ".join(parts)


def _snake(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def _clean_key(col: str) -> str | None:
    """Normalize a raw header to a canonical snaked token across all Madden schemas.

    Handles three observed formats:
      * "Overall Rating"           (weebly, spaced)
      * "overallRating"/"firstName" (camelCase)
      * "stats/throwAccuracyDeep/value" + ".../diff"  (EA API export)
    Returns None for the throwaway ".../diff" columns.
    """
    c = str(col).strip()
    if re.search(r"/diff$", c, re.I):
        return None
    c = re.sub(r"(?i)^stats/", "", c)
    c = re.sub(r"(?i)/value$", "", c)
    c = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", c)          # camelCase boundary
    c = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", c)        # e.g. bCVision -> b_C_Vision
    return _snake(c)


def _season_from_filename(path: str) -> int | None:
    digits = re.findall(r"\d+", os.path.basename(path))
    if not digits:
        return None
    tok = max(digits, key=len)          # prefer the longest number token
    game = int(tok[-2:]) if len(tok) >= 2 else int(tok)
    return 2000 + game - 1


# canonical madden feature -> ordered candidate source headers (snaked, exact match)
_SYN = {
    "overall": ["overall_rating", "overall"],
    "speed": ["speed"],
    "acceleration": ["acceleration"],
    "agility": ["agility"],
    "strength": ["strength", "stength"],
    "awareness": ["awareness"],
    "injury": ["injury"],
    "stamina": ["stamina"],
    "jumping": ["jumping"],
    "catching": ["catching"],
    "catch_in_traffic": ["catch_in_traffic"],
    "spectacular_catch": ["spectacular_catch"],
    "release": ["release"],
    "carrying": ["carrying"],
    "trucking": ["trucking"],
    "stiff_arm": ["stiff_arm"],
    "juke_move": ["juke_move"],
    "spin_move": ["spin_move"],
    "elusiveness": ["elusiveness"],
    "break_tackle": ["break_tackle"],
    "ball_carrier_vision": ["ball_carrier_vision", "b_c_vision", "bc_vision"],
    "break_sack": ["break_sack"],
    "throw_power": ["throw_power"],
    "throw_on_the_run": ["throw_on_the_run"],
    "play_action": ["play_action", "playaction"],
    "throw_under_pressure": ["throw_under_pressure"],
    # split vs single variants (harmonized into composites below)
    "srr": ["short_route_running"],
    "mrr": ["medium_route_running"],
    "drr": ["deep_route_running"],
    "route_running": ["route_running"],
    "tas": ["throw_accuracy_short", "short_throw_accuracy"],
    "tam": ["throw_accuracy_mid", "medium_throw_accuracy"],
    "tad": ["throw_accuracy_deep", "deep_throw_accuracy"],
    "throw_accuracy": ["throw_accuracy"],
    "m_age": ["age"],
    "m_years_pro": ["years_pro", "years_pro_"],
}


def _pick(cols: dict, key: str):
    for cand in _SYN.get(key, []):
        if cand in cols:
            return cols[cand]
    return None


def load_one(path: str) -> pd.DataFrame | None:
    season = _season_from_filename(path)
    if season is None:
        return None
    raw = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    cols = {}
    for c in raw.columns:
        k = _clean_key(c)
        if k and k not in cols:      # skip /diff cols; keep first on collision
            cols[k] = raw[c]

    # player name: "full_name" OR "first_name" + "last_name"
    if "full_name" in cols:
        name = cols["full_name"]
    elif "first_name" in cols and "last_name" in cols:
        name = cols["first_name"].astype(str) + " " + cols["last_name"].astype(str)
    else:
        name = raw.iloc[:, 2]

    def num(key):
        s = _pick(cols, key)
        return pd.to_numeric(s, errors="coerce") if s is not None else None

    out = pd.DataFrame({
        "season": season,
        "name_key": name.map(normalize_name),
        "madden_team": cols.get("team"),
        "madden_position": cols.get("position"),
    })

    # direct pass-through attributes
    for k in ["overall", "speed", "acceleration", "agility", "strength", "awareness",
              "injury", "stamina", "jumping", "catching", "catch_in_traffic",
              "spectacular_catch", "release", "carrying", "trucking", "stiff_arm",
              "ball_carrier_vision", "break_sack", "throw_power", "throw_on_the_run",
              "play_action", "throw_under_pressure", "m_age", "m_years_pro"]:
        v = num(k)
        if v is not None:
            out["madden_" + k] = v

    # composite: receiving route rating (mean of splits, else single "route_running")
    srr, mrr, drr, rr = num("srr"), num("mrr"), num("drr"), num("route_running")
    if srr is not None or mrr is not None or drr is not None:
        out["madden_route"] = pd.concat([x for x in [srr, mrr, drr] if x is not None], axis=1).mean(axis=1)
    elif rr is not None:
        out["madden_route"] = rr

    # composite: QB accuracy (mean of splits, else single "throw_accuracy")
    tas, tam, tad, ta = num("tas"), num("tam"), num("tad"), num("throw_accuracy")
    if tas is not None or tam is not None or tad is not None:
        out["madden_throw_acc"] = pd.concat([x for x in [tas, tam, tad] if x is not None], axis=1).mean(axis=1)
    elif ta is not None:
        out["madden_throw_acc"] = ta

    # composite: elusiveness (native, else mean of juke+spin)
    elu = num("elusiveness")
    if elu is not None:
        out["madden_elusiveness"] = elu
    else:
        juke, spin = num("juke_move"), num("spin_move")
        if juke is not None or spin is not None:
            out["madden_elusiveness"] = pd.concat([x for x in [juke, spin] if x is not None], axis=1).mean(axis=1)

    # de-dup normalized names within a season (keep highest overall)
    sort_key = "madden_overall" if "madden_overall" in out.columns else "name_key"
    out = (out.sort_values(sort_key, ascending=False)
              .drop_duplicates(["season", "name_key"], keep="first")
              .reset_index(drop=True))
    return out


def load_madden(madden_dir: str = MADDEN_DIR) -> pd.DataFrame:
    """Concatenate all available launch sheets into one (name_key, season) table."""
    paths = sorted(set(glob.glob(os.path.join(madden_dir, "*.xlsx")) +
                       glob.glob(os.path.join(madden_dir, "*.csv"))))
    frames = []
    seen_seasons = set()
    for path in paths:
        s = _season_from_filename(path)
        if s in seen_seasons:      # avoid double-loading same season (xlsx + csv dupe)
            continue
        df = load_one(path)
        if df is not None and len(df):
            frames.append(df)
            seen_seasons.add(s)
    if not frames:
        return pd.DataFrame(columns=["season", "name_key"])
    return pd.concat(frames, ignore_index=True)


def available_seasons(madden_dir: str = MADDEN_DIR) -> list[int]:
    seasons = set()
    for path in (glob.glob(os.path.join(madden_dir, "*.xlsx")) +
                 glob.glob(os.path.join(madden_dir, "*.csv"))):
        s = _season_from_filename(path)
        if s is not None:
            seasons.add(s)
    return sorted(seasons)


if __name__ == "__main__":
    m = load_madden()
    print("madden seasons available:", available_seasons())
    feats = [c for c in m.columns if c.startswith("madden_")]
    print("rows:", len(m), "| madden features:", len(feats))
    print("features:", feats)
    if len(m):
        print(m.groupby("season").size().to_string())
