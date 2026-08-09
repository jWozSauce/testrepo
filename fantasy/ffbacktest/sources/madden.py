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
    """Lowercase, strip punctuation and generational suffixes, and collapse initial
    runs so both sides of a join agree.

    Datasets disagree on initials spacing ("D.K."->'d k' vs "DK"->'dk'), so runs of
    single-letter tokens are merged: ['d','k','metcalf'] -> 'dk metcalf'. This is exact
    (no fuzzy guessing) and cannot mis-attach a different player.
    """
    if not isinstance(name, str):
        return ""
    n = name.lower().replace(".", " ").replace("-", " ").replace("'", "")
    n = re.sub(r"[^a-z ]", " ", n)
    parts = [p for p in n.split() if p and p not in _SUFFIXES]
    merged, buf = [], ""
    for p in parts:
        if len(p) == 1:
            buf += p
        else:
            if buf:
                merged.append(buf)
                buf = ""
            merged.append(p)
    if buf:
        merged.append(buf)
    return " ".join(merged)


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
    "tam": ["throw_accuracy_mid", "throw_accuracy_middle", "medium_throw_accuracy"],
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


def _extract(path: str) -> pd.DataFrame | None:
    raw = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    cols = {}
    for c in raw.columns:
        k = _clean_key(c)
        if k and k not in cols:      # skip /diff cols; keep first on collision
            cols[k] = raw[c]

    # season, per row: prefer a "Madden_Year" column (game number + 2000, so the NFL
    # season is year - 1); otherwise fall back to the game number in the filename.
    # This lets one multi-year panel file cover many seasons at once.
    if "madden_year" in cols:
        season = pd.to_numeric(cols["madden_year"], errors="coerce") - 1
    else:
        s = _season_from_filename(path)
        if s is None:
            return None
        season = pd.Series(s, index=raw.index)

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

    def first_col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    out = pd.DataFrame({
        "season": season,
        "name_key": name.map(normalize_name),
        # team/position headers vary: "Team", "team/label", "position/shortLabel", ...
        "madden_team": first_col("team", "team_label", "team_name"),
        "madden_position": first_col("position", "position_short_label",
                                     "position_label", "position_name"),
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

    out = out.dropna(subset=["season"])
    out["season"] = out["season"].astype(int)
    # de-dup normalized names within a (season) (keep highest overall)
    sort_key = "madden_overall" if "madden_overall" in out.columns else "name_key"
    out = (out.sort_values(sort_key, ascending=False)
              .drop_duplicates(["season", "name_key"], keep="first")
              .reset_index(drop=True))
    _add_depth_rank(out)
    return out


def _add_depth_rank(out: pd.DataFrame) -> None:
    """Rank players by Madden overall within their team+position (a role/opportunity
    signal): rank 1 = the highest-rated player at that position on that team. A good
    rating that sits behind a better one usually means a backup with little production.

    Computed per file so team naming is internally consistent; the columns then flow
    through the (name, season) join. Works for overall-only seasons too, since the
    panel carries Team + Position.
    """
    if not {"madden_team", "madden_position", "madden_overall"}.issubset(out.columns):
        return
    g = out.groupby(["season", "madden_team", "madden_position"])["madden_overall"]
    out["madden_pos_rank"] = g.rank(ascending=False, method="first")
    out["madden_ovr_gap_to_top"] = g.transform("max") - out["madden_overall"]
    out["madden_is_top1"] = (out["madden_pos_rank"] == 1).astype(float)
    out["madden_team_pos_n"] = g.transform("count")


def _attr_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("madden_")
            and c not in ("madden_team", "madden_position")]


def load_madden(madden_dir: str = MADDEN_DIR) -> pd.DataFrame:
    """Merge every file into one (name_key, season) table.

    Files may be single-season sheets or multi-year panels. Where the same
    (season, player) appears in more than one file, keep the row with the most
    populated attributes, so a detailed launch sheet always beats an overall-only
    panel entry for the seasons they share.
    """
    paths = sorted(set(glob.glob(os.path.join(madden_dir, "*.xlsx")) +
                       glob.glob(os.path.join(madden_dir, "*.csv"))))
    frames = [df for df in (_extract(p) for p in paths) if df is not None and len(df)]
    if not frames:
        return pd.DataFrame(columns=["season", "name_key"])
    allm = pd.concat(frames, ignore_index=True)
    allm["_complete"] = allm[_attr_cols(allm)].notna().sum(axis=1)
    allm = (allm.sort_values(["_complete", "madden_overall"], ascending=False)
                .drop_duplicates(["season", "name_key"], keep="first")
                .drop(columns="_complete")
                .reset_index(drop=True))
    return allm


def available_seasons(madden_dir: str = MADDEN_DIR) -> list[int]:
    m = load_madden(madden_dir)
    return sorted(m["season"].unique().tolist()) if len(m) else []


if __name__ == "__main__":
    m = load_madden()
    print("madden seasons available:", available_seasons())
    feats = [c for c in m.columns if c.startswith("madden_")]
    print("rows:", len(m), "| madden features:", len(feats))
    print("features:", feats)
    if len(m):
        print(m.groupby("season").size().to_string())
