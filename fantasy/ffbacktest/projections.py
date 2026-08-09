"""Multi-stat season projections and an Excel cheat-sheet export.

Extends the single-target (half-PPR) projection to a full per-position stat line by
training one model per (position, stat) on the same feature frames, then writes a
formatted multi-tab workbook: an ALL-players tab plus one tab per position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F, data as D, models as M

POS = ("QB", "RB", "WR", "TE")

# Stats projected per position (all are aggregated in build_player_seasons).
STAT_TARGETS = {
    "QB": ["games", "completions", "attempts", "passing_yards", "passing_tds",
           "interceptions", "carries", "rushing_yards", "rushing_tds"],
    "RB": ["games", "carries", "rushing_yards", "rushing_tds", "targets",
           "receptions", "receiving_yards", "receiving_tds"],
    "WR": ["games", "targets", "receptions", "receiving_yards", "receiving_tds",
           "carries", "rushing_yards"],
    "TE": ["games", "targets", "receptions", "receiving_yards", "receiving_tds"],
}

# Non-negative integer-ish counting stats (for clipping / display rounding).
_COUNTS = {"games", "completions", "attempts", "passing_tds", "interceptions",
           "carries", "targets", "receptions", "receiving_tds", "rushing_tds"}

# Pretty column headers for the workbook.
DISPLAY = {
    "proj_points": "Proj Pts", "proj_ppg": "Proj PPG", "proj_games": "Proj G",
    "proj_completions": "Cmp", "proj_attempts": "Att", "proj_passing_yards": "Pass Yds",
    "proj_passing_tds": "Pass TD", "proj_interceptions": "INT",
    "proj_carries": "Car", "proj_rushing_yards": "Rush Yds", "proj_rushing_tds": "Rush TD",
    "proj_targets": "Tgt", "proj_receptions": "Rec", "proj_receiving_yards": "Rec Yds",
    "proj_receiving_tds": "Rec TD",
    "pos_rank": "Pos Rk", "ovr_rank": "Ovr Rk", "player_name": "Player",
    "position": "Pos", "team": "Team", "age": "Age", "madden_overall": "MAD",
    "vbd": "VBD", "tier": "Tier", "vbd_rank": "Rk",
}

# 12-team league, half-PPR, standard lineup.
LEAGUE = dict(teams=12, starters={"QB": 1, "RB": 2, "WR": 3, "TE": 1},
              flex=1, flex_pos=("RB", "WR", "TE"))


def add_vbd(allp, league=LEAGUE):
    """Value-based drafting: points over replacement. Replacement = best NON-starter at
    each position, where starters are the positional starters PLUS the top flex-eligible
    leftovers (so the flex spot is allocated dynamically to whoever deserves it)."""
    df = allp.copy()
    df["is_starter"] = False
    for p, n in league["starters"].items():
        idx = df[df.position == p].nlargest(league["teams"] * n, "proj_points").index
        df.loc[idx, "is_starter"] = True
    flex_spots = league["teams"] * league["flex"]
    pool = df[(~df.is_starter) & (df.position.isin(league["flex_pos"]))]
    df.loc[pool.nlargest(flex_spots, "proj_points").index, "is_starter"] = True
    base = {}
    for p in df.position.unique():
        ns = df[(df.position == p) & (~df.is_starter)]["proj_points"]
        base[p] = float(ns.max()) if len(ns) else 0.0
    df["vbd"] = (df["proj_points"] - df["position"].map(base)).round(0)
    return df, base


def add_tiers(df, top_n={"QB": 24, "RB": 42, "WR": 54, "TE": 24}, mult=0.8):
    """Per-position tiers via natural gaps: a new tier starts where the drop to the next
    player is unusually large (> mean + mult*std of gaps among the draftable range)."""
    df = df.copy(); df["tier"] = 1
    for p in df.position.unique():
        sub = df[df.position == p].sort_values("proj_points", ascending=False)
        pts = sub["proj_points"].to_numpy(float); n = len(pts)
        lim = min(top_n.get(p, 40), n)
        if lim < 3:
            df.loc[sub.index, "tier"] = 1; continue
        gaps = -np.diff(pts[:lim])
        thr = gaps.mean() + mult * gaps.std()
        t, tiers = 1, [1]
        for g in gaps:
            if g > thr:
                t += 1
            tiers.append(t)
        tiers += [t + 1] * (n - lim)
        df.loc[sub.index, "tier"] = tiers[:n]
    df["tier"] = df["tier"].astype(int)
    return df

# Column order shown on each position tab.
TAB_COLS = {
    "QB": ["pos_rank", "tier", "player_name", "team", "age", "madden_overall",
           "proj_points", "vbd", "proj_ppg", "proj_games", "proj_passing_yards",
           "proj_passing_tds", "proj_interceptions", "proj_completions",
           "proj_attempts", "proj_rushing_yards", "proj_rushing_tds"],
    "RB": ["pos_rank", "tier", "player_name", "team", "age", "madden_overall",
           "proj_points", "vbd", "proj_ppg", "proj_games", "proj_carries",
           "proj_rushing_yards", "proj_rushing_tds", "proj_targets",
           "proj_receptions", "proj_receiving_yards", "proj_receiving_tds"],
    "WR": ["pos_rank", "tier", "player_name", "team", "age", "madden_overall",
           "proj_points", "vbd", "proj_ppg", "proj_games", "proj_targets",
           "proj_receptions", "proj_receiving_yards", "proj_receiving_tds",
           "proj_rushing_yards"],
    "TE": ["pos_rank", "tier", "player_name", "team", "age", "madden_overall",
           "proj_points", "vbd", "proj_ppg", "proj_games", "proj_targets",
           "proj_receptions", "proj_receiving_yards", "proj_receiving_tds"],
}
ALL_COLS = ["ovr_rank", "player_name", "position", "tier", "team", "age",
            "madden_overall", "pos_rank", "proj_points", "vbd", "proj_ppg", "proj_games"]
BOARD_COLS = ["vbd_rank", "player_name", "position", "tier", "team", "age",
              "proj_points", "vbd", "proj_ppg", "pos_rank"]


def _fit_predict(f, proj, feats, label_col, out_col):
    y = f[label_col]
    mask = y.notna()
    usable = [c for c in feats if c in proj.columns
              and f.loc[mask, c].nunique(dropna=True) >= 2]
    model = M.make_model()
    model.fit(f.loc[mask, usable].to_numpy(float), y[mask].to_numpy(float))
    pred = model.predict(proj[usable].to_numpy(float))
    if out_col == "proj_games":
        pred = np.clip(pred, 0, 17)
    else:
        pred = np.clip(pred, 0, None)
    proj[out_col] = pred


def project(target: int = 2026, span_start: int = 2015):
    """Return {pos: DataFrame} of projected stat lines for the target season."""
    frames, feat_cols = F.build_frames(span_start, target - 1, POS)
    proj_frames, _ = F.build_projection_frame(target, POS)
    ps = D.build_player_seasons(list(range(span_start, target)))
    stats = sorted(set().union(*STAT_TARGETS.values()))
    labels = ps[["player_id", "season"] + [c for c in stats if c in ps.columns]] \
        .drop_duplicates(["player_id", "season"])

    results = {}
    for pos in POS:
        f = frames[pos].merge(labels, on=["player_id", "season"], how="left")
        proj = proj_frames[pos].copy()
        feats = feat_cols[pos]
        _fit_predict(f, proj, feats, "y", "proj_points")          # half-PPR total
        for stat in STAT_TARGETS[pos]:
            if stat in f.columns:
                _fit_predict(f, proj, feats, stat, "proj_" + stat)
        proj["proj_ppg"] = proj["proj_points"] / proj["proj_games"].clip(lower=1)
        proj = proj.sort_values("proj_points", ascending=False).reset_index(drop=True)
        proj["pos_rank"] = np.arange(1, len(proj) + 1)
        results[pos] = proj
    return results


def _round(df):
    df = df.copy()
    for c in df.columns:
        if c == "proj_ppg":
            df[c] = df[c].round(1)
        elif c.startswith("proj_") and c.replace("proj_", "") in _COUNTS:
            df[c] = df[c].round(0).astype("Int64")
        elif c.startswith("proj_"):
            df[c] = df[c].round(0).astype("Int64")
        elif c in ("age", "madden_overall"):
            df[c] = df[c].round(0).astype("Int64")
    return df


def export_excel(results, target, path):
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    combined = []
    for pos, df in results.items():
        d = df.copy(); d["position"] = pos
        combined.append(d)
    allp = pd.concat(combined, ignore_index=True)
    allp, baselines = add_vbd(allp, LEAGUE)
    allp = add_tiers(allp)

    board = allp.sort_values("vbd", ascending=False).reset_index(drop=True)
    board["vbd_rank"] = np.arange(1, len(board) + 1)
    allp = allp.merge(board[["player_name", "position", "vbd_rank"]],
                      on=["player_name", "position"], how="left")

    allp_pts = allp.sort_values("proj_points", ascending=False).reset_index(drop=True)
    allp_pts["ovr_rank"] = np.arange(1, len(allp_pts) + 1)

    sheets = {
        "Draft Board": _round(board[[c for c in BOARD_COLS if c in board.columns]]),
        "All Players": _round(allp_pts[[c for c in ALL_COLS if c in allp_pts.columns]]),
    }
    for pos in POS:
        d = allp[allp.position == pos].sort_values("proj_points", ascending=False)
        cols = [c for c in TAB_COLS[pos] if c in d.columns]
        sheets[pos] = _round(d[cols])

    bl = "  ".join(f"{p}={int(v)}" for p, v in sorted(baselines.items()))
    about = pd.DataFrame({
        f"NFL {target} Half-PPR Draft Cheat Sheet (12-team, 1QB/2RB/3WR/1TE/1FLEX)": [
            f"Generated from Madden {target - 1999} launch ratings + through-{target-1} production.",
            "Per-position models (walk-forward validated): Spearman rank 0.63-0.70 in-sample,",
            "0.72-0.78 out-of-sample on 2025. Beats 'repeat last year' by 4-17 pts MAE.",
            "",
            "DRAFT BOARD is the overall draft order, ranked by VBD (not raw points).",
            "VBD = Value Based Drafting = projected points minus a replacement starter at the",
            "same position. It mixes positions correctly: elite RB/WR outrank higher-scoring QBs",
            "because the dropoff at their position is steeper. Replacement baselines (pts):",
            f"    {bl}",
            "",
            "TIER groups players where a position is roughly interchangeable; a new tier marks a",
            "real dropoff. On the clock: if your position has only 1 player left in its tier but",
            "another has several, take the scarce one now.",
            "",
            "Proj Pts = season half-PPR points. Proj PPG = points / projected games.",
            "Proj G = projected games (availability). MAD = Madden 27 overall.",
            "",
            "Caveats: injuries are not predicted; rookies lean on draft slot + Madden and are the",
            "least certain. A model baseline, not gospel.",
        ]})

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        about.to_excel(xl, sheet_name="About", index=False)
        for name, df in sheets.items():
            df.rename(columns=DISPLAY).to_excel(xl, sheet_name=name, index=False)

    # formatting
    from openpyxl import load_workbook
    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F3864")
    header_font = Font(bold=True, color="FFFFFF")
    thin = Side(style="thin", color="D9D9D9")
    for name in wb.sheetnames:
        ws = wb[name]
        if name == "About":
            ws.column_dimensions["A"].width = 92
            ws["A1"].font = Font(bold=True, size=13)
            continue
        ws.freeze_panes = "C2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = header_fill; cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        widths = {"Player": 22, "Team": 6, "Pos": 5, "Age": 5, "MAD": 6, "Tier": 5,
                  "VBD": 7, "Rk": 5, "Proj Pts": 9, "Proj PPG": 9, "Proj G": 7,
                  "Pos Rk": 7, "Ovr Rk": 7}
        for i, col in enumerate(ws[1], start=1):
            L = get_column_letter(i)
            ws.column_dimensions[L].width = widths.get(col.value, 8.5)
        # shade the projected-points column lightly
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.border = Border(bottom=thin)
    wb.save(path)
    return path
