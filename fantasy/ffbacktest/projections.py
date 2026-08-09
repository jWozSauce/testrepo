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
}

# Column order shown on each position tab.
TAB_COLS = {
    "QB": ["pos_rank", "player_name", "team", "age", "madden_overall", "proj_points",
           "proj_ppg", "proj_games", "proj_passing_yards", "proj_passing_tds",
           "proj_interceptions", "proj_completions", "proj_attempts",
           "proj_rushing_yards", "proj_rushing_tds"],
    "RB": ["pos_rank", "player_name", "team", "age", "madden_overall", "proj_points",
           "proj_ppg", "proj_games", "proj_carries", "proj_rushing_yards",
           "proj_rushing_tds", "proj_targets", "proj_receptions",
           "proj_receiving_yards", "proj_receiving_tds"],
    "WR": ["pos_rank", "player_name", "team", "age", "madden_overall", "proj_points",
           "proj_ppg", "proj_games", "proj_targets", "proj_receptions",
           "proj_receiving_yards", "proj_receiving_tds", "proj_rushing_yards"],
    "TE": ["pos_rank", "player_name", "team", "age", "madden_overall", "proj_points",
           "proj_ppg", "proj_games", "proj_targets", "proj_receptions",
           "proj_receiving_yards", "proj_receiving_tds"],
}
ALL_COLS = ["ovr_rank", "player_name", "position", "team", "age", "madden_overall",
            "pos_rank", "proj_points", "proj_ppg", "proj_games"]


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
    allp = allp.sort_values("proj_points", ascending=False).reset_index(drop=True)
    allp["ovr_rank"] = np.arange(1, len(allp) + 1)

    sheets = {"All Players": _round(allp[[c for c in ALL_COLS if c in allp.columns]])}
    for pos in POS:
        cols = [c for c in TAB_COLS[pos] if c in results[pos].columns]
        sheets[pos] = _round(results[pos][cols])

    about = pd.DataFrame({
        "NFL 2026 Half-PPR Projection Cheat Sheet": [
            f"Generated from Madden {target - 1999} launch ratings + through-{target-1} production.",
            "Per-position models (walk-forward validated): Spearman rank 0.63-0.70 in-sample,",
            "0.72-0.78 out-of-sample on 2025. Beats 'repeat last year' by 4-17 pts MAE.",
            "",
            "Proj Pts = projected season half-PPR points. Proj PPG = points / projected games.",
            "Proj G = projected games played (availability). Counting stats are season totals.",
            "MAD = Madden 27 launch overall rating.",
            "",
            "Caveats: injuries are not predicted (freak injuries drive the biggest misses);",
            "rookies lean on draft slot + Madden and are the least certain. Treat as a model",
            "baseline, not gospel.",
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
        widths = {"Player": 22, "Team": 6, "Pos": 5, "Age": 5, "MAD": 6,
                  "Proj Pts": 9, "Proj PPG": 9, "Proj G": 7, "Pos Rk": 7, "Ovr Rk": 7}
        for i, col in enumerate(ws[1], start=1):
            L = get_column_letter(i)
            ws.column_dimensions[L].width = widths.get(col.value, 8.5)
        # shade the projected-points column lightly
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.border = Border(bottom=thin)
    wb.save(path)
    return path
