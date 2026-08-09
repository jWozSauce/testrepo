"""Walk-forward backtest engine.

To predict season Y we train ONLY on player-seasons whose label year is < Y, so the
evaluation never sees information from the future. We report, per position and pooled:

  MAE / RMSE      absolute accuracy of the point projection
  spearman        rank accuracy (what matters for draft ordering)
  r2              variance explained
  top_hit         overlap of predicted vs actual top-N (N = starter count)
  mae_vs_prev     model MAE minus "repeat last year's points" baseline MAE (negative = better)

A ``use_madden=False`` run drops the Madden columns so you can measure their lift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import features as F
from . import models as M


def _metrics(pos, season, y, yhat, prev):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    mae = np.mean(np.abs(y - yhat))
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - yhat) ** 2) / ss if ss > 0 else np.nan
    rho = spearmanr(y, yhat).correlation if len(y) > 2 else np.nan
    n = M.STARTER_N.get(pos, 24)
    order_pred = np.argsort(-yhat)[:n]
    order_true = np.argsort(-y)[:n]
    hit = len(set(order_pred) & set(order_true)) / n
    prev_mae = np.mean(np.abs(y - np.asarray(prev, float)))
    return {"pos": pos, "season": season, "n": len(y), "mae": mae, "rmse": rmse,
            "r2": r2, "spearman": rho, "top_hit": hit,
            "mae_vs_prev": mae - prev_mae}


def walk_forward(span_start: int, span_end: int, eval_seasons,
                 positions=("QB", "RB", "WR", "TE"),
                 use_madden: bool = True, seed: int = 0):
    """Return (per_season_df, predictions_df)."""
    frames, feat_cols = F.build_frames(span_start, span_end, positions)
    rows, preds = [], []
    for pos in positions:
        if pos not in frames:
            continue
        f = frames[pos]
        feats = list(feat_cols[pos])
        if not use_madden:
            feats = [c for c in feats if not c.startswith("madden_")]
        for Y in eval_seasons:
            train = f[f["season"] < Y]
            test = f[f["season"] == Y]
            if len(train) < 40 or test.empty:
                continue
            # HistGB's binning crashes on features with <2 distinct non-NaN values
            # in the training slice, so keep only columns with real variation.
            usable = [c for c in feats if train[c].nunique(dropna=True) >= 2]
            Xtr, ytr = train[usable].to_numpy(float), train["y"].to_numpy(float)
            Xte, yte = test[usable].to_numpy(float), test["y"].to_numpy(float)
            model = M.make_model(seed)
            model.fit(Xtr, ytr)
            yhat = model.predict(Xte)
            prev = M.fill_baseline(M.baseline_prev_points(test), ytr.mean())
            rows.append(_metrics(pos, Y, yte, yhat, prev))
            p = test[["player_name", "season", "position", "y"]].copy()
            p["pred"] = yhat
            preds.append(p)
    per_season = pd.DataFrame(rows)
    predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return per_season, predictions


def project_season(target: int, span_start: int,
                   positions=("QB", "RB", "WR", "TE"), seed: int = 0):
    """Train per-position models on all labelled seasons < target, then project the
    target season from its (unlabelled) preseason feature frame. Returns a ranked
    projection DataFrame."""
    train_frames, feat_cols = F.build_frames(span_start, target - 1, positions)
    proj_frames, _ = F.build_projection_frame(target, positions)
    out = []
    for pos in positions:
        if pos not in train_frames or pos not in proj_frames:
            continue
        f, proj = train_frames[pos], proj_frames[pos]
        feats = [c for c in feat_cols[pos]
                 if c in proj.columns and f[c].nunique(dropna=True) >= 2]
        model = M.make_model(seed)
        model.fit(f[feats].to_numpy(float), f["y"].to_numpy(float))
        proj = proj.copy()
        proj["proj_half_ppr"] = model.predict(proj[feats].to_numpy(float))
        proj["pos_rank"] = proj["proj_half_ppr"].rank(ascending=False, method="first")
        out.append(proj[["player_name", "position", "team", "age",
                         "madden_overall", "proj_half_ppr", "pos_rank"]])
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return res.sort_values(["position", "proj_half_ppr"], ascending=[True, False])


def pooled(per_season: pd.DataFrame) -> pd.DataFrame:
    """Average metrics across evaluated seasons, per position (n-weighted where sensible)."""
    if per_season.empty:
        return per_season
    def agg(g):
        w = g["n"]
        return pd.Series({
            "seasons": g["season"].nunique(),
            "players": int(w.sum()),
            "mae": np.average(g["mae"], weights=w),
            "rmse": np.average(g["rmse"], weights=w),
            "spearman": np.average(g["spearman"], weights=w),
            "top_hit": np.average(g["top_hit"], weights=w),
            "mae_vs_prev": np.average(g["mae_vs_prev"], weights=w),
        })
    return per_season.groupby("pos").apply(agg).reset_index()
