"""Per-position estimators and naive baselines.

HistGradientBoostingRegressor is the workhorse: it ingests NaNs natively, so a
player with no prior season (rookie) or a season with no Madden file simply carries
missing values instead of being dropped or imputed to a misleading zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# How many players "count" at each position for top-N fantasy hit-rate.
STARTER_N = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}


def make_model(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=3,
        max_iter=400,
        learning_rate=0.05,
        l2_regularization=1.0,
        min_samples_leaf=15,
        random_state=seed,
    )


def baseline_prev_points(test: pd.DataFrame) -> np.ndarray:
    """Predict last season's half-PPR total (per-game pace x games played)."""
    return (test["lag1_half_ppr_ppg"] * test["lag1_games"]).to_numpy()


def baseline_prev_ppg(test: pd.DataFrame) -> np.ndarray:
    """Predict last season's per-game pace projected over a 17-game season."""
    return (test["lag1_half_ppr_ppg"] * 17).to_numpy()


def fill_baseline(pred: np.ndarray, train_y_mean: float) -> np.ndarray:
    out = np.asarray(pred, dtype=float)
    out[np.isnan(out)] = train_y_mean
    return out
