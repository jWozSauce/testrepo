"""ffbacktest: per-position NFL fantasy projection models with walk-forward backtesting.

Pipeline:
    data.py      -> season-level player dataset (half-PPR target) from nfl_data_py
    sources/     -> external feature tables merged by player-season (Madden, team proj)
    features.py  -> per-position modeling frames (lagged production + preseason-known attrs)
    models.py    -> per-position model specs + baselines
    backtest.py  -> walk-forward evaluation (no leakage) + metrics
    run.py       -> CLI entry point
"""

import os as _os
# Force single-threaded native binning: the joblib "threading" backend in
# HistGradientBoosting hits a numpy stride bug on this build. Set before sklearn's
# C extensions load; this package module imports before any submodule.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

__all__ = ["data", "features", "models", "backtest"]

POSITIONS = ["QB", "RB", "WR", "TE"]
