# NFL Fantasy Projection Backtest (half-PPR)

Per-position statistical models (QB / RB / WR / TE) that project **season-long
half-PPR fantasy points**, validated with a **walk-forward backtest** over several
seasons of history. Features combine `nfl_data_py` production data with **preseason
(launch) Madden ratings** and team/QB context.

Built as a 2026-season draft-prep project. Companion to the R scoring work in
`../NFL_Average_Scoring_Project_2.Rmd`.

## Why it's set up this way

- **Target = half-PPR season total** (`fantasy_points + 0.5·receptions`) — what you
  actually draft for.
- **Walk-forward, no leakage.** To predict season *Y*, models train only on
  player-seasons whose label year is `< Y`. Every feature (age, prior production,
  launch Madden rating, prior-year team context) is knowable before *Y* kicks off.
- **Per-position models.** Each position gets its own feature set (e.g. QB uses throw
  power/accuracy; WR uses route running, release, target share; RB uses carrying,
  elusiveness, prior carries).
- **`HistGradientBoostingRegressor`** ingests missing values natively, so a rookie
  (no prior season) or a season without a Madden file just carries NaNs instead of
  being dropped or zero-imputed.

## Layout

```
ffbacktest/
  data.py        pull + cache nfl_data_py; build season-level half-PPR dataset & team context
  features.py    per-position modeling frames (lagged production + preseason attrs + Madden + context)
  sources/
    madden.py    load preseason Madden launch ratings; harmonize per-year schemas; join by (name, season)
  models.py      per-position estimator + naive baselines; starter counts for hit-rate
  backtest.py    walk-forward engine + metrics (MAE/RMSE/Spearman/top-N hit/vs-baseline)
  run.py         CLI
data/madden/     drop launch-ratings files here (see below)
data_cache/      parquet cache of nfl_data_py pulls (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Walk-forward backtest across seasons (NFL features; Madden used where a file exists)
python -m ffbacktest.run backtest --start 2018 --end 2024 --show-top

# Restrict to positions / evaluation seasons
python -m ffbacktest.run backtest --start 2016 --end 2024 --eval 2023 2024 --pos WR TE

# Measure Madden's added value (needs Madden files in >= 2 labelled seasons)
python -m ffbacktest.run ablation --start 2018 --end 2024 --eval 2023 2024
```

### Metrics
| column | meaning |
|---|---|
| `mae`, `rmse` | point-projection error (half-PPR points) |
| `spearman` | rank accuracy — how well the ordering matches (draft-relevant) |
| `top_hit` | overlap of predicted vs actual top-N (N = 12 QB/TE, 24 RB/WR) |
| `mae_vs_prev` | model MAE minus "repeat last season's points" baseline (negative = model wins) |

## Madden launch-ratings files

Drop **preseason / day-one** launch spreadsheets into `data/madden/`. The loader
accepts `.csv` or `.xlsx` and auto-harmonizes the (very different) per-year schemas.

**Season mapping** — a Madden game is keyed to the season it is played during:

| file (game) | NFL season |
|---|---|
| Madden NFL 22 | 2021 |
| Madden NFL 23 | 2022 |
| Madden NFL 24 | 2023 |
| Madden NFL 25 *(2024 edition)* | 2024 |
| Madden NFL 26 | 2025 |

The game number is parsed from the filename (`madden24_launch.csv`,
`Madden2024Ratings.csv` → game 24 → 2023 season). **Only launch ratings** are used
so no mid-season rating updates leak into a preseason feature.

> ⚠️ "Madden NFL 25" is an ambiguous title — EA used it for both the **2013**
> 25th-anniversary edition and the **2024** edition. Use the 2024 roster (current
> players), not the 2013 one.

## Data notes

- `nfl_data_py` weekly/seasonal data is available through the **2024** season in this
  environment (2025 not yet published), so labelled backtests currently run through 2024.
- Madden's predictive **lift** can only be measured once launch files exist in **≥2 of
  the labelled training seasons** (the model must both train and test on the feature).
