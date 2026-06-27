# ⚽ World Cup Edge Finder

A player-rating betting model for the World Cup. It turns **per-player expected-goals
ratings, weighted by expected minutes played**, into team strength, prices every
common market, and compares those prices to the bookmaker to surface **+EV edges**.

Python backend, **Streamlit** front end — free to host and works on your phone.

## How it works

```
player ratings ──► team strength ──► match model ──► markets ──► edges
 (xg90, def90,      minutes-weighted   Dixon–Coles    1X2, O/U,    devig +
  exp_minutes)      attack/defense     bivariate      AH, BTTS,    EV + Kelly
                    indices            Poisson        correct score,
                                       score matrix   player props
```

1. **Ratings → team strength** (`worldcup_betting/ratings.py`).
   Each player has an attacking rating `xg90` (expected goals + assists per 90),
   a defensive rating `def90` (goals prevented per 90), and `exp_minutes` (how
   much of the match they're expected to play). A player's contribution is his
   per-90 rating × minutes share, summed over the squad. Expected minutes sum to
   990 (11 × 90) per team, so the minutes-weighted xG aggregates are physically
   consistent. Rule a player out and his minutes redistribute to the bench,
   automatically lowering the team's numbers.

2. **Match model** (`match_model.py`). Opponent-adjusted expected goals feed a
   **Dixon–Coles bivariate Poisson** to produce a full scoreline probability
   matrix (with the low-score correction that fixes plain-Poisson's 0-0/1-1 bias).

3. **Markets** (`markets.py`). Everything is priced off the score matrix:
   match result (1X2), totals (O/U), Asian handicap, BTTS, correct score, and
   **player props** (anytime / 2+ goalscorer) where the per-player xG share and
   minutes drive each price — the natural payoff of player-level data.

4. **Edges** (`edges.py`). Bookmaker odds → implied probability → de-vig →
   expected value → Kelly stake. Anything beating its price by your EV threshold
   is flagged.

> **The bundled `players.csv` is REAL data** — every player's `xg90` is their
> non-penalty xG + xA per 90 and `exp_minutes` is their actual usage, computed
> from **StatsBomb's free World Cup 2022 open data** (see *Data* below). Run
> `python -m worldcup_betting.generate_data` if you'd rather drop back to the
> synthetic tier-calibrated dataset for offline experiments.

## Run it

```bash
pip install -r requirements.txt

# Web app (phone-friendly UI)
streamlit run app.py

# CLI
python -m worldcup_betting.cli teams                 # strength table
python -m worldcup_betting.cli price Brazil Argentina
python -m worldcup_betting.cli price France Japan --out "Some Player"  # rule players out
python -m worldcup_betting.cli edges                 # scan bundled sample odds
```

## Put it on your phone (free)

1. Push this repo to GitHub (already set up).
2. Go to <https://share.streamlit.io>, sign in with GitHub, **New app**, pick this
   repo and `app.py`.
3. You get a public URL. Open it on your phone — Streamlit is mobile-responsive.

## Data

### Real player ratings (bundled, from StatsBomb open data)

The shipped `players.csv` is built from [StatsBomb's free open
data](https://github.com/statsbomb/open-data), which publishes full event +
lineup data (no API key) for the **World Cup 2018 & 2022**, the **Euros**,
**Copa América** and the **Women's World Cup**. The loader computes, per player:

- `xg90` — **real**: non-penalty StatsBomb xG + xA per 90 (xA credited to the
  key-pass player using each assisted shot's xG).
- `exp_minutes` — **real usage**: average minutes per team match, renormalised
  so each squad sums to 990.
- `def90` — **heuristic**: per-90 volume of defensive actions (tackles, blocks,
  interceptions, clearances, recoveries, pressures), z-scored within position.
  Public event data has no true "goals prevented" metric — swap in your own if
  you have one.

Rebuild it for any covered tournament:

```bash
pip install statsbombpy
python -m worldcup_betting.data_sources.statsbomb --competition 43 --season 106  # WC 2022
python -m worldcup_betting.data_sources.statsbomb --competition 43 --season 3    # WC 2018
# competition_id/season_id come from the StatsBomb competitions.json
```

> **Caveat — small samples.** A single tournament is only 3–7 matches per team,
> so per-90 rates are noisy and an eliminated side that generated lots of xG can
> look inflated. For stabler ratings, pool several tournaments or use full
> club-season data (below) and blend toward a prior.

### FBref club-season form (best for an upcoming tournament)

FBref's per-90 xG/xA over a full club season is a much bigger sample than one
tournament. FBref is Cloudflare-protected (hard to scrape), but you can **export
any table to CSV from your browser** — see
[`worldcup_betting/data/fbref/README.md`](worldcup_betting/data/fbref/README.md).
Drop the exports in `worldcup_betting/data/fbref/` and run:

```bash
python -m worldcup_betting.data_sources.fbref_csv     # auto-discovers the CSVs
```

It groups players into national teams by FBref's Nation column, derives `xg90`
from npxG + xAG per 90, computes a real `def90` if you include a Defensive
Actions export, and allocates expected minutes via a depth chart (club minutes
rank who starts). Pass `--roster roster.csv` (`team, player`) for official squads.

### Other free sources (and when to use them)

| Source | What you get | Best for | Access |
|---|---|---|---|
| **StatsBomb Open Data** | Event-level xG, lineups, minutes | World Cup / Euro / Copa squads (bundled here) | `statsbombpy`, free |
| **FBref** (Opta) | Per-90 xG/xA, defensive actions, minutes — all top leagues + internationals, full seasons | **An upcoming tournament**: map each national squad to its current club-season numbers (big sample) | [`soccerdata`](https://soccerdata.readthedocs.io) / [`worldfootballR`](https://jaseziv.github.io/worldfootballR/), free |
| **Understat** | Shot-level xG, xA, npxG | Club football (top-5 European leagues only — no national teams) | `soccerdata` / `understatapi`, free |
| **SoFIFA (FIFA ratings)** | Overall/attribute ratings (not xG) | A quality prior for minutes/selection or as a fallback | `soccerdata`, free |

To target a **future** World Cup, the realistic pipeline is: take each squad's
players → pull their **current club-season** per-90 xG/xA + minutes from FBref →
write them into the same `players.csv` columns. The model code doesn't change.

### Replace the data yourself

Any `players.csv` with columns `team, player, position, xg90, def90,
exp_minutes` works (keep each squad's `exp_minutes` summing to ~990).

### Market odds

In the app's *Edge finder* tab, upload a CSV with columns `fixture, market,
selection, market_odds` (decimal). The bundled `data/odds_sample.csv` shows the
format; regenerate with `python -m worldcup_betting.generate_odds`. For live
odds, [The Odds API](https://the-odds-api.com) has a free tier.

## Tests

```bash
PYTHONPATH=. python3 tests/test_model.py     # or: python -m pytest -q
```

## Project layout

```
app.py                          Streamlit front end
worldcup_betting/
  ratings.py                    players → minutes-weighted team strength
  match_model.py                strength → expected goals → score matrix
  markets.py                    price 1X2 / totals / AH / BTTS / scores / props
  edges.py                      devig, EV, Kelly, edge scan
  cli.py                        command-line interface
  generate_data.py              build the SYNTHETIC fallback player ratings
  generate_odds.py              build the sample bookmaker odds
  data_sources/statsbomb.py     build REAL ratings from StatsBomb open data
  data_sources/fbref_csv.py     build REAL ratings from FBref CSV exports
  data/fbref/                   drop your FBref CSV exports here
  data/players.csv, odds_sample.csv
tests/test_model.py
```

---

*For research and education. Models are uncertain and the bundled ratings are
synthetic — do your own work before risking money, and bet responsibly.*
