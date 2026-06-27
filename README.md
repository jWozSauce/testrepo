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

> The bundled ratings are **synthetic but tier-calibrated** so it runs offline
> immediately and strong teams beat weak ones by believable margins. Swap in real
> numbers any time (see below) — the model code doesn't change.

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

## Use real data

- **Player ratings:** replace `worldcup_betting/data/players.csv` with your own.
  Required columns: `team, player, position, xg90, def90, exp_minutes`
  (keep each squad's `exp_minutes` summing to ~990). Regenerate the sample with
  `python -m worldcup_betting.generate_data`. Good sources: FBref / Opta per-90
  numbers, FIFA ratings mapped to xG, or your own rating model.
- **Market odds:** in the app's *Edge finder* tab, upload a CSV with columns
  `fixture, market, selection, market_odds` (decimal). The bundled
  `data/odds_sample.csv` shows the format; regenerate with
  `python -m worldcup_betting.generate_odds`.

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
  generate_data.py              build the sample player ratings
  generate_odds.py              build the sample bookmaker odds
  data/players.csv, odds_sample.csv
tests/test_model.py
```

---

*For research and education. Models are uncertain and the bundled ratings are
synthetic — do your own work before risking money, and bet responsibly.*
