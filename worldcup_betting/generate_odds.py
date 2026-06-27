"""Generate a sample bookmaker-odds file for a few fixtures.

Real use: replace this with actual market odds (from a sportsbook, an odds
API, or a CSV export) keyed by fixture/market/selection. The columns below are
the contract the edge finder expects.

To make the edge demo meaningful, each selection's fair probability is nudged
by a small random shade and then loaded with a bookmaker margin. Where the
shade outweighs the margin, the listed price beats the model -> a visible edge.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import worldcup_betting as wb

FIXTURES = [
    ("Brazil", "Switzerland"),
    ("France", "Denmark"),
    ("Argentina", "Mexico"),
    ("England", "USA"),
    ("Spain", "Japan"),
]

VIG = 0.05  # bookmaker margin baked into prices


def _price(prob: float, shade: float) -> float:
    p = float(np.clip(prob * (1 + shade) * (1 + VIG), 1e-4, 0.999))
    return round(1.0 / p, 2)


def generate(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    model = wb.load_model()
    rows = []
    for home, away in FIXTURES:
        card = wb.price_match(model, home, away)
        fixture = f"{home} vs {away}"
        o = card["one_x_two"]
        sel = {
            ("1X2", f"{home} win"): o["home"]["prob"],
            ("1X2", "Draw"): o["draw"]["prob"],
            ("1X2", f"{away} win"): o["away"]["prob"],
            ("Totals 2.5", "Over 2.5"): card["totals"][2.5]["over"]["prob"],
            ("Totals 2.5", "Under 2.5"): card["totals"][2.5]["under"]["prob"],
            ("BTTS", "Yes"): card["btts"]["yes"]["prob"],
            ("BTTS", "No"): card["btts"]["no"]["prob"],
        }
        # one headline scorer prop per side
        for side, label in (("props_home", home), ("props_away", away)):
            top = card[side].iloc[0]
            sel[("Anytime scorer", f"{top['player']} ({label})")] = float(
                top["anytime_prob"]
            )

        for (market, selection), prob in sel.items():
            shade = rng.normal(0.0, 0.06)  # market disagreement vs model
            rows.append({
                "fixture": fixture,
                "market": market,
                "selection": selection,
                "market_odds": _price(prob, shade),
            })
    return pd.DataFrame(rows)


def main() -> None:
    out = os.path.join(os.path.dirname(__file__), "data", "odds_sample.csv")
    df = generate()
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} odds rows -> {out}")


if __name__ == "__main__":
    main()
