"""Command-line interface for the World Cup betting model.

Examples:
    python -m worldcup_betting.cli teams
    python -m worldcup_betting.cli price Brazil Argentina
    python -m worldcup_betting.cli price Brazil Argentina --out "Karim Park"
    python -m worldcup_betting.cli edges            # scan the bundled sample odds
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

import worldcup_betting as wb
from .edges import find_edges

ODDS_PATH = os.path.join(os.path.dirname(__file__), "data", "odds_sample.csv")


def _fmt(p: float) -> str:
    return f"{p*100:5.1f}%"


def cmd_teams(_: argparse.Namespace) -> None:
    model = wb.load_model()
    s = model.strengths.sort_values("attack_index", ascending=False)
    print(f"{'team':<16}{'attack':>8}{'defense':>9}  (index, 1.0 = field average)")
    for _, r in s.iterrows():
        print(f"{r['team']:<16}{r['attack_index']:>8.2f}{r['defense_index']:>9.2f}")


def cmd_price(args: argparse.Namespace) -> None:
    unavailable = {args.home: args.out, args.away: args.out_away} if (
        args.out or args.out_away
    ) else None
    model = wb.load_model(unavailable)
    card = wb.price_match(model, args.home, args.away, home_adv=args.home_adv)

    h, a = card["fixture"]
    eg = card["expected_goals"]
    print(f"\n{h} vs {a}   (neutral)" if args.home_adv == 1.0 else f"\n{h} vs {a}")
    print(f"expected goals: {h} {eg['home']:.2f} - {eg['away']:.2f} {a}\n")

    o = card["one_x_two"]
    print("1X2:")
    print(f"  {h:<14} {_fmt(o['home']['prob'])}  fair {o['home']['fair_odds']}")
    print(f"  {'Draw':<14} {_fmt(o['draw']['prob'])}  fair {o['draw']['fair_odds']}")
    print(f"  {a:<14} {_fmt(o['away']['prob'])}  fair {o['away']['fair_odds']}")

    t = card["totals"][2.5]
    print(f"\nTotals 2.5:  Over {_fmt(t['over']['prob'])} (fair {t['over']['fair_odds']})"
          f"   Under {_fmt(t['under']['prob'])} (fair {t['under']['fair_odds']})")
    b = card["btts"]
    print(f"BTTS:        Yes  {_fmt(b['yes']['prob'])} (fair {b['yes']['fair_odds']})"
          f"   No    {_fmt(b['no']['prob'])} (fair {b['no']['fair_odds']})")

    print("\nMost likely scores:")
    for _, r in card["correct_score"].head(5).iterrows():
        print(f"  {r['score']}   {_fmt(r['prob'])}  fair {r['fair_odds']}")

    print(f"\nTop scorers — {h}:")
    for _, r in card["props_home"].head(5).iterrows():
        print(f"  {r['player']:<18} {r['exp_minutes']:>4.0f}'  anytime {_fmt(r['anytime_prob'])}"
              f"  fair {r['anytime_fair_odds']}")
    print(f"Top scorers — {a}:")
    for _, r in card["props_away"].head(5).iterrows():
        print(f"  {r['player']:<18} {r['exp_minutes']:>4.0f}'  anytime {_fmt(r['anytime_prob'])}"
              f"  fair {r['anytime_fair_odds']}")


def cmd_edges(args: argparse.Namespace) -> None:
    model = wb.load_model()
    odds = pd.read_csv(args.odds)
    rows = []
    for fixture, grp in odds.groupby("fixture"):
        home, away = [s.strip() for s in fixture.split(" vs ")]
        card = wb.price_match(model, home, away)
        prob_lookup = _selection_probs(card, home, away)
        for _, r in grp.iterrows():
            key = (r["market"], r["selection"])
            mp = prob_lookup.get(key)
            if mp is None:
                continue
            rows.append({
                "market": f"{fixture} | {r['market']}",
                "selection": r["selection"],
                "model_prob": mp,
                "market_odds": r["market_odds"],
            })

    found = find_edges(rows, min_ev=args.min_ev, kelly_multiplier=args.kelly)
    if found.empty:
        print(f"No edges >= {args.min_ev:.0%} EV in {len(rows)} priced selections.")
        return
    print(f"Found {len(found)} edges (>= {args.min_ev:.0%} EV), quarter-Kelly stakes:\n")
    for _, r in found.iterrows():
        print(f"  +{r['ev']*100:4.1f}% EV  {r['selection']:<26} @ {r['market_odds']:>5}"
              f"  (model {_fmt(r['model_prob'])} vs implied {_fmt(r['market_implied'])})"
              f"  stake {r['kelly']*100:.1f}% bankroll")
        print(f"            {r['market']}")


def _selection_probs(card: dict, home: str, away: str) -> dict:
    """Map (market, selection) labels used in the odds file to model probs."""
    o = card["one_x_two"]
    out = {
        ("1X2", f"{home} win"): o["home"]["prob"],
        ("1X2", "Draw"): o["draw"]["prob"],
        ("1X2", f"{away} win"): o["away"]["prob"],
        ("Totals 2.5", "Over 2.5"): card["totals"][2.5]["over"]["prob"],
        ("Totals 2.5", "Under 2.5"): card["totals"][2.5]["under"]["prob"],
        ("BTTS", "Yes"): card["btts"]["yes"]["prob"],
        ("BTTS", "No"): card["btts"]["no"]["prob"],
    }
    for side, label in (("props_home", home), ("props_away", away)):
        for _, r in card[side].iterrows():
            out[("Anytime scorer", f"{r['player']} ({label})")] = float(r["anytime_prob"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="World Cup betting model")
    sub = p.add_subparsers(required=True)

    sp = sub.add_parser("teams", help="list teams and strength indices")
    sp.set_defaults(func=cmd_teams)

    sp = sub.add_parser("price", help="price all markets for a fixture")
    sp.add_argument("home")
    sp.add_argument("away")
    sp.add_argument("--home-adv", type=float, default=1.0, dest="home_adv")
    sp.add_argument("--out", nargs="*", default=[], help="home players ruled out")
    sp.add_argument("--out-away", nargs="*", default=[], dest="out_away",
                    help="away players ruled out")
    sp.set_defaults(func=cmd_price)

    sp = sub.add_parser("edges", help="scan market odds for +EV bets")
    sp.add_argument("--odds", default=ODDS_PATH)
    sp.add_argument("--min-ev", type=float, default=0.02, dest="min_ev")
    sp.add_argument("--kelly", type=float, default=0.25, help="Kelly multiplier")
    sp.set_defaults(func=cmd_edges)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
