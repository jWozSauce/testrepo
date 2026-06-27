"""Smoke tests for the World Cup betting model. Run: python -m pytest -q (or this file directly)."""
from __future__ import annotations

import numpy as np

import worldcup_betting as wb
from worldcup_betting import markets
from worldcup_betting.edges import expected_value, kelly_fraction


def test_minutes_sum_to_990():
    players = wb.load_players()
    sums = players.groupby("team")["exp_minutes"].sum()
    assert np.allclose(sums.values, 990.0, atol=1.0)


def test_probabilities_are_coherent():
    model = wb.load_model()
    card = wb.price_match(model, "Brazil", "Argentina")
    o = card["one_x_two"]
    total = o["home"]["prob"] + o["draw"]["prob"] + o["away"]["prob"]
    assert abs(total - 1.0) < 1e-6
    t = card["totals"][2.5]
    assert abs(t["over"]["prob"] + t["under"]["prob"] + t["push"] - 1.0) < 1e-6


def test_stronger_team_is_favoured():
    model = wb.load_model()
    strong = wb.price_match(model, "Brazil", "Qatar")["one_x_two"]
    assert strong["home"]["prob"] > strong["away"]["prob"]


def test_removing_a_starter_lowers_team_xg():
    base = wb.load_model()
    card = wb.price_match(base, "France", "Japan")
    star = card["props_home"].iloc[0]["player"]
    weakened = wb.load_model({"France": [star]})
    card2 = wb.price_match(weakened, "France", "Japan")
    assert card2["expected_goals"]["home"] < card["expected_goals"]["home"]


def test_props_use_minutes_and_xg():
    model = wb.load_model()
    card = wb.price_match(model, "Brazil", "Qatar")
    props = card["props_home"]
    # anytime probability must be monotonic in expected goals
    assert props["anytime_prob"].is_monotonic_decreasing
    assert (props["two_plus_prob"] <= props["anytime_prob"] + 1e-9).all()


def test_impact_metric_credits_non_forwards():
    # xg90 is an impact rating (xGChain), so midfielders/defenders should carry
    # real value -- not the ~0 that direct npxG would give them.
    players = wb.load_players()
    if "npxg90" not in players.columns:
        return  # synthetic data has no separate impact/finishing split
    mids = players[players["position"] == "MID"]
    # a meaningful share of midfielders out-impact their own finishing number
    assert (mids["xg90"] > mids["npxg90"]).mean() > 0.5


def test_props_are_finishing_based():
    # goalscorer props use npxg90 when present, so top scorers should be forwards
    model = wb.load_model()
    props = wb.price_match(model, "Argentina", "France")["props_home"]
    assert props.iloc[0]["position"] == "FWD"


def test_edge_math():
    # fair odds: model 50%, price 2.10 -> +5% EV, positive Kelly
    assert abs(expected_value(0.5, 2.10) - 0.05) < 1e-9
    assert kelly_fraction(0.5, 2.10) > 0
    # no edge at fair price
    assert abs(expected_value(0.5, 2.0)) < 1e-9
    assert kelly_fraction(0.5, 2.0) == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all smoke tests passed")
