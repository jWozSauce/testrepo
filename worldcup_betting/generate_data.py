"""Generate a realistic, offline player-rating dataset for World Cup squads.

Every player carries the three quantities the model is built on:

    xg90   - attacking output per 90 minutes (expected goals + expected assists)
    def90  - defensive rating: goals prevented per 90 vs. an average player
    exp_minutes - expected minutes in a given match (starter vs. rotation vs. bench)

`exp_minutes` is normalised so each squad sums to 990 (11 players x 90 min),
which keeps the minutes-weighted team aggregates physically consistent: a
team's expected goals in a match equals the sum of its on-pitch players'
expected goals.

The numbers here are synthetic but tier-calibrated so that strong sides beat
weak ones by believable margins. Swap this file out for real data (FBref,
Opta, a bespoke rating model) by producing a CSV with the same columns.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Rough attacking / defensive tiers for a 32-team field. The two scalars scale
# a team's attacking output and defensive solidity around the global average.
# (attack_base, defense_base): >1 = better than average.
TEAMS = {
    "Argentina": (1.30, 1.28), "France": (1.34, 1.22), "Brazil": (1.32, 1.20),
    "England": (1.22, 1.24), "Spain": (1.26, 1.18), "Portugal": (1.24, 1.12),
    "Netherlands": (1.18, 1.20), "Germany": (1.20, 1.10), "Belgium": (1.16, 1.06),
    "Italy": (1.10, 1.22), "Croatia": (1.08, 1.10), "Uruguay": (1.10, 1.08),
    "Morocco": (1.02, 1.18), "Colombia": (1.08, 1.04), "Mexico": (1.00, 1.02),
    "USA": (1.00, 1.00), "Senegal": (1.04, 1.02), "Japan": (1.02, 1.04),
    "Switzerland": (0.98, 1.08), "Denmark": (1.02, 1.06), "Korea Republic": (0.98, 0.96),
    "Serbia": (1.04, 0.92), "Poland": (0.98, 0.96), "Ecuador": (0.96, 1.00),
    "Australia": (0.90, 0.98), "Nigeria": (1.00, 0.94), "Canada": (0.96, 0.92),
    "Cameroon": (0.94, 0.90), "Ghana": (0.94, 0.88), "Saudi Arabia": (0.84, 0.90),
    "Qatar": (0.82, 0.86), "Iran": (0.90, 0.98),
}

# squad composition (sums to 26)
SQUAD = {"GK": 3, "DEF": 8, "MID": 8, "FWD": 7}
# starting XI by position (sums to 11)
STARTERS = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}

# baseline per-90 attacking output by position (xG + xA)
ATT_BASE = {"GK": 0.01, "DEF": 0.08, "MID": 0.26, "FWD": 0.48}
# baseline per-90 defensive rating (goals prevented vs. average outfielder)
DEF_BASE = {"GK": 0.55, "DEF": 0.34, "MID": 0.12, "FWD": 0.02}

FIRST = ["Alex", "Diego", "Luka", "Marco", "Yuki", "Omar", "Leon", "Karim",
         "Bruno", "Ivan", "Noah", "Sami", "Theo", "Hugo", "Mateo", "Kai",
         "Dani", "Pavel", "Jude", "Cole", "Eden", "Niko", "Raul", "Tariq",
         "Felix", "Joao", "Lars", "Ali", "Ze", "Tom"]
LAST = ["Silva", "Müller", "Kovac", "Rossi", "Tanaka", "Haddad", "Berg",
        "Diallo", "Costa", "Petrov", "Vega", "Nadir", "Laurent", "Park",
        "Olsen", "Mensah", "Reyes", "Novak", "Bauer", "Fofana", "Suzuki",
        "Ferrari", "Andersen", "Owusu", "Marin", "Sokol", "Bello", "Kemal"]


def _player_name(rng: np.random.Generator, used: set) -> str:
    for _ in range(200):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if name not in used:
            used.add(name)
            return name
    # fall back to a numbered name if we somehow exhaust combinations
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)} {len(used)}"
    used.add(name)
    return name


def generate(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    used_names: set = set()
    rows = []

    for team, (att_base, def_base) in TEAMS.items():
        for pos, count in SQUAD.items():
            # quality rank within the position group: index 0 is the best player
            quality = np.sort(rng.normal(0, 1, count))[::-1]
            n_start = STARTERS[pos]
            for i in range(count):
                q = quality[i]
                is_starter = i < n_start

                xg90 = ATT_BASE[pos] * att_base * (1 + 0.35 * q)
                xg90 = max(0.0, xg90 + rng.normal(0, 0.02))

                def90 = DEF_BASE[pos] * def_base * (1 + 0.30 * q)
                def90 = def90 + rng.normal(0, 0.03)

                # expected minutes: starters carry the load, depth players rotate in
                if is_starter:
                    mins = float(np.clip(rng.normal(80, 6), 55, 90))
                elif i < n_start + 2:  # primary rotation options
                    mins = float(np.clip(rng.normal(16, 7), 0, 40))
                else:  # deep bench
                    mins = float(np.clip(rng.normal(4, 4), 0, 20))

                rows.append({
                    "team": team,
                    "player": _player_name(rng, used_names),
                    "position": pos,
                    "xg90": round(xg90, 3),
                    "def90": round(def90, 3),
                    "exp_minutes_raw": mins,
                    "is_nominal_starter": is_starter,
                })

    df = pd.DataFrame(rows)

    # normalise expected minutes so every squad sums to exactly 990 (11 * 90)
    team_total = df.groupby("team")["exp_minutes_raw"].transform("sum")
    df["exp_minutes"] = (df["exp_minutes_raw"] * 990.0 / team_total).round(1)
    df = df.drop(columns=["exp_minutes_raw"])
    return df.reset_index(drop=True)


def main() -> None:
    here = os.path.dirname(__file__)
    out = os.path.join(here, "data", "players.csv")
    df = generate()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} players across {df['team'].nunique()} teams -> {out}")


if __name__ == "__main__":
    main()
