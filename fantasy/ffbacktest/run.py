"""CLI: walk-forward backtest of per-position half-PPR projections.

Examples
--------
  # Backtest the last several seasons (NFL features only where Madden is absent):
  python -m ffbacktest.run backtest --start 2018 --end 2024

  # Restrict evaluation to specific seasons and positions:
  python -m ffbacktest.run backtest --start 2016 --end 2024 --eval 2023 2024 --pos WR TE

  # Measure Madden lift on seasons that have a launch file:
  python -m ffbacktest.run ablation --start 2018 --end 2024 --eval 2023
"""
from __future__ import annotations

import argparse
import warnings
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

from . import backtest as B
from .sources.madden import available_seasons


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:8.3f}")


def cmd_backtest(a):
    eval_seasons = a.eval or list(range(a.start, a.end + 1))
    per, preds = B.walk_forward(a.start, a.end, eval_seasons, tuple(a.pos),
                                use_madden=not a.no_madden)
    if per.empty:
        print("No evaluable (position, season) cells. Widen --start/--end.")
        return
    print(f"\nMadden launch files present for seasons: {available_seasons() or 'none'}")
    print("\n=== per season ===")
    print(_fmt(per.sort_values(["pos", "season"])))
    print("\n=== pooled across seasons ===")
    print(_fmt(B.pooled(per)))
    if a.show_top and not preds.empty:
        yr = max(eval_seasons)
        print(f"\n=== {yr} top-15 WR by projection ===")
        wr = preds[(preds.position == "WR") & (preds.season == yr)]
        print(_fmt(wr.sort_values("pred", ascending=False)
                   .head(15)[["player_name", "pred", "y"]]))


def cmd_ablation(a):
    eval_seasons = a.eval or [a.end]
    with_m, _ = B.walk_forward(a.start, a.end, eval_seasons, tuple(a.pos), use_madden=True)
    no_m, _ = B.walk_forward(a.start, a.end, eval_seasons, tuple(a.pos), use_madden=False)
    if with_m.empty:
        print("No evaluable cells.")
        return
    pw, pn = B.pooled(with_m), B.pooled(no_m)
    merged = pw.merge(pn, on="pos", suffixes=("_madden", "_base"))
    merged["mae_lift"] = merged["mae_base"] - merged["mae_madden"]
    merged["spearman_lift"] = merged["spearman_madden"] - merged["spearman_base"]
    print(f"\nMadden ablation on seasons {eval_seasons} "
          f"(launch files: {available_seasons()})")
    print(_fmt(merged[["pos", "mae_base", "mae_madden", "mae_lift",
                       "spearman_base", "spearman_madden", "spearman_lift"]]))
    print("\nmae_lift > 0  and  spearman_lift > 0  mean Madden helped.")


def cmd_project(a):
    res = B.project_season(a.target, a.start, tuple(a.pos))
    if res.empty:
        print("No projections produced.")
        return
    print(f"\n{a.target} half-PPR projections (Madden {a.target - 1999} launch ratings; "
          f"trained on {a.start}-{a.target - 1})")
    for pos in a.pos:
        sub = res[res.position == pos].head(a.top)
        if sub.empty:
            continue
        print(f"\n=== {pos} — top {a.top} ===")
        show = sub[["pos_rank", "player_name", "team", "age", "madden_overall", "proj_half_ppr"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


def cmd_cheatsheet(a):
    from . import projections as P
    res = P.project(a.target, a.start)
    adp = None
    if a.adp_live:
        adp = P.fetch_live_adp(a.target, save_path=a.adp_snapshot)
    elif a.adp:
        adp = a.adp
    path = P.export_excel(res, a.target, a.out, adp=adp)
    n = sum(len(v) for v in res.values())
    extra = " + Values/Reaches (ADP)" if adp is not None else ""
    print(f"Wrote {a.target} cheat sheet ({n} players, tabs: Draft Board + All Players + "
          f"{', '.join(P.POS)}{extra}) -> {path}")


def main():
    p = argparse.ArgumentParser(prog="ffbacktest")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("backtest", "ablation"):
        s = sub.add_parser(name)
        s.add_argument("--start", type=int, default=2018, help="first labelled season")
        s.add_argument("--end", type=int, default=2024, help="last labelled season")
        s.add_argument("--eval", type=int, nargs="*", help="seasons to evaluate")
        s.add_argument("--pos", nargs="*", default=["QB", "RB", "WR", "TE"])
        s.add_argument("--no-madden", action="store_true")
        s.add_argument("--show-top", action="store_true")
    sp = sub.add_parser("project")
    sp.add_argument("--target", type=int, default=2026, help="season to project")
    sp.add_argument("--start", type=int, default=2015, help="first training season")
    sp.add_argument("--pos", nargs="*", default=["QB", "RB", "WR", "TE"])
    sp.add_argument("--top", type=int, default=20)
    sc = sub.add_parser("cheatsheet")
    sc.add_argument("--target", type=int, default=2026, help="season to project")
    sc.add_argument("--start", type=int, default=2015, help="first training season")
    sc.add_argument("--out", default="exports/cheatsheet_2026.xlsx", help="output .xlsx path")
    sc.add_argument("--adp", default=None, help="path to an ADP .xlsx to compare against")
    sc.add_argument("--adp-live", action="store_true",
                    help="pull current ADP from Fantasy Football Calculator instead of a file")
    sc.add_argument("--adp-snapshot", default="data/adp/live_adp_2026.csv",
                    help="where to save the fetched live-ADP snapshot")
    args = p.parse_args()
    {"backtest": cmd_backtest, "ablation": cmd_ablation,
     "project": cmd_project, "cheatsheet": cmd_cheatsheet}[args.cmd](args)


if __name__ == "__main__":
    main()
