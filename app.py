"""World Cup betting model — Streamlit frontend (phone-friendly).

Run locally:   streamlit run app.py
Deploy free:   push to GitHub, then create an app at https://share.streamlit.io
               pointing at this repo / app.py. The URL works on mobile.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import worldcup_betting as wb
from worldcup_betting.edges import expected_value, find_edges, implied_prob, kelly_fraction

st.set_page_config(page_title="World Cup Edge Finder", page_icon="⚽", layout="centered")

st.title("⚽ World Cup Edge Finder")
st.caption(
    "Player xG ratings weighted by expected minutes → team strength → "
    "match & player markets → edges vs. the book."
)


@st.cache_data
def _team_list() -> list[str]:
    return wb.load_model().teams


teams = _team_list()

# ----- fixture controls -------------------------------------------------------
c1, c2 = st.columns(2)
home = c1.selectbox("Home / Team A", teams, index=teams.index("Brazil"))
away = c2.selectbox("Away / Team B", teams, index=teams.index("Argentina"))

with st.expander("Match settings"):
    neutral = st.checkbox("Neutral venue (typical for World Cup)", value=True)
    home_adv = 1.0 if neutral else st.slider("Home advantage", 1.0, 1.25, 1.10, 0.01)
    total_line = st.selectbox("Totals line", [1.5, 2.5, 3.5], index=1)
    hcap_line = st.selectbox("Asian handicap (on Team A)",
                             [-2.5, -1.5, -0.5, 0.5, 1.5], index=2)

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

# ----- availability (uses the player-minutes model) ---------------------------
full = wb.load_model()
with st.expander("Rule players out (injury / suspension / rotation)"):
    ca, cb = st.columns(2)
    home_players = full.players[full.players.team == home].sort_values(
        "exp_minutes", ascending=False)["player"].tolist()
    away_players = full.players[full.players.team == away].sort_values(
        "exp_minutes", ascending=False)["player"].tolist()
    out_home = ca.multiselect(f"{home} out", home_players)
    out_away = cb.multiselect(f"{away} out", away_players)

unavailable = {home: out_home, away: out_away}
model = wb.load_model(unavailable if (out_home or out_away) else None)
card = wb.price_match(model, home, away, home_adv=home_adv,
                      total_lines=(total_line,), handicap_line=hcap_line)

eg = card["expected_goals"]
m1, m2 = st.columns(2)
m1.metric(f"{home} expected goals", f"{eg['home']:.2f}")
m2.metric(f"{away} expected goals", f"{eg['away']:.2f}")

tab_markets, tab_props, tab_edges = st.tabs(["Match markets", "Player props", "Edge finder"])

# ----- match markets ----------------------------------------------------------
with tab_markets:
    o = card["one_x_two"]
    st.subheader("Match result (1X2)")
    st.dataframe(pd.DataFrame({
        "selection": [home, "Draw", away],
        "model prob": [o["home"]["prob"], o["draw"]["prob"], o["away"]["prob"]],
        "fair odds": [o["home"]["fair_odds"], o["draw"]["fair_odds"], o["away"]["fair_odds"]],
    }).style.format({"model prob": "{:.1%}", "fair odds": "{:.2f}"}),
        hide_index=True, use_container_width=True)

    t = card["totals"][total_line]
    b = card["btts"]
    hc = card["handicap"]
    st.subheader("Goals & handicap")
    st.dataframe(pd.DataFrame({
        "selection": [f"Over {total_line}", f"Under {total_line}", "BTTS yes", "BTTS no",
                      f"{home} {hcap_line:+g}", f"{away} {-hcap_line:+g}"],
        "model prob": [t["over"]["prob"], t["under"]["prob"], b["yes"]["prob"],
                       b["no"]["prob"], hc["home"]["prob"], hc["away"]["prob"]],
        "fair odds": [t["over"]["fair_odds"], t["under"]["fair_odds"], b["yes"]["fair_odds"],
                      b["no"]["fair_odds"], hc["home"]["fair_odds"], hc["away"]["fair_odds"]],
    }).style.format({"model prob": "{:.1%}", "fair odds": "{:.2f}"}),
        hide_index=True, use_container_width=True)

    st.subheader("Most likely scores")
    cs = card["correct_score"].copy()
    st.dataframe(cs.style.format({"prob": "{:.1%}", "fair_odds": "{:.2f}"}),
                 hide_index=True, use_container_width=True)

# ----- player props -----------------------------------------------------------
with tab_props:
    st.caption("Anytime & 2+ goalscorer, driven by each player's xG share × expected minutes.")
    for label, key in ((home, "props_home"), (away, "props_away")):
        st.subheader(label)
        p = card[key].head(8)[
            ["player", "position", "exp_minutes", "anytime_prob",
             "anytime_fair_odds", "two_plus_prob", "two_plus_fair_odds"]
        ].rename(columns={
            "exp_minutes": "exp min", "anytime_prob": "anytime",
            "anytime_fair_odds": "fair", "two_plus_prob": "2+",
            "two_plus_fair_odds": "fair (2+)"})
        st.dataframe(p.style.format({
            "exp min": "{:.0f}", "anytime": "{:.1%}", "fair": "{:.2f}",
            "2+": "{:.1%}", "fair (2+)": "{:.2f}"}),
            hide_index=True, use_container_width=True)

# ----- edge finder ------------------------------------------------------------
with tab_edges:
    st.subheader("Single bet check")
    cc1, cc2 = st.columns(2)
    mp = cc1.number_input("Model probability (%)", 1.0, 99.0, 50.0, 0.5) / 100.0
    mo = cc2.number_input("Bookmaker decimal odds", 1.01, 50.0, 2.10, 0.01)
    ev = expected_value(mp, mo)
    k = kelly_fraction(mp, mo) * 0.25
    cc1.metric("Edge (EV)", f"{ev:+.1%}", delta="bet" if ev > 0 else "pass")
    cc2.metric("¼-Kelly stake", f"{k:.1%}")
    st.caption(f"Book implies {implied_prob(mo):.1%}; you estimate {mp:.1%}.")

    st.divider()
    st.subheader("Scan a slate of odds")
    st.caption("Upload odds (columns: fixture, market, selection, market_odds) "
               "or use the bundled sample.")
    up = st.file_uploader("Odds CSV", type="csv")
    min_ev = st.slider("Minimum EV to flag", 0.0, 0.15, 0.02, 0.01)

    from worldcup_betting.cli import ODDS_PATH, _selection_probs
    odds = pd.read_csv(up) if up is not None else pd.read_csv(ODDS_PATH)

    rows = []
    for fixture, grp in odds.groupby("fixture"):
        try:
            h, a = [s.strip() for s in str(fixture).split(" vs ")]
            fcard = wb.price_match(model, h, a)
        except Exception:
            continue
        probs = _selection_probs(fcard, h, a)
        for _, r in grp.iterrows():
            mpx = probs.get((r["market"], r["selection"]))
            if mpx is None:
                continue
            rows.append({"market": f"{fixture} | {r['market']}",
                         "selection": r["selection"], "model_prob": mpx,
                         "market_odds": r["market_odds"]})

    found = find_edges(rows, min_ev=min_ev, kelly_multiplier=0.25)
    if found.empty:
        st.info(f"No edges ≥ {min_ev:.0%} EV across {len(rows)} priced selections.")
    else:
        show = found.rename(columns={"model_prob": "model", "market_implied": "implied",
                                     "market_odds": "odds", "kelly": "¼-Kelly"})
        st.dataframe(show[["selection", "market", "odds", "model", "implied", "ev", "¼-Kelly"]]
                     .style.format({"odds": "{:.2f}", "model": "{:.1%}", "implied": "{:.1%}",
                                    "ev": "{:+.1%}", "¼-Kelly": "{:.1%}"}),
                     hide_index=True, use_container_width=True)

st.caption("Model uses synthetic tier-calibrated ratings. Swap in real data by "
           "replacing worldcup_betting/data/players.csv. For research/education — bet responsibly.")
