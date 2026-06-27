# Drop your FBref CSV exports here

Export from FBref in your browser (no scraping needed):

1. Open a stats table, e.g. **2025-2026 Big 5 European Leagues → Player Standard Stats**
   (`https://fbref.com/en/comps/Big5/2025-2026/stats/players/`).
2. Above the table: **Share & Export → Get table as CSV (for Excel)**.
3. Select the CSV text it shows, copy it into a file, and save it in this folder.

### Files

| File (suggested name) | FBref table | Used for |
|---|---|---|
| `*standard*.csv` | Player **Standard Stats** | finishing `npxg90` + fallback `xg90` — **required** |
| `*creation*.csv` | Player **Goal & Shot Creation** | impact `xg90` from SCA90 (credits buildup) — recommended |
| `*defense*.csv`  | Player **Defensive Actions** | real `def90` (Tkl/Int/Blocks/Clr) — optional |

Without the creation export, `xg90` falls back to direct npxG+xAG, which
under-credits midfielders and defenders. The Goal & Shot Creation table's SCA90
is FBref's closest impact metric (it credits the buildup to a shot, not just the
finish), so include it for an impact-based rating.

You can drop multiple `*standard*.csv` files (e.g. one per league) to cover
players outside the Big 5. The loader auto-discovers files matching the name
patterns above.

### Build `players.csv`

```bash
python -m worldcup_betting.data_sources.fbref_csv
# or be explicit:
python -m worldcup_betting.data_sources.fbref_csv \
    --standard worldcup_betting/data/fbref/big5_standard.csv \
    --defense  worldcup_betting/data/fbref/big5_defense.csv
```

National squads are formed by grouping on the FBref **Nation** column and
keeping the most-used players (club minutes proxy who starts). For official
26-man squads instead, pass `--roster roster.csv` (columns: `team, player`).
