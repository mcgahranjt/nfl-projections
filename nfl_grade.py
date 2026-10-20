"""
NFL model grading

Checks the saved weekly projections against what happened. Only projections saved before kickoff
count: for each game, the latest projection made on or before game day.

Games (output/game_projection_history.csv vs final scores):
  - spread error compared with "home team by the home-field edge" and with the closing market line
  - straight-up winner %, total error, and the market's total error on the same games
Fantasy (output/player_projection_history.csv vs box scores, your league's scoring):
  - average miss by position compared with the player's season average so far
  - range check: about 10% of results should land below P10 and 10% above P90

The market lines come from nflverse's schedule and are used only here, as a benchmark.

Usage: python3 nfl_grade.py
Outputs (output/): grades_games.csv, grades_fantasy.csv, grades_summary.json (read by the website)
"""

import json
import os

import numpy as np
import pandas as pd

import nfl_data as N
import nfl_players as P
import nfl_ratings as R

OUT_DIR = "output"


def grade_games(games):
    path = os.path.join(OUT_DIR, "game_projection_history.csv")
    if not os.path.exists(path):
        return None, pd.DataFrame()
    h = pd.read_csv(path)
    h = h[h.as_of <= h.gameday].sort_values("as_of").groupby("game_id").tail(1)
    g = games[games.played][["game_id", "home_score", "away_score", "spread_line", "total_line", "neutral"]]
    m = h.merge(g, on="game_id")
    if m.empty:
        return None, m
    margin, total = m.home_score - m.away_score, m.home_score + m.away_score
    hist = games[games.played & (games.season.between(N.CURRENT_SEASON - 5, N.CURRENT_SEASON - 1)) & (games.neutral == 0)]
    edge = (hist.home_score - hist.away_score).mean()
    m["margin"], m["total"] = margin, total
    s = {"games": int(len(m)),
         "spread_MAE": float((m.proj_margin - margin).abs().mean()),
         "home_edge_MAE": float(((1 - m.neutral) * edge - margin).abs().mean()),
         "market_spread_MAE": float((m.spread_line - margin).abs().mean()),
         "winner_pct": float(100 * ((m.proj_margin > 0) == (margin > 0))[margin != 0].mean()),
         "total_MAE": float((m.proj_total - total).abs().mean()),
         "market_total_MAE": float((m.total_line - total).abs().mean())}
    return s, m


def grade_fantasy():
    path = os.path.join(OUT_DIR, "player_projection_history.csv")
    if not os.path.exists(path):
        return [], pd.DataFrame()
    proj = pd.read_csv(path)
    pw = N.player_weekly(range(N.CURRENT_SEASON - 1, N.CURRENT_SEASON + 1))
    pw = pw[pw.season_type.isin(["REG", "POST"])].copy()
    pw["team"] = pw.team.replace(R.FRANCHISE)
    num = [c for c in pw.columns if pw[c].dtype.kind in "if"]
    pw[num] = pw[num].fillna(0)
    pw["fumbles_lost"] = pw.sack_fumbles_lost + pw.rushing_fumbles_lost + pw.receiving_fumbles_lost
    pw["twopt"] = pw.passing_2pt_conversions + pw.rushing_2pt_conversions + pw.receiving_2pt_conversions
    pw["actual"] = P.actual_fantasy(pw)
    m = proj.merge(pw[["player_id", "season", "week", "actual"]], on=["player_id", "season", "week"])
    if m.empty:
        return [], m
    # baseline: season average in earlier games (2+), else last season's average
    pw = pw.sort_values(["season", "week"])
    pw["prior"] = pw.groupby(["player_id", "season"]).actual.transform(lambda x: x.shift().expanding().mean())
    pw["n"] = pw.groupby(["player_id", "season"]).cumcount()
    last = pw[pw.season == N.CURRENT_SEASON - 1].groupby("player_id").actual.mean()
    pw["baseline"] = np.where(pw.n >= 2, pw.prior, pw.player_id.map(last))
    m = m.merge(pw[["player_id", "season", "week", "baseline"]], on=["player_id", "season", "week"], how="left")
    rows = []
    for pos in P.POSITIONS + ["All"]:
        x = m if pos == "All" else m[m.position == pos]
        b = x.baseline.notna()
        rows.append({"position": pos, "players": int(len(x)),
                     "MAE": float((x.fpts - x.actual).abs().mean()),
                     "season_avg_MAE": float((x.baseline - x.actual)[b].abs().mean()) if b.any() else None,
                     "model_MAE_same": float((x.fpts - x.actual)[b].abs().mean()) if b.any() else None,
                     "below_P10": float(100 * (x.actual < x.fpts_p10).mean()),
                     "above_P90": float(100 * (x.actual > x.fpts_p90).mean())})
    return rows, m


def backtest_reference():
    """Accuracy on past seasons (from the backtests), shown next to this season's live results."""
    ref = {}
    s = os.path.join(OUT_DIR, "backtest_by_season.csv")
    if os.path.exists(s):
        b = pd.read_csv(s)
        b = b[b.season.between(2016, 2025)]
        w = b.games
        ref["spread"] = {"seasons": "2016-2025", "model": float((b.with_QB * w).sum() / w.sum()),
                         "market": float((b.market * w).sum() / w.sum())}
    c = os.path.join(OUT_DIR, "player_calibration_report.csv")
    if os.path.exists(c):
        r = pd.read_csv(c)
        r = r[r.seasons.str.startswith("2020-2025")]
        ref["fantasy"] = [{"position": x.pos, "model": float(x.MAE_blend), "season_avg": float(x.MAE_season_avg)}
                          for x in r.itertuples()]
    return ref


def main():
    games = R.load_games()
    gs, gm = grade_games(games)
    fs, fm = grade_fantasy()
    os.makedirs(OUT_DIR, exist_ok=True)
    if len(gm):
        gm.to_csv(os.path.join(OUT_DIR, "grades_games.csv"), index=False)
    if len(fm):
        fm.to_csv(os.path.join(OUT_DIR, "grades_fantasy.csv"), index=False)
    summary = {"season": N.CURRENT_SEASON, "games": gs, "fantasy": fs, "reference": backtest_reference(),
               "first_week_graded": int(gm.week.min()) if len(gm) else None}
    with open(os.path.join(OUT_DIR, "grades_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if gs:
        print(f"Games graded: {gs['games']}. Spread MAE {gs['spread_MAE']:.2f} (market {gs['market_spread_MAE']:.2f}, "
              f"home edge only {gs['home_edge_MAE']:.2f}); winners {gs['winner_pct']:.0f}%; "
              f"total MAE {gs['total_MAE']:.2f} (market {gs['market_total_MAE']:.2f}).")
    else:
        print("No completed games yet for the saved projections (grading starts after week 7).")
    if fs:
        print("\nFantasy (MAE in points; season average on the same player-games):")
        print(pd.DataFrame(fs).round(2).to_string(index=False))
    else:
        print("No completed games yet for the saved fantasy projections.")


if __name__ == "__main__":
    main()
