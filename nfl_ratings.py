"""
NFL team ratings, spreads and totals

Each team gets an offense rating and a defense rating, both opponent-adjusted, from two measures:
  - points scored per game
  - offensive EPA per game (expected points added on pass and run plays), which is steadier than points
Ratings are ridge regressions anchored to a preseason starting point: last season's final ratings,
pulled toward average by PRESEASON_CARRY. The preseason starting point counts as PRIOR_GAMES games
of evidence, so this season's results take over gradually.

Quarterback adjustment: a team's rating reflects the quarterbacks who played in the games behind it.
When a different quarterback starts, the projection shifts by
  (starter's value - value of the QBs behind the team's rating) x DROPBACKS_PER_GAME
where a QB's value is his EPA per dropback (recent seasons count more), pulled toward backup level
when he has few dropbacks.

A game's projection comes from a small regression on those inputs:
  home margin = home field + b1 x points diff + b2 x EPA diff + b3 x rest edge + b4 x QB adjustment diff
  total       = 2 x recent league scoring + c1 x points sum + c2 x EPA sum + c3 x QB adjustment sum
The weights are fit on earlier seasons only (the last BETA_SEASONS), using projections that were
themselves made only from games before each one.

Starters: the schedule's listed starter for each team. data/qb_overrides.csv (columns: team, qb_name)
replaces it by hand, e.g. when a starter is ruled out after the schedule was updated.

Usage:
  python3 nfl_ratings.py             # this week's spreads and totals, plus team ratings
  python3 nfl_ratings.py --backtest  # tune on 2006-2015, test on 2016-2025, check 2026; with and without QBs

Outputs (output/): team_ratings.csv, game_projections.csv, game_projection_history.csv,
and with --backtest: backtest_tuning.csv, backtest_qb_tuning.csv, backtest_by_season.csv
"""

import datetime as dt
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import nfl_data as N

OUT_DIR = "output"
FIRST_SEASON = 2006                 # 2005 supplies the first preseason starting point
TUNE_SEASONS = range(2006, 2016)
TEST_SEASONS = range(2016, 2026)
BETA_SEASONS = 8                    # regression weights come from the previous 8 seasons
PRIOR_GAMES_GRID = [2, 4, 6, 8, 12]
CARRY_GRID = [0.3, 0.45, 0.6, 0.75]
QB_PRIOR_GRID = [150, 300, 600]     # dropbacks of backup-level play blended into each QB's record
QB_BACKUP_GRID = [0.05, 0.10, 0.15] # backup level = league EPA per dropback minus this
FINAL_PRIOR_GAMES = 1               # light anchor when computing a finished season's ratings
QB_HALF_LIFE_DAYS = 365             # a dropback from a year ago counts half
QB_BASE_DAYS = 300                  # games behind a team's rating: roughly this season plus last
DROPBACKS_PER_GAME = 35
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# chosen by the backtest (python3 nfl_ratings.py --backtest)
PRIOR_GAMES = 6
PRESEASON_CARRY = 0.6
QB_PRIOR_DROPBACKS = 600
QB_BACKUP_GAP = 0.15
MODEL = {"measures": ("pts", "epa"),      # what the ratings are built from
         "half_life": None,               # days; recent games count more within a season (None = equal)
         "pg_off": PRIOR_GAMES, "pg_def": PRIOR_GAMES,
         "carry_off": PRESEASON_CARRY, "carry_def": PRESEASON_CARRY}

# per-game team measures available from play-by-play (offense; the defense side is the opponent's)
PBP_MEASURES = ["epa", "epa_live", "pass_epa", "rush_epa", "pass_epa_live", "rush_epa_live",
                "success", "success_live"]
ALL_MEASURES = ["pts"] + PBP_MEASURES


# ---------------------------------------------------------------- data
def load_games():
    s = N.schedule()
    s = s[s.season >= FIRST_SEASON - 1].copy()
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(FRANCHISE)
    tg = N.team_games(range(FIRST_SEASON - 1, N.CURRENT_SEASON + 1))
    tg["team"] = tg.team.replace(FRANCHISE)
    tg = tg.set_index(["game_id", "team"])
    for m in PBP_MEASURES:
        col = tg[m]
        s[f"home_{m}"] = [col.get((g, t), np.nan) for g, t in zip(s.game_id, s.home_team)]
        s[f"away_{m}"] = [col.get((g, t), np.nan) for g, t in zip(s.game_id, s.away_team)]
    s["home_pts"], s["away_pts"] = s.home_score, s.away_score
    s["neutral"] = (s.location == "Neutral").astype(int)
    s["home_field"] = 1 - s.neutral
    s["const"] = 1.0
    s["rest_edge"] = (s.home_rest - s.away_rest).clip(-7, 7).fillna(0)
    s["played"] = s.home_score.notna() & s.home_epa.notna()
    return s.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def load_qbs(games):
    q = N.qb_games(range(FIRST_SEASON - 1, N.CURRENT_SEASON + 1))
    q["team"] = q.team.replace(FRANCHISE)
    q = q.merge(games[["game_id", "gameday"]], on="game_id")
    return q


# ------------------------------------------------------------- ratings
def fit(games, teams, measure, prior_off, prior_def, pg_off, pg_def=None, weights=None):
    """
    Ridge fit of one measure ("pts" or "epa"):  team's measure - league = off[team] + def[opp] + home
    def is what the defense allows above average (positive = worse defense).
    """
    n = len(teams)
    if games.empty:
        return prior_off.copy(), prior_def.copy()
    idx = {t: i for i, t in enumerate(teams)}
    h = games.home_team.map(idx).to_numpy()
    a = games.away_team.map(idx).to_numpy()
    hv = games[f"home_{measure}"].to_numpy(float)
    av = games[f"away_{measure}"].to_numpy(float)
    lg = np.concatenate([hv, av]).mean()
    home = 0.5 * (1 - games.neutral.to_numpy())
    m = len(games)
    X = np.zeros((2 * m, 2 * n + 1))
    X[np.arange(m), h] = 1
    X[np.arange(m), n + a] = 1
    X[np.arange(m), 2 * n] = home
    X[m + np.arange(m), a] = 1
    X[m + np.arange(m), n + h] = 1
    X[m + np.arange(m), 2 * n] = -home
    prior = np.concatenate([prior_off, prior_def, [0.0]])
    y = np.concatenate([hv, av]) - lg - X @ prior
    pen = np.concatenate([np.full(n, float(pg_off)), np.full(n, float(pg_off if pg_def is None else pg_def)), [1e-3]])
    w = np.ones(2 * m) if weights is None else np.tile(weights, 2)   # home field is estimated freely
    dev = np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(pen), X.T @ (w * y))
    est = prior + dev
    off, dfn = est[:n], est[n:2 * n]
    return off - off.mean(), dfn - dfn.mean()


def season_finals(games, teams, seasons):
    """Each finished season's ratings (lightly anchored to zero), for next season's starting point."""
    out = {}
    z = np.zeros(len(teams))
    for s in seasons:
        g = games[(games.season == s) & games.played]
        out[s] = {m: fit(g, teams, m, z, z, FINAL_PRIOR_GAMES) for m in ALL_MEASURES}
    return out


def ratings_before(games, teams, finals, season, before, cfg=None):
    """Ratings using this season's games before a date, anchored to last season's carried-over ratings."""
    cfg = cfg or MODEL
    g = games[(games.season == season) & games.played & (games.gameday < before)]
    w = None
    if cfg["half_life"] and len(g):
        w = 0.5 ** ((pd.Timestamp(before) - g.gameday).dt.days.to_numpy() / cfg["half_life"])
    r = {}
    for m in cfg["measures"]:
        po, pd_ = finals[season - 1][m]
        r[m] = fit(g, teams, m, cfg["carry_off"] * po, cfg["carry_def"] * pd_, cfg["pg_off"], cfg["pg_def"], w)
    return r


def features(games, r, teams):
    """Ratings differences and sums for each game (home perspective)."""
    idx = {t: i for i, t in enumerate(teams)}
    h = games.home_team.map(idx).to_numpy()
    a = games.away_team.map(idx).to_numpy()
    f = pd.DataFrame(index=games.index)
    for m in r:
        off, dfn = r[m]
        hs, as_ = off[h] + dfn[a], off[a] + dfn[h]
        f[f"diff_{m}"], f[f"sum_{m}"] = hs - as_, hs + as_
    return f


def week_start(day):
    day = pd.Timestamp(day)
    return day - pd.Timedelta(days=(day.weekday() - 1) % 7)   # the Tuesday before


def walk_forward(games, teams, finals, seasons, cfg=None):
    """Projection features for every game, each made only from games before its week."""
    rows = []
    for s in seasons:
        sg = games[games.season == s]
        for day in sorted(sg.gameday.unique()):
            cutoff = week_start(day)
            todo = sg[sg.gameday == day]
            r = ratings_before(games, teams, finals, s, cutoff, cfg)
            f = features(todo, r, teams)
            f["cutoff"] = cutoff
            rows.append(f)
    return games.join(pd.concat(rows), how="inner")


# ---------------------------------------------------------- quarterbacks
def qb_values(qbs, cutoff, prior_db, backup_gap):
    """Each QB's EPA per dropback relative to league average, as of a date. Returns (values, backup level)."""
    q = qbs[qbs.gameday < cutoff]
    if q.empty:
        return pd.Series(dtype=float), -backup_gap
    w = 0.5 ** ((cutoff - q.gameday).dt.days / QB_HALF_LIFE_DAYS)
    recent = q[q.gameday >= cutoff - pd.Timedelta(days=365)]
    lg = recent.qb_epa.sum() / max(recent.dropbacks.sum(), 1)
    agg = pd.DataFrame({"db": q.dropbacks * w, "epa": q.qb_epa * w, "qb_id": q.qb_id}).groupby("qb_id").sum()
    val = (agg.epa + prior_db * (lg - backup_gap)) / (agg.db + prior_db) - lg
    return val, -backup_gap


def qb_adjustments(games, qbs, prior_db, backup_gap, starters=None):
    """
    Points to add for each side's starter compared with the QBs behind its rating.
    games needs a cutoff column; starters (optional) maps game_id -> (home_qb_id, away_qb_id).
    """
    hq, aq = np.zeros(len(games)), np.zeros(len(games))
    pos = {i: k for k, i in enumerate(games.index)}
    for cutoff, grp in games.groupby("cutoff"):
        val, backup = qb_values(qbs, cutoff, prior_db, backup_gap)
        base_q = qbs[(qbs.gameday < cutoff) & (qbs.gameday >= cutoff - pd.Timedelta(days=QB_BASE_DAYS))]
        base_q = base_q.assign(v=base_q.qb_id.map(val).fillna(backup))
        base = (base_q.v * base_q.dropbacks).groupby(base_q.team).sum() / base_q.groupby("team").dropbacks.sum()
        for i, g in grp.iterrows():
            ids = starters.get(g.game_id, (g.home_qb_id, g.away_qb_id)) if starters else (g.home_qb_id, g.away_qb_id)
            for side, team, qid, arr in [("home", g.home_team, ids[0], hq), ("away", g.away_team, ids[1], aq)]:
                if pd.isna(qid) or team not in base.index:
                    continue
                arr[pos[i]] = (val.get(qid, backup) - base[team]) * DROPBACKS_PER_GAME
    out = pd.DataFrame({"qb_home": hq, "qb_away": aq}, index=games.index)
    out["qb_diff"], out["qb_sum"] = out.qb_home - out.qb_away, out.qb_home + out.qb_away
    return out


# ------------------------------------------------------- projection model
def league_points(games, before, days=365):
    g = games[games.played & (games.gameday < before) & (games.gameday >= before - pd.Timedelta(days=days))]
    return (g.home_score.sum() + g.away_score.sum()) / max(2 * len(g), 1)


def fit_betas(wf, use_qb=True, measures=None):
    w = wf[wf.played]
    ms = measures or MODEL["measures"]
    mc = ["home_field"] + [f"diff_{m}" for m in ms] + ["rest_edge"] + (["qb_diff"] if use_qb else [])
    tc = ["const"] + [f"sum_{m}" for m in ms] + (["qb_sum"] if use_qb else [])
    bm = np.linalg.lstsq(w[mc].to_numpy(float), (w.home_score - w.away_score).to_numpy(float), rcond=None)[0]
    bt = np.linalg.lstsq(w[tc].to_numpy(float), (w.home_score + w.away_score - 2 * w.lg_pts).to_numpy(float),
                         rcond=None)[0]
    resid = (w.home_score - w.away_score) - w[mc].to_numpy(float) @ bm
    return {"margin": dict(zip(mc, bm)), "total": dict(zip(tc, bt)), "sd": float(resid.std())}


def predict(wf, b):
    p = wf.copy()
    p["proj_margin"] = sum(v * p[c] for c, v in b["margin"].items())
    p["proj_total"] = 2 * p.lg_pts + sum(v * p[c] for c, v in b["total"].items())
    return p


def score(p, label):
    p = p[p.played]
    margin, total = p.home_score - p.away_score, p.home_score + p.away_score
    hfa = margin[p.neutral == 0].mean()
    out = {"label": label, "games": len(p),
           "margin_MAE": (p.proj_margin - margin).abs().mean(),
           "home_edge_MAE": ((1 - p.neutral) * hfa - margin).abs().mean(),
           "winner_pct": 100 * ((p.proj_margin > 0) == (margin > 0))[margin != 0].mean(),
           "total_MAE": (p.proj_total - total).abs().mean()}
    m = p.spread_line.notna() & p.total_line.notna()
    if m.any():
        out["market_margin_MAE"] = (p.spread_line[m] - margin[m]).abs().mean()
        out["market_total_MAE"] = (p.total_line[m] - total[m]).abs().mean()
    return out


def prepare(games):
    games = games.copy()
    weeks = {d: week_start(d) for d in games.gameday.unique()}
    lg = {c: league_points(games, c) for c in set(weeks.values())}
    games["lg_pts"] = games.gameday.map(weeks).map(lg)
    return games


# ------------------------------------------------------------ backtest
def rolling_test(wf, seasons, use_qb):
    preds = []
    for s in seasons:
        b = fit_betas(wf[wf.season.between(s - BETA_SEASONS, s - 1)], use_qb)
        preds.append(predict(wf[wf.season == s], b))
    return pd.concat(preds)


def backtest(games, teams):
    finals = season_finals(games, teams, range(FIRST_SEASON - 1, N.CURRENT_SEASON))
    games = prepare(games)
    qbs = load_qbs(games)
    seasons = range(FIRST_SEASON, N.CURRENT_SEASON + 1)

    print("1) Team rating anchor, tuned on 2006-2015 (each game projected from earlier games only)")
    tune, cache = [], {}
    for pg, carry in itertools.product(PRIOR_GAMES_GRID, CARRY_GRID):
        wf = walk_forward(games, teams, finals, seasons,
                          dict(MODEL, pg_off=pg, pg_def=pg, carry_off=carry, carry_def=carry))
        cache[(pg, carry)] = wf
        tr = wf[wf.season.isin(TUNE_SEASONS)]
        s = score(predict(tr, fit_betas(tr, use_qb=False)), "tune")
        tune.append({"prior_games": pg, "carry": carry, "margin_MAE": s["margin_MAE"], "total_MAE": s["total_MAE"]})
    tune = pd.DataFrame(tune).sort_values("margin_MAE")
    pg, carry = int(tune.iloc[0].prior_games), float(tune.iloc[0].carry)
    print(f"   best: anchor {pg} games, carry-over {carry} (margin MAE {tune.iloc[0].margin_MAE:.3f}; "
          f"worst setting {tune.margin_MAE.max():.3f})")
    wf = cache[(pg, carry)]

    print("\n2) Quarterback adjustment, tuned on 2006-2015")
    qtune, qcache = [], {}
    for pdb, gap in itertools.product(QB_PRIOR_GRID, QB_BACKUP_GRID):
        w2 = wf.join(qb_adjustments(wf, qbs, pdb, gap))
        qcache[(pdb, gap)] = w2
        tr = w2[w2.season.isin(TUNE_SEASONS)]
        s = score(predict(tr, fit_betas(tr)), "tune")
        qtune.append({"qb_prior_dropbacks": pdb, "backup_gap": gap, "margin_MAE": s["margin_MAE"],
                      "total_MAE": s["total_MAE"]})
    qtune = pd.DataFrame(qtune).sort_values("margin_MAE")
    pdb, gap = int(qtune.iloc[0].qb_prior_dropbacks), float(qtune.iloc[0].backup_gap)
    print(qtune.round(3).to_string(index=False))
    print(f"   best: {pdb} prior dropbacks, backup level {gap} EPA per dropback below average")
    w2 = qcache[(pdb, gap)]

    test_seasons = list(TEST_SEASONS)
    rows = []
    for label, sel in [("2016-2025", test_seasons), (f"{N.CURRENT_SEASON} weeks 1-6", [N.CURRENT_SEASON])]:
        base = score(rolling_test(w2, sel, use_qb=False), label)
        withqb = score(rolling_test(w2, sel, use_qb=True), label)
        rows.append({"seasons": label, "games": base["games"], "home_edge": base["home_edge_MAE"],
                     "model_no_QB": base["margin_MAE"], "model_with_QB": withqb["margin_MAE"],
                     "market": base["market_margin_MAE"],
                     "total_no_QB": base["total_MAE"], "total_with_QB": withqb["total_MAE"],
                     "market_total": base["market_total_MAE"],
                     "winner_pct_no_QB": base["winner_pct"], "winner_pct_with_QB": withqb["winner_pct"]})
    comp = pd.DataFrame(rows)
    by = []
    for s in test_seasons + [N.CURRENT_SEASON]:
        a = score(rolling_test(w2, [s], False), str(s))
        b = score(rolling_test(w2, [s], True), str(s))
        by.append({"season": s, "games": a["games"], "no_QB": a["margin_MAE"], "with_QB": b["margin_MAE"],
                   "market": a["market_margin_MAE"]})
    by = pd.DataFrame(by)

    # games where a team's starter differed from its usual QB by a lot
    p = rolling_test(w2, test_seasons, True)
    p0 = rolling_test(w2, test_seasons, False)
    big = p.qb_diff.abs() >= 3
    margin = p.home_score - p.away_score
    print(f"\n3) Results (margin and total MAE in points; weights from the previous {BETA_SEASONS} seasons)")
    print(comp.round(2).to_string(index=False))
    print("\nBy season (margin MAE):")
    print(by.round(2).to_string(index=False))
    print(f"\nGames with a quarterback adjustment of 3+ points ({big.sum()} of {len(p)} in 2016-2025): "
          f"no QB {(p0.proj_margin[big] - margin[big]).abs().mean():.2f}, "
          f"with QB {(p.proj_margin[big] - margin[big]).abs().mean():.2f}, "
          f"market {(p.spread_line[big] - margin[big]).abs().mean():.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    tune.to_csv(os.path.join(OUT_DIR, "backtest_tuning.csv"), index=False)
    qtune.to_csv(os.path.join(OUT_DIR, "backtest_qb_tuning.csv"), index=False)
    by.to_csv(os.path.join(OUT_DIR, "backtest_by_season.csv"), index=False)


# -------------------------------------------------------------- current
def starters_for(week_games, games, qbs):
    """
    Each team's starting QB: data/qb_overrides.csv if given, otherwise the schedule's listed starter.
    If that QB is Out, Doubtful, on IR or suspended in the injury report, the team's healthy QB with the
    most recent dropbacks starts instead (or a backup-level QB if there is none).
    Returns ({game_id: (home_qb_id, away_qb_id)}, {qb_id: name}, [notes]).
    """
    path = os.path.join(N.DATA_DIR, "qb_overrides.csv")
    if not os.path.exists(path):
        pd.DataFrame(columns=["team", "qb_name"]).to_csv(path, index=False)
    ov = pd.read_csv(path)
    names = pd.concat([games[["home_qb_id", "home_qb_name"]].set_axis(["id", "name"], axis=1),
                       games[["away_qb_id", "away_qb_name"]].set_axis(["id", "name"], axis=1)]).dropna()
    pw = N.player_weekly(range(N.CURRENT_SEASON - 1, N.CURRENT_SEASON + 1))
    names = pd.concat([names, pw[pw.position == "QB"][["player_id", "player_display_name"]].set_axis(["id", "name"], axis=1)])
    by_name = names.drop_duplicates("name", keep="last").set_index("name").id
    id_name = {i: n for n, i in by_name.items()}
    ovr = {r.team: by_name.get(r.qb_name) for r in ov.itertuples()}
    inj = N.injury_report()
    recent = qbs[qbs.gameday >= pd.Timestamp(dt.date.today()) - pd.Timedelta(days=400)].sort_values(
        ["gameday", "dropbacks"], ascending=False)
    notes, chosen = [], {}

    def pick(team, listed):
        if team in chosen:
            return chosen[team]
        qid = ovr.get(team) or listed
        status = inj.get((team, N.name_key(id_name.get(qid, ""))), "")
        if not ovr.get(team) and N.is_out(status):
            alt = None
            for c in recent[recent.team == team].qb_id.drop_duplicates():
                if c != qid and not N.is_out(inj.get((team, N.name_key(id_name.get(c, ""))), "")):
                    alt = c
                    break
            notes.append(f"{team}: {id_name.get(qid, qid)} is {status}; starting "
                         f"{id_name.get(alt, 'a backup-level QB') if alt else 'a backup-level QB'}")
            qid = alt if alt else "BACKUP"
        chosen[team] = qid
        return qid

    out = {g.game_id: (pick(g.home_team, g.home_qb_id), pick(g.away_team, g.away_qb_id)) for g in week_games.itertuples()}
    id_name["BACKUP"] = "Backup QB"
    return out, id_name, notes


def current(games, teams):
    finals = season_finals(games, teams, range(FIRST_SEASON - 1, N.CURRENT_SEASON))
    games = prepare(games)
    qbs = load_qbs(games)
    wf = walk_forward(games, teams, finals, range(N.CURRENT_SEASON - BETA_SEASONS, N.CURRENT_SEASON))
    wf = wf.join(qb_adjustments(wf, qbs, QB_PRIOR_DROPBACKS, QB_BACKUP_GAP))
    b = fit_betas(wf)

    today = pd.Timestamp(dt.date.today())
    cur = games[games.season == N.CURRENT_SEASON]
    upcoming = cur[~cur.played & (cur.gameday >= today - pd.Timedelta(days=1))]
    if upcoming.empty:
        raise SystemExit("No upcoming games found.")
    week = int(upcoming.week.min())
    wk = upcoming[upcoming.week == week].copy()
    cutoff = today + pd.Timedelta(days=1)
    r = ratings_before(games, teams, finals, N.CURRENT_SEASON, cutoff)
    wk = wk.join(features(wk, r, teams))
    wk["cutoff"] = cutoff
    starters, id_name, qb_notes = starters_for(wk, games, qbs)
    wk = wk.join(qb_adjustments(wk, qbs, QB_PRIOR_DROPBACKS, QB_BACKUP_GAP, starters))
    wk["lg_pts"] = league_points(games, cutoff)
    p = predict(wk, b)
    p["proj_home"], p["proj_away"] = (p.proj_total + p.proj_margin) / 2, (p.proj_total - p.proj_margin) / 2
    p["home_win_prob"] = norm.cdf(p.proj_margin / b["sd"])
    p["home_qb"] = [id_name.get(starters[g][0], "") for g in p.game_id]
    p["away_qb"] = [id_name.get(starters[g][1], "") for g in p.game_id]

    # team table on the spread scale: power = offense + defense, points per game vs average, neutral field
    i = {t: k for k, t in enumerate(teams)}
    wts = {m: b["margin"][f"diff_{m}"] for m in r}
    team = pd.DataFrame({"team": teams})
    team["offense"] = [sum(w * r[m][0][i[t]] for m, w in wts.items()) for t in teams]
    team["defense"] = [-sum(w * r[m][1][i[t]] for m, w in wts.items()) for t in teams]
    team["power"] = team.offense + team.defense
    played = cur[cur.played]
    rec = pd.concat([pd.DataFrame({"team": played.home_team, "w": played.home_score > played.away_score,
                                   "l": played.home_score < played.away_score}),
                     pd.DataFrame({"team": played.away_team, "w": played.away_score > played.home_score,
                                   "l": played.away_score < played.home_score})]).groupby("team").sum()
    team["W"], team["L"] = team.team.map(rec.w).fillna(0).astype(int), team.team.map(rec.l).fillna(0).astype(int)
    team = team[team.team.isin(set(cur.home_team) | set(cur.away_team))]
    team = team.sort_values("power", ascending=False).reset_index(drop=True)
    team.insert(0, "rank", team.index + 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    team.round(2).to_csv(os.path.join(OUT_DIR, "team_ratings.csv"), index=False)
    out = p[["game_id", "week", "gameday", "away_team", "home_team", "away_qb", "home_qb", "neutral",
             "proj_away", "proj_home", "proj_margin", "proj_total", "home_win_prob", "qb_away", "qb_home"]].copy()
    out.insert(0, "as_of", today.date().isoformat())
    out["gameday"] = out.gameday.dt.date.astype(str)
    out = out.round(3)
    out.to_csv(os.path.join(OUT_DIR, "game_projections.csv"), index=False)
    hist = os.path.join(OUT_DIR, "game_projection_history.csv")
    out_h = out
    if os.path.exists(hist):
        old = pd.read_csv(hist)
        out_h = pd.concat([old[old.as_of != out.as_of.iloc[0]], out], ignore_index=True)
    out_h.to_csv(hist, index=False)
    with open(os.path.join(OUT_DIR, "qb_notes.json"), "w") as f:
        json.dump(qb_notes, f)

    def line(m, h, a):
        q = round(m * 2) / 2
        return "Pick'em" if q == 0 else (f"{h} -{q:.1f}" if q > 0 else f"{a} -{-q:.1f}")
    print(f"Week {week} projections (ratings through {cur[cur.played].gameday.max():%b %d}; "
          f"home field {b['margin']['home_field']:.1f}, game SD {b['sd']:.1f})\n")
    qb_note = lambda name, adj: f"{name} ({adj:+.1f})" if abs(adj) >= 0.5 else name
    show = pd.DataFrame({"Date": out.gameday, "Away": out.away_team, "Home": out.home_team,
                         "Away QB": [qb_note(n, a) for n, a in zip(out.away_qb, out.qb_away)],
                         "Home QB": [qb_note(n, a) for n, a in zip(out.home_qb, out.qb_home)],
                         "Score": [f"{a:.1f}-{h:.1f}" for a, h in zip(out.proj_away, out.proj_home)],
                         "Line": [line(m, h, a) for m, h, a in zip(out.proj_margin, out.home_team, out.away_team)],
                         "Total": (out.proj_total * 2).round() / 2,
                         "Home win": (out.home_win_prob * 100).round().astype(int).astype(str) + "%"})
    print(show.to_string(index=False))
    for n in qb_notes:
        print(f"  Starter change from the injury report - {n}")
    print("\nQB adjustments in parentheses: points vs the quarterbacks behind that team's rating (shown when 0.5+).")
    print("\nTeam ratings (points per game vs an average team, neutral field)\n")
    print(team.round(1).to_string(index=False))


def history(games, teams, first=2014):
    """
    Every played game's projection made the way it would have been at the time (earlier games and
    earlier seasons' weights only). Saved to output/game_preds_history.csv for the player backtest.
    """
    finals = season_finals(games, teams, range(FIRST_SEASON - 1, N.CURRENT_SEASON))
    games = prepare(games)
    qbs = load_qbs(games)
    wf = walk_forward(games, teams, finals, range(first - BETA_SEASONS, N.CURRENT_SEASON + 1))
    wf = wf.join(qb_adjustments(wf, qbs, QB_PRIOR_DROPBACKS, QB_BACKUP_GAP))
    p = rolling_test(wf, range(first, N.CURRENT_SEASON + 1), use_qb=True)
    p = p[p.played][["game_id", "season", "week", "gameday", "home_team", "away_team", "proj_margin", "proj_total"]]
    os.makedirs(OUT_DIR, exist_ok=True)
    p.to_csv(os.path.join(OUT_DIR, "game_preds_history.csv"), index=False)
    print(f"Saved {len(p)} past game projections ({first}-{N.CURRENT_SEASON}).")


def main():
    games = load_games()
    teams = sorted(set(games.home_team) | set(games.away_team))
    if "--history" in sys.argv:
        history(games, teams)
    elif "--backtest" in sys.argv:
        backtest(games, teams)
    else:
        current(games, teams)


if __name__ == "__main__":
    main()
