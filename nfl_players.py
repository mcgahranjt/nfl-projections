"""
NFL player stat and fantasy projections (QB, RB, WR, TE)

Each player's projection is team volume x his share x his efficiency, tied to the game projections:
  1. Team volume. Pass attempts and carries from the team's recent games (pulled toward league average),
     shifted by game script: teams projected to lead run more, teams projected to trail pass more.
  2. Shares. Each player's share of the team's targets and carries in recent games he played (recent
     games count more). Out, Doubtful, IR and suspended players are removed and their shares go to
     teammates in proportion to their own. Questionable players are projected as playing and flagged.
  3. Efficiency. Catch rate, yards per target, yards per carry, touchdown and fumble rates, and the
     QB's completion, yardage, touchdown and interception rates, each pulled toward the position average
     when the sample is small. Yards are adjusted for the opponent's defense.
  4. Consistency with the game projection. Receivers' catches and yards add up to the QB's passing line,
     and the team's touchdowns are scaled to its projected points.
  5. Ranges. Each player's game is simulated N_SIMS times (team volume varies together for teammates),
     and the league's scoring from league_scoring.json is applied to every simulated stat line, so the
     tiered yardage bonuses are counted exactly.

  6. Calibration (from --calibrate): by position, the projection is blended with the player's season
     average so far, and the range is widened or narrowed so about 10% of results land below P10 and
     10% above P90. Both are chosen on 2016-2019 only and checked on 2020-2025.

Usage:
  python3 nfl_players.py              # this week's projections (run nfl_ratings.py first)
  python3 nfl_players.py --backtest   # project every past game from earlier data (2016-now)
  python3 nfl_players.py --calibrate  # choose the blend and range widths from the backtest

Outputs (output/): player_projections_week<N>.csv, player_projection_history.csv,
and with --backtest: player_backtest.csv, player_backtest_summary.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd

import nfl_data as N
import nfl_ratings as R

OUT_DIR = "output"
SEASONS = range(2015, N.CURRENT_SEASON + 1)
TRAIN_SEASONS = range(2016, 2020)          # game-script effects and dispersion are set here
TEST_SEASONS = range(2020, 2026)
POSITIONS = ["QB", "RB", "WR", "TE"]
N_SIMS = 5000
SEED = 11

SHARE_HALF_LIFE = 4        # team games; a player's role 4 games ago counts half
OFFSEASON_GAMES = 6        # last season's games count as if this many extra games older
TEAM_PRIOR_GAMES = 4       # team volume is pulled toward league average by this many games
SHARE_PRIOR_GAMES = 1      # each player's share is pulled toward his position's typical share
EFF_HALF_LIFE_DAYS = 365
EFF_PRIOR = {"catch": 40, "ypt": 60, "td_tgt": 150, "ypc": 80, "td_car": 150, "fum": 250,
             "comp": 300, "ypa": 300, "td_att": 400, "int_att": 400, "twopt": 30}
OPP_PRIOR_GAMES = 6
NEW_PLAYER_SHARE = 0.5     # players with no games yet start at half their position's typical share
SHARE_FLOOR = 0.005
# simulation spreads (higher k = steadier); checked against the 10th-90th percentile coverage in --backtest
TEAM_VOLUME_SD = 0.12
K_TARGETS, K_CARRIES = 8.0, 6.0
K_REC_YARDS, K_PASS_YARDS = 1.3, 4.0
RUSH_YD_SD_PER_CARRY = 5.5

SCORING = json.load(open("league_scoring.json"))


# ---------------------------------------------------------------- data
def load():
    pw = N.player_weekly(SEASONS)
    pw = pw[pw.season_type.isin(["REG", "POST"])].copy()
    pw["team"] = pw.team.replace(R.FRANCHISE)
    pw["opponent_team"] = pw.opponent_team.replace(R.FRANCHISE)
    pw["position"] = pw.position.replace({"FB": "RB"})
    num = [c for c in pw.columns if c not in ("player_id", "player_display_name", "position", "season_type",
                                              "game_id", "team", "opponent_team")]
    pw[num] = pw[num].fillna(0)
    pw["fumbles_lost"] = pw.sack_fumbles_lost + pw.rushing_fumbles_lost + pw.receiving_fumbles_lost
    pw["twopt"] = pw.passing_2pt_conversions + pw.rushing_2pt_conversions + pw.receiving_2pt_conversions
    games = R.load_games()
    g = games[["game_id", "gameday", "season", "week", "home_team", "away_team", "home_score", "away_score",
               "home_qb_id", "away_qb_id", "played"]]
    pw = pw.drop(columns=["season", "week"]).merge(g[["game_id", "gameday", "season", "week"]], on="game_id")
    pw = pw.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    # team totals per game
    tg = pw.groupby(["game_id", "team", "opponent_team", "gameday", "season", "week"]).agg(
        att=("attempts", "sum"), comp=("completions", "sum"), pass_yds=("passing_yards", "sum"),
        sacks=("sacks_suffered", "sum"), carries=("carries", "sum"), rush_yds=("rushing_yards", "sum"),
        targets=("targets", "sum"), pass_tds=("passing_tds", "sum"), rush_tds=("rushing_tds", "sum")).reset_index()
    tg = tg.sort_values(["team", "gameday"]).reset_index(drop=True)
    return pw, tg, games


def fantasy(stats):
    """League fantasy points for a dict (or DataFrame) of stat arrays."""
    s = SCORING
    pts = (s["passing_yards"] * stats["pass_yds"] + s["passing_tds"] * stats["pass_td"]
           + s["passing_interceptions"] * stats["int"] + s["rushing_yards"] * stats["rush_yds"]
           + s["rushing_tds"] * stats["rush_td"] + s["receptions"] * stats["rec"]
           + s["receiving_yards"] * stats["rec_yds"] + s["receiving_tds"] * stats["rec_td"]
           + s["fumbles_lost"] * stats["fum"] + s["passing_2pt"] * stats["twopt"])
    for stat, key in [("rush_yds", "rushing_yards"), ("rec_yds", "receiving_yards"), ("pass_yds", "passing_yards")]:
        tiers = sorted(s["bonuses"][key])
        bonus = 0
        for threshold, value in tiers:                 # tiered: the highest tier reached replaces lower ones
            bonus = np.where(stats[stat] >= threshold, value, bonus)
        pts = pts + bonus
    return pts


def actual_fantasy(pw):
    return fantasy({"pass_yds": pw.passing_yards, "pass_td": pw.passing_tds, "int": pw.passing_interceptions,
                    "rush_yds": pw.rushing_yards, "rush_td": pw.rushing_tds, "rec": pw.receptions,
                    "rec_yds": pw.receiving_yards, "rec_td": pw.receiving_tds, "fum": pw.fumbles_lost,
                    "twopt": pw.twopt}) + SCORING["special_teams_tds"] * pw.special_teams_tds \
        + SCORING["fumble_recovery_tds"] * pw.fumble_recovery_tds


# ------------------------------------------------------------- inputs
def team_game_weights(tg, team, season):
    """Recency weights for a team's past games (most recent first), with last season pushed further back."""
    t = tg[tg.team == team].sort_values("gameday", ascending=False)
    ago = np.arange(len(t)) + np.where(t.season < season, OFFSEASON_GAMES, 0)
    return t.assign(w=0.5 ** (ago / SHARE_HALF_LIFE))


def league_rates(pw, tg):
    """Position-average rates used as the pull-toward values, from the data before the cutoff."""
    r = {}
    for pos in POSITIONS:
        p = pw[pw.position == pos]
        tgt, car = max(p.targets.sum(), 1), max(p.carries.sum(), 1)
        r[pos] = {"catch": p.receptions.sum() / tgt, "ypt": p.receiving_yards.sum() / tgt,
                  "td_tgt": p.receiving_tds.sum() / tgt, "ypc": p.rushing_yards.sum() / car,
                  "td_car": p.rushing_tds.sum() / car,
                  "fum": p.fumbles_lost.sum() / max(p.targets.sum() + p.carries.sum() + p.attempts.sum(), 1),
                  "twopt": p.twopt.sum() / max(len(p), 1)}
    q = pw[pw.position == "QB"]
    att = max(q.attempts.sum(), 1)
    r["QB"].update({"comp": q.completions.sum() / att, "ypa": q.passing_yards.sum() / att,
                    "td_att": q.passing_tds.sum() / att, "int_att": q.passing_interceptions.sum() / att})
    r["team"] = {"att": tg.att.mean(), "carries": tg.carries.mean(), "tgt_per_att": tg.targets.sum() / tg.att.sum(),
                 "ypa": tg.pass_yds.sum() / tg.att.sum(), "ypc": tg.rush_yds.sum() / tg.carries.sum(),
                 "td_share_pass": tg.pass_tds.sum() / max(tg.pass_tds.sum() + tg.rush_tds.sum(), 1)}
    return r


def shrink(num, den, prior_rate, k):
    return (num + k * prior_rate) / (den + k)


def player_efficiency(pw, cutoff, lg):
    """Each player's efficiency rates as of a date (recent seasons count more)."""
    p = pw[pw.gameday < cutoff]
    w = 0.5 ** ((cutoff - p.gameday).dt.days / EFF_HALF_LIFE_DAYS)
    cols = ["targets", "receptions", "receiving_yards", "receiving_tds", "carries", "rushing_yards", "rushing_tds",
            "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions", "fumbles_lost", "twopt"]
    agg = p[cols].mul(w, axis=0).assign(player_id=p.player_id, games=w).groupby("player_id").sum()
    pos = p.groupby("player_id").position.last()
    e = pd.DataFrame(index=agg.index)
    get = lambda key: pos.map(lambda x: lg.get(x, lg["WR"])[key])
    e["catch"] = shrink(agg.receptions, agg.targets, get("catch"), EFF_PRIOR["catch"])
    e["ypt"] = shrink(agg.receiving_yards, agg.targets, get("ypt"), EFF_PRIOR["ypt"])
    e["td_tgt"] = shrink(agg.receiving_tds, agg.targets, get("td_tgt"), EFF_PRIOR["td_tgt"])
    e["ypc"] = shrink(agg.rushing_yards, agg.carries, get("ypc"), EFF_PRIOR["ypc"])
    e["td_car"] = shrink(agg.rushing_tds, agg.carries, get("td_car"), EFF_PRIOR["td_car"])
    touches = agg.targets + agg.carries + agg.attempts
    e["fum"] = shrink(agg.fumbles_lost, touches, get("fum"), EFF_PRIOR["fum"])
    e["twopt"] = shrink(agg.twopt, agg.games, get("twopt"), EFF_PRIOR["twopt"])
    qb = lg["QB"]
    e["comp"] = shrink(agg.completions, agg.attempts, qb["comp"], EFF_PRIOR["comp"])
    e["ypa"] = shrink(agg.passing_yards, agg.attempts, qb["ypa"], EFF_PRIOR["ypa"])
    e["td_att"] = shrink(agg.passing_tds, agg.attempts, qb["td_att"], EFF_PRIOR["td_att"])
    e["int_att"] = shrink(agg.passing_interceptions, agg.attempts, qb["int_att"], EFF_PRIOR["int_att"])
    return e


def opponent_factors(tg, cutoff, lg):
    """How many passing and rushing yards each defense allows per play, relative to league (recent games count more)."""
    d = tg[tg.gameday < cutoff].sort_values("gameday", ascending=False)
    out = {}
    for opp, g in d.groupby("opponent_team"):
        g = g.head(20)
        w = 0.5 ** (np.arange(len(g)) / 8)
        ypa = shrink((g.pass_yds * w).sum(), (g.att * w).sum(), lg["team"]["ypa"], OPP_PRIOR_GAMES * lg["team"]["att"])
        ypc = shrink((g.rush_yds * w).sum(), (g.carries * w).sum(), lg["team"]["ypc"],
                     OPP_PRIOR_GAMES * lg["team"]["carries"])
        out[opp] = (ypa / lg["team"]["ypa"], ypc / lg["team"]["ypc"])
    return out


# ------------------------------------------------------------ projection
def project_team(team, opp, season, active, qb_id, team_pts, margin, pw, tg, eff, lg, opp_f, script, td_per_pt):
    """Expected stat lines for one team's active players in one game."""
    T = lg["team"]
    tw = team_game_weights(tg[tg.gameday < pd.Timestamp.max], team, season).head(24)
    wsum = tw.w.sum()
    att = shrink((tw.att * tw.w).sum(), wsum, T["att"], TEAM_PRIOR_GAMES) + script["att"] * margin
    car = shrink((tw.carries * tw.w).sum(), wsum, T["carries"], TEAM_PRIOR_GAMES) + script["car"] * margin
    tgt_per_att = shrink((tw.targets * tw.w).sum(), (tw.att * tw.w).sum(), T["tgt_per_att"], TEAM_PRIOR_GAMES * T["att"])
    f_pass, f_rush = opp_f.get(opp, (1.0, 1.0))

    # shares in recent games each player played for this team
    wmap = tw.set_index("game_id").w
    team_tot = tw.set_index("game_id")[["targets", "carries"]]
    rows = []
    for pid, pos, name in active:
        g = pw[(pw.player_id == pid) & pw.game_id.isin(wmap.index)]
        typical = TYPICAL_SHARE[pos]
        if len(g):
            w = g.game_id.map(wmap)
            tot = team_tot.loc[g.game_id]
            ts = shrink((g.targets * w.values).sum(), (tot.targets.values * w.values).sum(),
                        typical["tgt"], SHARE_PRIOR_GAMES * T["att"] * T["tgt_per_att"])
            cs = shrink((g.carries * w.values).sum(), (tot.carries.values * w.values).sum(),
                        typical["car"], SHARE_PRIOR_GAMES * T["carries"])
        else:
            ts, cs = NEW_PLAYER_SHARE * typical["tgt"], NEW_PLAYER_SHARE * typical["car"]
        if pos == "QB" and pid != qb_id:
            continue                                   # only the starting QB is projected
        rows.append({"player_id": pid, "name": name, "position": pos,
                     "tgt_share": 0.0 if pos == "QB" else max(ts, SHARE_FLOOR), "car_share": max(cs, SHARE_FLOOR)})
    if not rows:
        return None
    p = pd.DataFrame(rows)
    for c in ["att", "comp", "pass_yds", "int", "pass_td"]:
        p[c] = 0.0
    p["tgt_share"] /= max(p.tgt_share.sum(), 1e-9)
    p["car_share"] /= max(p.car_share.sum(), 1e-9)
    e = eff.reindex(p.player_id)
    for k in ["catch", "ypt", "td_tgt", "ypc", "td_car", "fum", "twopt", "comp", "ypa", "td_att", "int_att"]:
        fill = [lg[pos].get(k, lg["QB"].get(k, 0)) for pos in p.position]
        p[k] = np.where(e[k].isna(), fill, e[k].to_numpy())

    team_tgts = att * tgt_per_att
    p["targets"] = p.tgt_share * team_tgts
    p["rec"] = p.targets * p["catch"]
    p["rec_yds"] = p.targets * p.ypt * f_pass
    p["carries"] = p.car_share * car
    p["rush_yds"] = p.carries * p.ypc * f_rush
    raw_rec_td, raw_rush_td = p.targets * p.td_tgt, p.carries * p.td_car
    # QB line: the starter's passing, with receivers' catches and yards added up to match it
    qbm = p.position == "QB"
    if qbm.any():
        q = p[qbm].iloc[0]
        pass_yds = att * q.ypa * f_pass
        comp = att * q["comp"]
        p.loc[qbm, ["att", "comp", "pass_yds", "int"]] = [att, comp, pass_yds, att * q.int_att]
        if p.rec.sum() > 0:
            p["rec_yds"] *= pass_yds / max(p.rec_yds.sum(), 1e-9)
            p["rec"] *= comp / max(p.rec.sum(), 1e-9)
            p["rec"] = np.minimum(p.rec, p.targets)
    # touchdowns scaled to the team's projected points
    tds = td_per_pt * team_pts
    raw = raw_rec_td.sum() + raw_rush_td.sum()
    scale = tds / raw if raw > 0 else 1.0
    p["rec_td"], p["rush_td"] = raw_rec_td * scale, raw_rush_td * scale
    if qbm.any():
        p.loc[qbm, "pass_td"] = p.rec_td.sum()
    p["fum"] = (p.targets + p.carries + p.att) * p.fum
    p["team"], p["opp"] = team, opp
    return p


TYPICAL_SHARE = {}


def set_typical_shares(pw, tg):
    """Average share of team targets and carries for a player at each position when he plays."""
    m = pw.merge(tg[["game_id", "team", "targets", "carries"]].rename(columns={"targets": "t_tgt", "carries": "t_car"}),
                 on=["game_id", "team"])
    for pos in POSITIONS:
        x = m[m.position == pos]
        TYPICAL_SHARE[pos] = {"tgt": (x.targets / x.t_tgt.clip(lower=1)).mean(),
                              "car": (x.carries / x.t_car.clip(lower=1)).mean()}


def simulate(p, rng, n=N_SIMS):
    """Simulated stat lines for a team's players; volume moves together for teammates."""
    k = len(p)
    vol = np.clip(rng.normal(1.0, TEAM_VOLUME_SD, n), 0.5, 1.6)[None, :]
    def negbin(mean, kk):
        mean = np.maximum(mean, 1e-6)
        return rng.negative_binomial(kk, kk / (kk + mean))
    tgt = negbin(p.targets.to_numpy()[:, None] * vol, K_TARGETS)
    catch = np.clip((p.rec / p.targets.clip(lower=1e-9)).to_numpy(), 0, 1)[:, None]
    rec = rng.binomial(tgt, np.broadcast_to(catch, tgt.shape))
    ypr = (p.rec_yds / p.rec.clip(lower=1e-9)).to_numpy()[:, None]
    rec_yds = np.where(rec > 0, rng.gamma(np.maximum(rec, 1) * K_REC_YARDS, np.broadcast_to(ypr / K_REC_YARDS, rec.shape)), 0)
    car = negbin(p.carries.to_numpy()[:, None] * vol, K_CARRIES)
    ypc = (p.rush_yds / p.carries.clip(lower=1e-9)).to_numpy()[:, None]
    rush_yds = np.round(car * ypc + RUSH_YD_SD_PER_CARRY * np.sqrt(car) * rng.standard_normal(car.shape))
    rec_td = rng.poisson(p.rec_td.to_numpy()[:, None] * vol)
    rush_td = rng.poisson(p.rush_td.to_numpy()[:, None] * vol)
    att = np.round(p.att.to_numpy()[:, None] * vol + 3 * rng.standard_normal((k, n))).clip(0)
    cp = np.clip((p.comp / p.att.clip(lower=1e-9)).to_numpy(), 0, 1)[:, None]
    comp = rng.binomial(att.astype(int), np.broadcast_to(cp, att.shape))
    ypcomp = (p.pass_yds / p.comp.clip(lower=1e-9)).to_numpy()[:, None]
    pass_yds = np.where(comp > 0, rng.gamma(np.maximum(comp, 1) * K_PASS_YARDS,
                                            np.broadcast_to(ypcomp / K_PASS_YARDS, comp.shape)), 0)
    pass_td = rng.poisson(p.pass_td.to_numpy()[:, None] * vol)
    ints = rng.poisson(np.broadcast_to(p["int"].to_numpy()[:, None], (k, n)))
    fum = rng.poisson(np.broadcast_to(p.fum.to_numpy()[:, None], (k, n)))
    twopt = rng.poisson(np.broadcast_to(p.twopt.to_numpy()[:, None], (k, n)))
    stats = {"pass_yds": np.round(pass_yds), "pass_td": pass_td, "int": ints, "rush_yds": rush_yds,
             "rush_td": rush_td, "rec": rec, "rec_yds": np.round(rec_yds), "rec_td": rec_td, "fum": fum,
             "twopt": twopt, "targets": tgt, "carries": car}
    stats["fpts"] = fantasy(stats)
    return stats


def summarize(p, sims):
    out = p[["player_id", "name", "position", "team", "opp"]].copy()
    for c in ["pass_yds", "pass_td", "int", "carries", "rush_yds", "rush_td", "targets", "rec", "rec_yds", "rec_td"]:
        out[c] = sims[c].mean(axis=1)
    f = sims["fpts"]
    out["fpts"] = f.mean(axis=1)
    out["fpts_p10"], out["fpts_median"], out["fpts_p90"] = np.percentile(f, [10, 50, 90], axis=1)
    for stat, key in [("pass_yds", "passing_yards"), ("rush_yds", "rushing_yards"), ("rec_yds", "receiving_yards")]:
        out[f"{stat}_bonus_pct"] = 100 * (sims[stat] >= sorted(SCORING["bonuses"][key])[0][0]).mean(axis=1)
    return out


# ------------------------------------------------------------ backtest
def week_inputs(pw, tg, season, week):
    cut = pw[(pw.season < season) | ((pw.season == season) & (pw.week < week))]
    cut_tg = tg[(tg.season < season) | ((tg.season == season) & (tg.week < week))]
    return cut, cut_tg


def fit_script_and_tds(pw, tg, preds):
    """Game-script effect on pass attempts and carries, and touchdowns per projected point (training seasons)."""
    gp = preds.copy()
    long = pd.concat([gp.assign(team=gp.home_team, margin=gp.proj_margin, pts=(gp.proj_total + gp.proj_margin) / 2),
                      gp.assign(team=gp.away_team, margin=-gp.proj_margin, pts=(gp.proj_total - gp.proj_margin) / 2)])
    m = tg.merge(long[["game_id", "team", "margin", "pts"]], on=["game_id", "team"])
    m = m[m.season.isin(TRAIN_SEASONS)]
    base_att = m.groupby(["team", "season"]).att.transform("mean")
    base_car = m.groupby(["team", "season"]).carries.transform("mean")
    script = {"att": np.polyfit(m.margin, m.att - base_att, 1)[0], "car": np.polyfit(m.margin, m.carries - base_car, 1)[0]}
    td_per_pt = (m.pass_tds + m.rush_tds).sum() / m.pts.sum()
    return script, td_per_pt


def run_week(pw, tg, games, preds, season, week, script, td_per_pt, rng, n_sims, actual_active=True):
    hist_pw, hist_tg = week_inputs(pw, tg, season, week)
    lg = league_rates(hist_pw[hist_pw.season >= season - 3], hist_tg[hist_tg.season >= season - 3])
    eff = player_efficiency(hist_pw, pd.Timestamp(games[(games.season == season) & (games.week == week)].gameday.min()), lg)
    opp_f = opponent_factors(hist_tg, pd.Timestamp.max, lg)
    wk = games[(games.season == season) & (games.week == week) & games.played]
    out = []
    this = pw[(pw.season == season) & (pw.week == week)]
    for g in wk.itertuples():
        pr = preds.get(g.game_id)
        if pr is None:
            continue
        margin_h, total = pr
        for team, opp, qb, sign in [(g.home_team, g.away_team, g.home_qb_id, 1), (g.away_team, g.home_team, g.away_qb_id, -1)]:
            act = this[(this.game_id == g.game_id) & (this.team == team)]
            active = list(zip(act.player_id, act.position, act.player_display_name))
            pts = (total + sign * margin_h) / 2
            p = project_team(team, opp, season, active, qb, pts, sign * margin_h, hist_pw[hist_pw.team == team],
                             hist_tg[hist_tg.team == team], eff, lg, opp_f, script, td_per_pt)
            if p is None:
                continue
            s = summarize(p, simulate(p, rng, n_sims))
            s["game_id"], s["season"], s["week"] = g.game_id, season, week
            out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def backtest(pw, tg, games):
    preds_df = pd.read_csv(os.path.join(OUT_DIR, "game_preds_history.csv"))
    preds = {r.game_id: (r.proj_margin, r.proj_total) for r in preds_df.itertuples()}
    set_typical_shares(pw[pw.season.isin(TRAIN_SEASONS)], tg[tg.season.isin(TRAIN_SEASONS)])
    script, td_per_pt = fit_script_and_tds(pw, tg, preds_df)
    print(f"Game script: {script['att']:+.2f} pass attempts and {script['car']:+.2f} carries per point of "
          f"projected margin; {td_per_pt:.3f} offensive TDs per projected point (fit on 2016-2019).")
    rng = np.random.default_rng(SEED)
    pw = pw.assign(fpts_actual=actual_fantasy(pw))
    rows = []
    for season in list(TRAIN_SEASONS) + list(TEST_SEASONS) + [N.CURRENT_SEASON]:
        weeks = sorted(games[(games.season == season) & games.played & (games.game_type == "REG")].week.unique())
        for week in weeks:
            r = run_week(pw, tg, games, preds, season, week, script, td_per_pt, rng, 1000)
            if len(r):
                rows.append(r)
        print(f"  {season} done", flush=True)
    proj = pd.concat(rows, ignore_index=True)
    act = pw[["game_id", "player_id", "fpts_actual", "gameday", "season", "week", "team"]]
    m = proj.merge(act, on=["game_id", "player_id", "season", "week", "team"])
    # baseline: the player's average fantasy points in this season's earlier games (2+ games), else last season's
    allp = pw[["player_id", "season", "week", "gameday", "fpts_actual"]].sort_values("gameday")
    allp["prior_avg"] = allp.groupby(["player_id", "season"]).fpts_actual.transform(lambda x: x.shift().expanding().mean())
    allp["prior_n"] = allp.groupby(["player_id", "season"]).cumcount()
    last = allp.groupby(["player_id", "season"]).fpts_actual.mean().rename("last_season").reset_index()
    last["season"] += 1
    allp = allp.merge(last, on=["player_id", "season"], how="left")
    allp["baseline"] = np.where(allp.prior_n >= 2, allp.prior_avg, allp.last_season)
    m = m.merge(allp[["player_id", "season", "week", "baseline"]], on=["player_id", "season", "week"], how="left")
    os.makedirs(OUT_DIR, exist_ok=True)
    m.to_csv(os.path.join(OUT_DIR, "player_backtest.csv"), index=False)

    def summary(x, label):
        b = x.baseline.notna()
        rank = []
        for _, g in x[b].groupby(["season", "week", "position"]):
            if len(g) >= 8:
                rank.append((g.fpts.rank().corr(g.fpts_actual.rank()), g.baseline.rank().corr(g.fpts_actual.rank())))
        rank = np.array(rank)
        return {"group": label, "player_games": int(b.sum()),
                "model_MAE": (x.fpts - x.fpts_actual)[b].abs().mean(),
                "baseline_MAE": (x.baseline - x.fpts_actual)[b].abs().mean(),
                "model_bias": (x.fpts - x.fpts_actual)[b].mean(),
                "model_rank_corr": rank[:, 0].mean() if len(rank) else np.nan,
                "baseline_rank_corr": rank[:, 1].mean() if len(rank) else np.nan,
                "pct_below_P10": 100 * (x.fpts_actual < x.fpts_p10).mean(),
                "pct_above_P90": 100 * (x.fpts_actual > x.fpts_p90).mean()}
    rows = []
    test = m[m.season.isin(TEST_SEASONS)]
    for pos in POSITIONS:
        rows.append(summary(test[test.position == pos], f"2020-2025 {pos}"))
    rows.append(summary(test, "2020-2025 all"))
    cur = m[m.season == N.CURRENT_SEASON]
    rows.append(summary(cur, f"{N.CURRENT_SEASON} weeks so far"))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT_DIR, "player_backtest_summary.csv"), index=False)
    print("\nFantasy points: model vs the player's season average so far (same player-games)")
    print("rank_corr = how well each method orders players at a position each week (1 = perfect)")
    print(s.round(2).to_string(index=False))


# -------------------------------------------------------------- current
def current(pw, tg, games):
    proj_path = os.path.join(OUT_DIR, "game_projections.csv")
    if not os.path.exists(proj_path):
        raise SystemExit("Run nfl_ratings.py first.")
    gp = pd.read_csv(proj_path)
    preds_df = pd.read_csv(os.path.join(OUT_DIR, "game_preds_history.csv"))
    set_typical_shares(pw[pw.season.isin(TRAIN_SEASONS)], tg[tg.season.isin(TRAIN_SEASONS)])
    script, td_per_pt = fit_script_and_tds(pw, tg, preds_df)
    season, week = N.CURRENT_SEASON, int(gp.week.iloc[0])
    lg = league_rates(pw[pw.season >= season - 3], tg[tg.season >= season - 3])
    eff = player_efficiency(pw, pd.Timestamp(gp.gameday.min()), lg)
    opp_f = opponent_factors(tg, pd.Timestamp.max, lg)
    inj = N.injury_report()
    rng = np.random.default_rng(SEED)
    sched = games.set_index("game_id")
    names = pd.concat([games[["home_qb_id", "home_qb_name"]].set_axis(["id", "name"], axis=1),
                       games[["away_qb_id", "away_qb_name"]].set_axis(["id", "name"], axis=1)]).dropna()
    qb_name_to_id = names.drop_duplicates("name", keep="last").set_index("name").id
    out, removed = [], []
    for g in gp.itertuples():
        for team, opp, qb_name, sign in [(g.home_team, g.away_team, g.home_qb, 1), (g.away_team, g.home_team, g.away_qb, -1)]:
            qb_id = qb_name_to_id.get(qb_name)
            recent = tg[(tg.team == team) & (tg.season == season)].sort_values("gameday").tail(4).game_id
            cand = pw[pw.game_id.isin(recent) & (pw.team == team)].drop_duplicates("player_id", keep="last")
            active, flags = [], {}
            for r in cand.itertuples():
                status = inj.get((team, N.name_key(r.player_display_name)), "")
                if N.is_out(status):
                    removed.append(f"{r.player_display_name} ({team}) - {status}")
                    continue
                if r.position == "QB" and r.player_id != qb_id:
                    continue
                active.append((r.player_id, r.position, r.player_display_name))
                flags[r.player_id] = "Questionable" if status.lower().startswith("questionable") else ""
            if qb_id and qb_id not in [a[0] for a in active]:
                active.append((qb_id, "QB", qb_name))   # the starter chosen by nfl_ratings.py (already injury-checked)
            pts = g.proj_home if sign == 1 else g.proj_away
            p = project_team(team, opp, season, active, qb_id, pts, sign * g.proj_margin, pw[pw.team == team],
                             tg[tg.team == team],
                             eff, lg, opp_f, script, td_per_pt)
            if p is None:
                continue
            s = summarize(p, simulate(p, rng))
            s["status"] = s.player_id.map(flags).fillna("")
            s["gameday"], s["game_id"] = g.gameday, g.game_id
            out.append(s)
    proj = pd.concat(out, ignore_index=True)
    if os.path.exists(CALIBRATION_PATH):
        cal = json.load(open(CALIBRATION_PATH))
        fp = pw.assign(f=actual_fantasy(pw))
        cur_avg = fp[fp.season == season].groupby("player_id").f.agg(["mean", "size"])
        last_avg = fp[fp.season == season - 1].groupby("player_id").f.mean()
        base = cur_avg["mean"].where(cur_avg["size"] >= 2)
        proj["baseline"] = proj.player_id.map(base).fillna(proj.player_id.map(last_avg))
        proj = apply_calibration(proj, cal)
    proj = proj[(proj.fpts >= 1.0)].sort_values("fpts", ascending=False)
    cols = ["name", "position", "team", "opp", "status", "fpts", "fpts_p10", "fpts_median", "fpts_p90",
            "pass_yds", "pass_td", "int", "carries", "rush_yds", "rush_td", "targets", "rec", "rec_yds", "rec_td",
            "pass_yds_bonus_pct", "rush_yds_bonus_pct", "rec_yds_bonus_pct", "gameday", "game_id", "player_id"]
    proj = proj[cols].round(2)
    os.makedirs(OUT_DIR, exist_ok=True)
    proj.to_csv(os.path.join(OUT_DIR, f"player_projections_week{week}.csv"), index=False)
    hist = os.path.join(OUT_DIR, "player_projection_history.csv")
    snap = proj.assign(season=season, week=week, as_of=pd.Timestamp.today().date().isoformat())
    if os.path.exists(hist):
        old = pd.read_csv(hist)
        # replace only the games projected in this run (a Sunday run must not wipe Thursday's game)
        keep = ~((old.season == season) & (old.week == week) & old.team.isin(set(snap.team)))
        snap = pd.concat([old[keep], snap], ignore_index=True)
    snap.to_csv(hist, index=False)
    if removed:
        print("Not projected (Out, Doubtful, IR or suspended):\n  " + "\n  ".join(sorted(set(removed))))
    print(f"\nWeek {week} fantasy projections (your league's scoring; P10-P90 = range of outcomes)\n")
    for pos in POSITIONS:
        x = proj[proj.position == pos].head(15)
        show = pd.DataFrame({"Player": x.name, "Team": x.team, "Opp": x.opp, "Status": x.status,
                             "Proj": x.fpts.round(1), "P10": x.fpts_p10.round(1), "P90": x.fpts_p90.round(1)})
        print(f"{pos}\n{show.to_string(index=False)}\n")



# ---------------------------------------------------------- calibration
CALIBRATION_PATH = os.path.join(OUT_DIR, "player_calibration.json")


def apply_calibration(df, cal):
    """
    Blend the model with the player's season average and rescale the range, by position.
    df needs fpts, fpts_p10, fpts_median, fpts_p90, position and baseline (NaN when there is none).
    """
    out = df.copy()
    for pos, c in cal.items():
        m = out.position == pos
        med = out.loc[m, "fpts_median"]
        lo = med - c["low_scale"] * (med - out.loc[m, "fpts_p10"])
        hi = med + c["high_scale"] * (out.loc[m, "fpts_p90"] - med)
        base = out.loc[m, "baseline"]
        blend = np.where(base.notna(), c["blend_weight"] * out.loc[m, "fpts"] + (1 - c["blend_weight"]) * base,
                         out.loc[m, "fpts"])
        shift = blend - out.loc[m, "fpts"]
        out.loc[m, "fpts"] = blend
        out.loc[m, "fpts_median"] = med + shift
        out.loc[m, "fpts_p10"] = np.maximum(lo + shift, -5)
        out.loc[m, "fpts_p90"] = hi + shift
    return out


def calibrate():
    """Pick range scaling and blend weights on 2016-2019; report 2020-2025 and this season."""
    m = pd.read_csv(os.path.join(OUT_DIR, "player_backtest.csv"))
    train = m[m.season.isin(TRAIN_SEASONS)]
    cal = {}
    for pos in POSITIONS:
        t = train[train.position == pos]
        # 1) blend weight with the season average (where there is one)
        tb = t[t.baseline.notna()]
        wts = np.arange(0, 1.001, 0.05)
        w = min(wts, key=lambda w: (w * tb.fpts + (1 - w) * tb.baseline - tb.fpts_actual).abs().mean())
        # 2) range widths around the blended projection
        shifted = apply_calibration(t, {pos: {"low_scale": 1.0, "high_scale": 1.0, "blend_weight": float(w)}})
        med = shifted.fpts_median
        grid = np.arange(0.4, 3.01, 0.02)
        low = min(grid, key=lambda a: abs(100 * (t.fpts_actual < med - a * (med - shifted.fpts_p10)).mean() - 10))
        high = min(grid, key=lambda b: abs(100 * (t.fpts_actual > med + b * (shifted.fpts_p90 - med)).mean() - 10))
        cal[pos] = {"low_scale": round(float(low), 2), "high_scale": round(float(high), 2), "blend_weight": round(float(w), 2)}
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(cal, f, indent=2)

    def rank_corr(x, col):
        r = [g[col].rank().corr(g.fpts_actual.rank()) for _, g in x.groupby(["season", "week"]) if len(g) >= 8]
        return np.mean(r) if r else np.nan

    rows = []
    for label, sel in [("2016-2019 (tuning)", list(TRAIN_SEASONS)), ("2020-2025 (test)", list(TEST_SEASONS)),
                       (f"{N.CURRENT_SEASON} weeks so far", [N.CURRENT_SEASON])]:
        x = m[m.season.isin(sel)]
        y = apply_calibration(x, cal)
        for pos in POSITIONS + ["all"]:
            a = x if pos == "all" else x[x.position == pos]
            b = y if pos == "all" else y[y.position == pos]
            has = a.baseline.notna()
            rows.append({"seasons": label, "pos": pos, "n": int(has.sum()),
                         "MAE_model": (a.fpts - a.fpts_actual)[has].abs().mean(),
                         "MAE_season_avg": (a.baseline - a.fpts_actual)[has].abs().mean(),
                         "MAE_blend": (b.fpts - b.fpts_actual)[has].abs().mean(),
                         "rank_model": np.mean([rank_corr(a[has & (a.position == q)], "fpts") for q in POSITIONS]) if pos == "all" else rank_corr(a[has], "fpts"),
                         "rank_season_avg": np.mean([rank_corr(a[has & (a.position == q)], "baseline") for q in POSITIONS]) if pos == "all" else rank_corr(a[has], "baseline"),
                         "rank_blend": np.mean([rank_corr(b[has & (b.position == q)], "fpts") for q in POSITIONS]) if pos == "all" else rank_corr(b[has], "fpts"),
                         "below_P10_before": 100 * (a.fpts_actual < a.fpts_p10).mean(),
                         "above_P90_before": 100 * (a.fpts_actual > a.fpts_p90).mean(),
                         "below_P10_after": 100 * (b.fpts_actual < b.fpts_p10).mean(),
                         "above_P90_after": 100 * (b.fpts_actual > b.fpts_p90).mean()})
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(OUT_DIR, "player_calibration_report.csv"), index=False)
    print("Calibration chosen on 2016-2019 (saved to output/player_calibration.json):")
    print(pd.DataFrame(cal).T.to_string())
    print("\nAccuracy (MAE, fantasy points) and weekly ranking within position (rank, 1 = perfect):")
    print(rep[["seasons", "pos", "n", "MAE_model", "MAE_season_avg", "MAE_blend", "rank_model", "rank_season_avg",
               "rank_blend"]].round(3).to_string(index=False))
    print("\nRanges: % of results below P10 and above P90 (target about 10% each):")
    print(rep[["seasons", "pos", "below_P10_before", "above_P90_before", "below_P10_after",
               "above_P90_after"]].round(1).to_string(index=False))


def main():
    if "--calibrate" in sys.argv:
        calibrate()
        return
    pw, tg, games = load()
    if "--backtest" in sys.argv:
        backtest(pw, tg, games)
    else:
        current(pw, tg, games)


if __name__ == "__main__":
    main()
