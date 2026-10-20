"""
NFL data (nflverse, through the nflreadpy package), cached in ./data

  schedule()            every game since 1999: teams, scores, rest days, neutral sites and the closing
                        market spread and total (used only as a benchmark, never as a model input)
  team_games(seasons)   one row per team per game: offensive plays, EPA and points, from play-by-play

Past seasons are cached once; the current season is refreshed on every run.
"""

import datetime as dt
import os

import pandas as pd

DATA_DIR = "data"
CURRENT_SEASON = 2026


def _today_season():
    t = dt.date.today()
    return t.year if t.month >= 3 else t.year - 1


def schedule(refresh=True):
    """All games with results so far. Saved as data/schedule.csv as a fallback."""
    path = os.path.join(DATA_DIR, "schedule.csv")
    if refresh or not os.path.exists(path):
        try:
            import nflreadpy as nfl
            s = nfl.load_schedules().to_pandas()
            os.makedirs(DATA_DIR, exist_ok=True)
            s.to_csv(path, index=False)
        except Exception as e:
            if not os.path.exists(path):
                raise SystemExit(f"Couldn't get the NFL schedule ({e.__class__.__name__}).")
            print(f"Couldn't refresh the schedule ({e.__class__.__name__}); using the saved copy.")
    s = pd.read_csv(path, low_memory=False)
    s["gameday"] = pd.to_datetime(s.gameday)
    return s


def _team_games_season(season):
    """One season's play-by-play summed to team-game level (offense only: pass and run plays)."""
    import nflreadpy as nfl
    pbp = nfl.load_pbp([season]).to_pandas()
    plays = pbp[pbp.play_type.isin(["pass", "run"]) & pbp.epa.notna() & pbp.posteam.notna()].copy()
    live = plays.wp.between(0.1, 0.9)                    # game still in doubt (garbage time left out)
    is_pass = plays["pass"] == 1
    plays["epa_live"] = plays.epa.where(live, 0.0)
    plays["live"] = live.astype(int)
    plays["pass_epa"] = plays.epa.where(is_pass, 0.0)
    plays["rush_epa"] = plays.epa.where(~is_pass, 0.0)
    plays["pass_epa_live"] = plays.epa.where(is_pass & live, 0.0)
    plays["rush_epa_live"] = plays.epa.where(~is_pass & live, 0.0)
    plays["success_live"] = plays.success.where(live, 0.0)
    g = plays.groupby(["game_id", "posteam", "defteam"]).agg(
        plays=("epa", "size"), epa=("epa", "sum"),
        pass_plays=("pass", "sum"), success=("success", "sum"),
        live_plays=("live", "sum"), epa_live=("epa_live", "sum"),
        pass_epa=("pass_epa", "sum"), rush_epa=("rush_epa", "sum"),
        pass_epa_live=("pass_epa_live", "sum"), rush_epa_live=("rush_epa_live", "sum"),
        success_live=("success_live", "sum")).reset_index()
    g.insert(0, "season", season)
    return g.rename(columns={"posteam": "team", "defteam": "opp"})


def _qb_games_season(season):
    """One season's quarterback dropbacks (passes, sacks, scrambles) and EPA credited to the QB, per game."""
    import nflreadpy as nfl
    pbp = nfl.load_pbp([season]).to_pandas()
    d = pbp[(pbp.qb_dropback == 1) & pbp.id.notna() & pbp.qb_epa.notna()]
    g = d.groupby(["game_id", "posteam", "id"]).agg(
        qb_name=("name", "first"), dropbacks=("qb_epa", "size"), qb_epa=("qb_epa", "sum")).reset_index()
    g.insert(0, "season", season)
    return g.rename(columns={"posteam": "team", "id": "qb_id"})


def _cached(kind, builder, seasons):
    out = []
    current = _today_season()
    for season in seasons:
        path = os.path.join(DATA_DIR, f"{kind}_{season}.csv")
        if season >= current or not os.path.exists(path):
            try:
                builder(season).to_csv(path, index=False)
            except Exception as e:
                if not os.path.exists(path):
                    print(f"Couldn't get {season} play-by-play ({e.__class__.__name__}); skipping it.")
                    continue
                print(f"Couldn't refresh {season} play-by-play ({e.__class__.__name__}); using the saved copy.")
        out.append(pd.read_csv(path))
    return pd.concat(out, ignore_index=True)


ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
OUT_STATUSES = ("out", "doubtful", "injured reserve", "suspension", "suspended", "pup")


def name_key(s):
    """Normalized player name for matching across sources."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[.'`,]", "", s).replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def team_names():
    """Full team name -> abbreviation (e.g. 'Philadelphia Eagles' -> 'PHI'), cached."""
    path = os.path.join(DATA_DIR, "teams.csv")
    if not os.path.exists(path):
        import nflreadpy as nfl
        nfl.load_teams().to_pandas()[["team_abbr", "team_name"]].to_csv(path, index=False)
    t = pd.read_csv(path)
    return dict(zip(t.team_name, t.team_abbr))


def injury_report():
    """
    Current injury statuses from ESPN's public feed, keyed by (team abbreviation, name key).
    Used by both the spread model (starting QBs) and the player projections.
    """
    import requests
    try:
        r = requests.get(ESPN_INJURIES, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"Couldn't read the injury report ({e.__class__.__name__}); no injuries applied.")
        return {}
    abbr = team_names()
    out = {}
    for t in r.json().get("injuries", []):
        team = abbr.get(t.get("displayName", ""), "")
        for i in t.get("injuries", []):
            out[(team, name_key(i.get("athlete", {}).get("displayName", "")))] = str(i.get("status", ""))
    return out


def is_out(status):
    return str(status).lower().startswith(OUT_STATUSES)


PLAYER_COLS = ["player_id", "player_display_name", "position", "season", "week", "season_type", "game_id",
               "team", "opponent_team", "completions", "attempts", "passing_yards", "passing_tds",
               "passing_interceptions", "sacks_suffered", "passing_2pt_conversions", "carries", "rushing_yards",
               "rushing_tds", "rushing_2pt_conversions", "receptions", "targets", "receiving_yards", "receiving_tds",
               "receiving_2pt_conversions", "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost",
               "special_teams_tds", "fumble_recovery_tds"]


def _player_weekly_season(season):
    import nflreadpy as nfl
    ps = nfl.load_player_stats([season]).to_pandas()
    ps = ps[ps.position.isin(["QB", "RB", "FB", "WR", "TE"])]
    return ps[[c for c in PLAYER_COLS if c in ps.columns]]


def player_weekly(seasons):
    """Offensive players' box-score stats by game (players who recorded a stat)."""
    return _cached("player_weekly", _player_weekly_season, seasons)


def team_games(seasons):
    return _cached("team_games_v2", _team_games_season, seasons)


def qb_games(seasons):
    return _cached("qb_games", _qb_games_season, seasons)
