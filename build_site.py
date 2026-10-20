"""
Projections website

Builds a one-page website from the files the other scripts already write (nothing is re-estimated):
  This Week  - output/game_projections.csv and qb_notes.json, with each team's fantasy projections
  Fantasy    - the latest output/player_projections_week<N>.csv
  Rankings   - output/team_ratings.csv
  Grades     - output/grades_summary.json
The page itself is site_template.html; this script fills in the data and writes site/index.html.

With --publish, the page is copied into a local clone of the GitHub Pages repository (SITE_REPO)
and pushed. Until that clone exists, --publish only builds.

Usage:
  python3 build_site.py             # build site/index.html
  python3 build_site.py --publish   # build, then commit and push to GitHub Pages
"""

import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TEMPLATE = os.path.join(HERE, "site_template.html")
SITE_DIR = os.path.join(HERE, "site")
SITE_REPO = os.path.expanduser("~/Documents/NFLProj/nfl-projections-site")


def records(df):
    return json.loads(df.to_json(orient="records"))


def gather():
    gp = pd.read_csv(os.path.join(OUT, "game_projections.csv"))
    files = glob.glob(os.path.join(OUT, "player_projections_week*.csv"))
    week_of = lambda f: int(re.search(r"week(\d+)", f).group(1))
    week = int(gp.week.iloc[0])
    pfile = os.path.join(OUT, f"player_projections_week{week}.csv")
    if not os.path.exists(pfile) and files:
        pfile = max(files, key=week_of)
    players = pd.read_csv(pfile) if os.path.exists(pfile) else pd.DataFrame()
    if len(players):
        home = dict(zip(gp.home_team, [True] * len(gp)))
        players["home"] = players.team.map(home).fillna(False)
        players["status"] = players.status.fillna("")
        players["player_id"] = players.player_id.astype(str)
    games = [{"id": r.game_id, "day": r.gameday, "away": r.away_team, "home": r.home_team,
              "awayQB": r.away_qb, "homeQB": r.home_qb, "awayAdj": r.qb_away, "homeAdj": r.qb_home,
              "awayPts": r.proj_away, "homePts": r.proj_home, "margin": r.proj_margin, "total": r.proj_total,
              "hwp": r.home_win_prob, "neutral": int(r.neutral)} for r in gp.sort_values(["gameday", "game_id"]).itertuples()]
    rank = pd.read_csv(os.path.join(OUT, "team_ratings.csv"))
    notes_path, grades_path = os.path.join(OUT, "qb_notes.json"), os.path.join(OUT, "grades_summary.json")
    return {"week": week, "built": dt.datetime.now().strftime("%b %-d, %-I:%M %p"), "games": games,
            "players": records(players) if len(players) else [], "rank": records(rank),
            "qbNotes": json.load(open(notes_path)) if os.path.exists(notes_path) else [],
            "grades": json.load(open(grades_path)) if os.path.exists(grades_path) else {}}


def build():
    data = gather()
    html = open(TEMPLATE, encoding="utf-8").read()
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    os.makedirs(SITE_DIR, exist_ok=True)
    path = os.path.join(SITE_DIR, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Built {path}: week {data['week']}, {len(data['games'])} games, {len(data['players'])} players.")
    return path


def git(*args):
    return subprocess.run(["git", "-C", SITE_REPO] + list(args), capture_output=True, text=True)


def publish(path):
    if not os.path.isdir(os.path.join(SITE_REPO, ".git")):
        print(f"Not published: no GitHub Pages repository at {SITE_REPO} yet.")
        return True
    shutil.copy(path, os.path.join(SITE_REPO, "index.html"))
    git("add", "index.html")
    if git("diff", "--cached", "--quiet").returncode == 0:
        print("Website unchanged since the last publish.")
        return True
    c = git("commit", "-m", f"Update projections {dt.datetime.now():%Y-%m-%d %H:%M}")
    p = git("push", "origin", "HEAD")
    if c.returncode or p.returncode:
        print("Couldn't publish the website:\n" + (c.stdout + c.stderr + p.stderr).strip()[-600:])
        return False
    print("Published to GitHub Pages (live in a minute or two).")
    return True


def main():
    path = build()
    if "--publish" in sys.argv and not publish(path):
        sys.exit(1)


if __name__ == "__main__":
    main()
