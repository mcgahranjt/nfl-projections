"""
Weekly routine

Runs the scripts in order and saves everything they print to logs/<date>.log.

  Full run (Tuesday):            ratings and spreads -> fantasy projections -> grading -> website
  Update run (Thursday, Sunday): ratings and spreads -> fantasy projections -> website
                                 (picks up the latest injury report and starting QBs)

If the ratings fail, the run stops, since the fantasy projections depend on that week's game
projections. Any other failure is logged and the run continues.

Usage:
  python3 run_weekly.py            # full run
  python3 run_weekly.py --update   # Thursday/Sunday update
"""

import datetime as dt
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
FULL = [("Ratings and spreads", ["nfl_ratings.py"], True),
        ("Fantasy projections", ["nfl_players.py"], False),
        ("Grading", ["nfl_grade.py"], False),
        ("Website", ["build_site.py", "--publish"], False)]
UPDATE = [("Ratings and spreads", ["nfl_ratings.py"], True),
          ("Fantasy projections", ["nfl_players.py"], False),
          ("Website", ["build_site.py", "--publish"], False)]


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    steps = UPDATE if "--update" in sys.argv else FULL
    started = dt.datetime.now()
    summary = []
    with open(os.path.join(LOG_DIR, f"{started:%Y-%m-%d}.log"), "a", encoding="utf-8") as log:
        def out(text):
            print(text)
            log.write(text + "\n")
            log.flush()

        out(f"\n{'=' * 70}\n{'Update run' if steps is UPDATE else 'Full run'} started {started:%Y-%m-%d %H:%M}")
        for name, args, required in steps:
            out(f"\n--- {name}")
            p = subprocess.run([sys.executable, "-W", "ignore"] + args, cwd=HERE, capture_output=True, text=True)
            out((p.stdout + p.stderr).rstrip())
            status = "ok" if p.returncode == 0 else ("no games this week" if "No upcoming games" in p.stdout + p.stderr
                                                     else f"FAILED (exit code {p.returncode})")
            summary.append((name, status))
            if required and status != "ok":
                out(f"\n{name} did not finish, so the rest of the run was skipped.")
                break
        secs = int((dt.datetime.now() - started).total_seconds())
        out(f"\nSummary ({secs // 60} min {secs % 60} sec):")
        for name, status in summary:
            out(f"  {name:<22} {status}")
    if any(s.startswith("FAILED") for _, s in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
