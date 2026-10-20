# NFL Projections: 2026

**[Live site](https://mcgahranjt.github.io/nfl-projections-site/)**, updated on Tuesdays, Thursdays and Sundays.

Weekly NFL game projections (spreads, totals and win probabilities) and fantasy point projections for quarterbacks, running backs, wide receivers and tight ends, using my league's scoring. Everything is built from free play-by-play and box-score data from [nflverse](https://github.com/nflverse), and every part of the model was tested on past seasons using only the information that would have been available at the time.

## Game projections

### Team ratings

Each team gets an offense rating and a defense rating, adjusted for the opponents it has played. They come from two measures:

- **Points** scored and allowed per game.
- **EPA** (expected points added) per game on pass and run plays. EPA measures how much each play changed a team's chances of scoring, so it is steadier from week to week than points.

The ratings are ridge regressions anchored to a preseason starting point: 60% of last season's final rating. That starting point counts as six games of evidence, so this season's results take over gradually. Both settings were chosen on 2006-2015, and the results were nearly the same across the range tested.

### Quarterback adjustment

A team's rating reflects the quarterbacks who played in the games behind it. When a different quarterback starts, the projection shifts by the difference between the starter and those quarterbacks, measured in EPA per dropback (recent seasons count more, and players with few dropbacks are pulled toward backup level), times 35 dropbacks. The starter is the one listed on the schedule, unless ESPN's injury report has him Out, Doubtful, on injured reserve or suspended. In that case the team's healthy quarterback with the most recent dropbacks starts instead.

### Spread and total

A small regression turns the ratings into a projected margin and total, with home field (about 1.7 points in recent seasons), rest days and the quarterback adjustment. Its weights come from the previous eight seasons only. Win probability assumes a game margin standard deviation of about 13 points.

### Results

Every game from 2016 through 2025 was projected using only earlier games, with the regression weights fit on the eight seasons before each one.

| | Average miss on the final margin (points) |
|---|---|
| Home team by the home-field edge | 10.98 |
| Model without the quarterback adjustment | 10.19 |
| **Model** | **10.13** |
| Closing market line | 9.78 |

The model beats the simple baseline by about 0.85 points a game but trails the closing line by about 0.35. The quarterback adjustment helped in seven of the ten seasons. I also tested removing garbage-time plays, splitting EPA into passing and rushing, adding success rate, weighting recent games more and anchoring offense and defense separately. The best combination improved the test seasons by only 0.05 points, so I kept the simpler model. Most of the remaining gap is probably information the market has and the model does not, such as injuries beyond the quarterback.

On 2026 weeks 1-6, replayed the same way, the model missed by 9.38 points against the market's 9.31.

## Fantasy projections

Each player's projection is team volume × his share × his efficiency:

1. **Team volume.** Pass attempts and carries from the team's recent games, pulled toward the league average and shifted by the projected margin, since teams that are ahead run more.
2. **Shares.** Each player's share of the team's targets and carries in recent games he played, with recent games counting more. Players listed as Out, Doubtful, on injured reserve or suspended are removed, and their shares go to teammates. Questionable players are projected as playing and flagged.
3. **Efficiency.** Catch rate, yards per target, yards per carry, touchdown and fumble rates, and the quarterback's completion, yardage, touchdown and interception rates. Each is pulled toward the position average when the sample is small, and yards are adjusted for the opponent's defense.
4. **Consistency with the game projection.** Receivers' catches and yards add up to the quarterback's passing line, and each team's touchdowns are scaled to its projected points.
5. **Simulation.** Each game is simulated 5,000 times, with team volume moving together for teammates. The league's scoring, including the tiered yardage bonuses, is applied to every simulated stat line, which gives the projection, a 10th-90th percentile range and the chance of reaching each bonus.
6. **Calibration.** By position, the projection is blended with the player's season average so far, and the ranges are widened or narrowed so that about 10% of results land below the 10th percentile and 10% above the 90th. Both were chosen on 2016-2019 and checked on 2020-2025.

Scoring lives in `league_scoring.json`, so a rule change only needs that file edited.

### Results

Average miss in fantasy points on 2020-2025, on the same player-games:

| Position | Model alone | Player's season average | **Model blended with season average** |
|---|---|---|---|
| QB | 7.70 | 8.25 | **7.61** |
| RB | 4.64 | 4.58 | **4.49** |
| WR | 4.85 | 4.79 | **4.73** |
| TE | 3.76 | 3.75 | **3.68** |

The blend also ranks players within each position slightly better than either method alone (a weekly rank correlation of 0.558, against 0.551 and 0.530). After calibration, 8-11% of results fall outside each end of the range at every position. These tests used each game's actual list of players who played, so a late scratch the injury report missed will make real results slightly worse.

## Grading

`nfl_grade.py` grades each week's saved projections after the games are played: spread and total error against the home-field baseline and the closing line, and fantasy error by position against each player's season average, plus how often results land outside the projected ranges. Only projections saved before kickoff count. The results feed the site's Grades tab. Live grading starts with week 7 of 2026.

## Running it

```
pip3 install --only-binary=:all: -r requirements.txt
python3 nfl_ratings.py --history     # once: past game projections used to fit the fantasy model
python3 run_weekly.py                # full run: spreads, fantasy, grading, website
python3 run_weekly.py --update       # spreads, fantasy and website with the latest injury report
```

The backtests:

```
python3 nfl_ratings.py --backtest    # spreads and totals, with and without the quarterback adjustment
python3 nfl_players.py --backtest    # fantasy, every week since 2016
python3 nfl_players.py --calibrate   # blend weights and range widths from the fantasy backtest
```

`data/qb_overrides.csv` (columns `team`, `qb_name`) sets a starting quarterback by hand.

The weekly routine runs through two macOS launchd jobs in `scheduling/`: a full run on Tuesdays at 10:00 AM Eastern, after Monday night's game is in the data, and updates on Thursdays at 4:00 PM and Sundays at 10:30 AM, after the week's injury reports. To turn them on:

```
cp scheduling/*.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.nflprojections.full.plist ~/Library/LaunchAgents/com.nflprojections.update.plist
```

`build_site.py --publish` pushes the site to a local clone of [nfl-projections-site](https://github.com/mcgahranjt/nfl-projections-site), which GitHub Pages serves.

**Files:**
- `nfl_data.py`: downloads and caches schedules, play-by-play summaries, player stats and the injury report
- `nfl_ratings.py`: team ratings, the quarterback adjustment, spreads and totals
- `nfl_players.py`: player and fantasy projections
- `nfl_grade.py`: grading
- `run_weekly.py`, `build_site.py`, `site_template.html`: the weekly routine and the website
- `output/`: projections, backtest results and grades

## Data

- Schedules, scores, play-by-play with EPA and weekly player stats: [nflverse](https://github.com/nflverse) through the `nflreadpy` package, 2005 onward
- Injury statuses: ESPN's public injury feed
- Closing market lines (used only to grade the model): nflverse schedules
