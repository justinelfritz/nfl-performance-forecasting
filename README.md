# NFL Rankings

A Streamlit dashboard for ranking NFL teams and forecasting games, built on data pulled from ESPN.

It does three things:

- **Rankings.** Rank all 32 teams by a weighted score. You choose the stats and set the weights.
- **Compare teams.** Put any two teams side by side: stats, league ranks, schedule strength, results, head-to-head and common opponents.
- **Model forecasts.** A model learns its own weights from game history and predicts every game: the winner, the margin and the win probability. It has been tested week by week on past seasons and compared with the Vegas line.

## Quick start

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

The first time you open the **Model forecasts** view, it downloads game history from 2012 to today into `data/`. This takes a few minutes. After that, only new games are downloaded.

## The dashboard

### Sidebar

- **Source**
  - **Live from ESPN:** fetches team stats for the chosen season.
  - **Local CSV:** uses `raw_data_csv.csv`.
- **Refresh:** fetches fresh data from ESPN.
- **Save to CSV:** writes the current data to `raw_data_csv.csv`.
- **Load game results:** turns on the schedule metrics (SOS, SOV, OppMargin, RemainingSOS) and the schedules and results on Compare teams.
- **Scoring inputs and Weights:** choose which stats count toward the score and how much each counts. Stats marked ↓ are better when lower (yards and points allowed, for example). Their sign is flipped before scoring, so a positive weight always rewards the team that is better at that stat.

These settings affect only the **Rankings** view. The model in **Model forecasts** learns its own weights.

### Rankings

Each stat is standardized across the league, combined using your weights and rescaled to a score from 0 to 10.

### Compare teams

- **Statistics table:** each team's value and league rank for every stat, and which team has the edge.
- **League-average chart:** how far each team is above or below average on each scoring input, in standard deviations, with its league rank at the end of each bar.
- **Schedule:** strength-of-schedule metrics and a game-by-game log.
- **Head-to-head:** results of the two teams' games against each other.
- **Common opponents:** how each team did against opponents they both played.

### Model forecasts

- **Forecast table:** for each game of the chosen week:
  - the pick, predicted margin and win probability;
  - the Vegas line, and which side the model takes against it;
  - for completed games, the result, whether the pick was right, how far off the margin was, and whether the pick covered the spread (ATS).

  Click 📈 on a row to see the chance the pick wins by more than any given number of points, for both the model and the Vegas line. Click anywhere else in the row to open the two teams in Compare teams.
- **Power rankings and learned weights:** each team's rating going into the chosen week, and how much the model weights each stat.
- **Backtest:**
  - pick accuracy, margin error (MAE), Brier score and log-loss against simple baselines, from a week-by-week test on 2018–2025;
  - the record against the spread;
  - **Win probability calibration:** whether favorites given X% actually won X% of the time, for the model and the Vegas line, overall and for each part of the season.

## How the model works

1. **Stats as of each week.** For every team and week, stats are computed only from games played earlier in that season. Three adjustments are made:
   - **Opponent adjustment:** ratings for margin, points and yards per play account for how strong each opponent was, in the style of a simple rating system (SRS).
   - **Recency weighting:** recent games can count more than older ones. The tuned setting weights all games equally.
   - **Prior season for weeks 1–4 only:** early-season stats are blended with the team's final stats from last season. From week 5 on, only current-season data is used.
2. **Standardize.** Each stat is centered on that week's league average and scaled by its typical spread. Stats where lower is better are flipped. Small samples are pulled toward the league average.
3. **Fit.** A ridge regression with non-negative weights predicts the home team's margin from the difference between the two teams' standardized stats, plus home-field advantage. The learned weights turn a team's stats into a rating in points against an average team.
4. **Win probability.** P(win) = Φ(predicted margin / σ), where Φ is the normal distribution's cumulative probability. σ is the model's typical miss, about 13 points.
5. **Walk-forward.** Week *w* of season *S* is predicted by a model fit only on games completed before that week. Nothing from the future is used.

The model's settings (recency decay, shrinkage, prior-season weight, ridge strength) were tuned on 2015–2017 and tested on 2018–2025.

### Results (2018–2025 walk-forward test, 2,127 games)

| | Pick accuracy | Margin error (MAE) | Brier score |
|---|---|---|---|
| Model | 64.1% | 10.19 pts | 0.222 |
| Always pick the home team | 53.9% | 11.06 pts | 0.248 |
| Vegas line (1,419 games with a line) | 64.7% | 9.93 pts | 0.211 |

On those same 1,419 games, the model scores 63.2% accuracy, 10.27 pts MAE and a Brier score of 0.223.

The win probabilities are well calibrated: for nearly every probability range, the favorites' actual win rate is within normal random variation of the prediction. The model is close to, but a little behind, the Vegas line. Its record against the spread (50.8% overall) is below the 52.4% needed to break even at standard −110 odds. **It is not a betting system.**

The numbers above come from `python3 model.py backtest`. They change when the model is retuned or more data is added.

## Command line

```bash
python3 model.py              # rankings + forecasts for the next week
python3 model.py tune         # tune the model's settings on 2015-2017
python3 model.py backtest     # walk-forward test on 2018-2025, writes data/backtest_predictions.csv
python3 model.py ablation     # how much each model improvement contributes

python3 history.py 2012 2026  # download or update games + box scores into data/
python3 scrape.py 2025 out.csv        # one season's team stats (default: current season -> raw_data_csv.csv)
python3 games.py 2020 2024 out.csv    # game results + strength of schedule
```

## Project layout

| File | Purpose |
|---|---|
| `dashboard.py` | The Streamlit app |
| `model.py` | Feature building, model fitting, tuning, backtesting, rankings and forecasts |
| `history.py` | Downloads and caches games, box scores and betting lines into `data/` |
| `games.py` | Game results, team game logs and strength-of-schedule metrics |
| `scrape.py` | Season team stats from ESPN |
| `process.py` | The original one-off scoring script the dashboard grew from |
| `raw_data_csv.csv` | One season of team stats, used by the "Local CSV" source |
| `data/` | Cached games (`games_<season>.csv`), box scores (`box_<season>.csv`), tuned settings (`model_params.json`), tuning and backtest results |

## Data notes

- Data comes from ESPN's public site API, which is undocumented and can change without notice.
- Box scores are complete from 2012 on, which is why the history starts there.
- Betting lines come from ESPN and are missing for much of 2023, all of 2024 and about two-thirds of 2025. The Vegas comparisons use only games that have a line.
- Lines for upcoming games are fetched live and cached for 30 minutes.
