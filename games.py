"""Scrape NFL game results from ESPN and compute strength-of-schedule metrics.

Usage:
    python3 games.py                    # current season -> games.csv + prints SOS
    python3 games.py 2020 2024 out.csv  # range of seasons -> out.csv

From Python:
    from games import fetch_games, team_game_log, strength_of_schedule
    games = fetch_games(2024)                     # one row per game
    log = team_game_log(games)                    # one row per team per game
    sos = strength_of_schedule(games, through_week=10)   # no look-ahead
"""
from __future__ import annotations

import sys
from collections.abc import Iterable

import pandas as pd

from scrape import _get

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
REGULAR_SEASON, POSTSEASON = 2, 3
SKIP_WEEK_LABELS = {"Pro Bowl"}  # all-star game, not real teams


# --------------------------------------------------------------------------
# Scraping
# --------------------------------------------------------------------------
def _scoreboard(season: int | None, season_type: int | None = None, week: int | None = None) -> dict:
    params = {"limit": 1000}
    if season is not None:
        params["dates"] = season
    if season_type is not None:
        params["seasontype"] = season_type
    if week is not None:
        params["week"] = week
    return _get(SCOREBOARD_URL, params).json()


def current_season() -> int:
    """The season ESPN currently treats as active (e.g. 2026 for 2026-27)."""
    return int(_scoreboard(None)["season"]["year"])


def _weeks(season: int, season_type: int) -> list[int]:
    calendar = _scoreboard(season, season_type, 1)["leagues"][0]["calendar"]
    for block in calendar:
        if int(block["value"]) == season_type:
            return [int(e["value"]) for e in block["entries"]
                    if e["label"] not in SKIP_WEEK_LABELS]
    return []


def _parse_event(event: dict, season: int, season_type: int, week: int) -> dict:
    comp = event["competitions"][0]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    home, away = sides["home"], sides["away"]
    completed = bool(comp["status"]["type"]["completed"])

    def score(side: dict) -> float | None:
        return float(side["score"]) if completed and side.get("score") not in (None, "") else None

    return {
        "Season": season,
        "SeasonType": season_type,
        "Week": week,
        "Date": pd.to_datetime(event["date"], utc=True),
        "GameID": event["id"],
        "HomeTeam": home["team"]["displayName"],
        "AwayTeam": away["team"]["displayName"],
        "HomeID": str(home["team"]["id"]),
        "AwayID": str(away["team"]["id"]),
        "HomeScore": score(home),
        "AwayScore": score(away),
        "Completed": completed,
        "NeutralSite": bool(comp.get("neutralSite", False)),
    }


def fetch_games(
    seasons: int | Iterable[int] | None = None,
    season_types: Iterable[int] = (REGULAR_SEASON,),
) -> pd.DataFrame:
    """Return one row per game (played and scheduled).

    seasons:      a year, a list of years, or None for the current season.
    season_types: 2 = regular season, 3 = postseason; e.g. (2, 3) for both.
    """
    if seasons is None:
        seasons = [current_season()]
    elif isinstance(seasons, int):
        seasons = [seasons]

    rows = []
    for season in seasons:
        for season_type in season_types:
            for week in _weeks(season, season_type):
                data = _scoreboard(season, season_type, week)
                rows += [_parse_event(e, season, season_type, week) for e in data.get("events", [])]

    games = pd.DataFrame(rows)
    if games.empty:
        return games
    games = games.drop_duplicates("GameID")
    for col in ("HomeScore", "AwayScore"):
        games[col] = games[col].astype("Int64")
    return games.sort_values(["Season", "SeasonType", "Week", "Date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Reshaping + metrics
# --------------------------------------------------------------------------
def team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: one from each team's point of view."""
    base = ["Season", "SeasonType", "Week", "Date", "GameID", "Completed", "NeutralSite"]
    home = games[base].assign(
        Team=games["HomeTeam"], Opponent=games["AwayTeam"], Home=~games["NeutralSite"],
        PointsFor=games["HomeScore"], PointsAgainst=games["AwayScore"],
    )
    away = games[base].assign(
        Team=games["AwayTeam"], Opponent=games["HomeTeam"], Home=False,
        PointsFor=games["AwayScore"], PointsAgainst=games["HomeScore"],
    )
    log = pd.concat([home, away], ignore_index=True)
    log["Margin"] = log["PointsFor"] - log["PointsAgainst"]
    log["Result"] = pd.NA
    played = log["Completed"]
    log.loc[played, "Result"] = log.loc[played, "Margin"].map(
        lambda m: "W" if m > 0 else ("L" if m < 0 else "T")
    )
    return log.sort_values(["Team", "Season", "SeasonType", "Week"]).reset_index(drop=True)


def strength_of_schedule(
    games: pd.DataFrame,
    season: int | None = None,
    through_week: int | None = None,
) -> pd.DataFrame:
    """Per-team schedule-difficulty metrics for one regular season.

    Only games completed in weeks <= through_week count as "played", so passing
    through_week gives the metrics exactly as they stood at that point (useful
    for backtesting without look-ahead).

    Columns:
      W, L, T, GamesPlayed  record in the counted games
      SOS            NFL-style strength of schedule: combined win% of every
                     opponent faced (each meeting counted), ties = half a win
      SOV            strength of victory: combined win% of opponents beaten
      OppMargin      average per-game point margin of opponents faced
      RemainingSOS   combined win% of opponents still to be played
    """
    g = games[games["SeasonType"] == REGULAR_SEASON]
    if season is None:
        season = int(g["Season"].max())
    g = g[g["Season"] == season]

    log = team_game_log(g)
    if through_week is not None:
        cutoff = log["Week"] > through_week
        log.loc[cutoff, ["Completed", "Result"]] = [False, pd.NA]
    played = log[log["Completed"]]
    remaining = log[~log["Completed"]]

    record = (
        played.assign(W=played["Result"].eq("W"), L=played["Result"].eq("L"), T=played["Result"].eq("T"))
        .groupby("Team")[["W", "L", "T"]].sum()
    )
    record["GamesPlayed"] = record.sum(axis=1)
    record["WinPct"] = (record["W"] + 0.5 * record["T"]) / record["GamesPlayed"]
    record["Margin"] = played.groupby("Team")["Margin"].mean()

    def combined_win_pct(opponents: pd.Series) -> float:
        r = record.reindex(opponents).fillna(0)
        n = r["GamesPlayed"].sum()
        return (r["W"].sum() + 0.5 * r["T"].sum()) / n if n else float("nan")

    teams = sorted(set(log["Team"]))
    out = pd.DataFrame(index=pd.Index(teams, name="Team"))
    out = out.join(record[["W", "L", "T", "GamesPlayed"]]).fillna(0).astype(int)
    out["SOS"] = played.groupby("Team")["Opponent"].apply(combined_win_pct)
    out["SOV"] = played[played["Result"] == "W"].groupby("Team")["Opponent"].apply(combined_win_pct)
    out["OppMargin"] = played.groupby("Team")["Opponent"].apply(
        lambda opp: record["Margin"].reindex(opp).mean()
    )
    out["RemainingSOS"] = remaining.groupby("Team")["Opponent"].apply(combined_win_pct)
    out.insert(0, "Season", season)
    return out.round(4).reset_index()


if __name__ == "__main__":
    args = sys.argv[1:]
    out_path = args.pop() if args and args[-1].endswith(".csv") else "games.csv"
    years = list(range(int(args[0]), int(args[-1]) + 1)) if args else None

    all_games = fetch_games(years, season_types=(REGULAR_SEASON, POSTSEASON))
    all_games.to_csv(out_path, index=False)
    print(f"Saved {len(all_games)} games to {out_path}")
    print(strength_of_schedule(all_games).sort_values("SOS", ascending=False).to_string(index=False))
