"""Download and cache historical NFL games + per-game box scores from ESPN.

Everything is cached under ./data so each completed season is downloaded once:
    data/games_<season>.csv   one row per regular-season game
    data/box_<season>.csv     one row per team per game (box-score stats + spread)

Usage:
    python3 history.py 2012 2026     # download / update a range of seasons
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from games import REGULAR_SEASON, current_season, fetch_games
from scrape import _get

DATA_DIR = Path(__file__).parent / "data"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
FIRST_SEASON = 2012  # ESPN box scores are complete from here on
MAX_WORKERS = 8


# --------------------------------------------------------------------------
# Games (schedule + scores)
# --------------------------------------------------------------------------
def load_games(season: int, refresh: bool = False) -> pd.DataFrame:
    """Regular-season games for one season, cached once the season is over."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"games_{season}.csv"
    if path.exists() and not refresh:
        games = pd.read_csv(path, dtype={"GameID": str, "HomeID": str, "AwayID": str},
                            parse_dates=["Date"])
        if games["Completed"].all():
            return games
    games = fetch_games(season, season_types=(REGULAR_SEASON,))
    games.to_csv(path, index=False)
    return games


# --------------------------------------------------------------------------
# Box scores
# --------------------------------------------------------------------------
def _split(value: str, sep: str = "-") -> tuple[float, float]:
    try:
        a, b = str(value).split(sep)
        return float(a), float(b)
    except ValueError:
        return float("nan"), float("nan")


def _num(value) -> float:
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return float("nan")


def _line(data: dict) -> tuple[float, float]:
    """(home spread, over/under) from a summary response; spread < 0 = home favored."""
    for line in data.get("pickcenter") or []:
        if line.get("spread") is not None:
            return _num(line["spread"]), _num(line.get("overUnder"))
    return float("nan"), float("nan")


def fetch_line(game_id: str) -> float:
    """Current betting spread for one game (works for games not yet played)."""
    return _line(_get(SUMMARY_URL, {"event": game_id}).json())[0]


def fetch_lines(game_ids) -> dict[str, float]:
    """{GameID: home spread} for several games, fetched concurrently (NaN on failure)."""
    def safe(gid):
        try:
            return fetch_line(gid)
        except Exception:
            return float("nan")
    ids = [str(g) for g in game_ids]
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        return dict(zip(ids, pool.map(safe, ids)))


def fetch_boxscore(game_id: str) -> list[dict]:
    """Two rows (one per team) of box-score stats for a completed game."""
    data = _get(SUMMARY_URL, {"event": game_id}).json()
    spread, over_under = _line(data)

    rows = []
    for team in data.get("boxscore", {}).get("teams", []):
        stats = {s["name"]: s.get("displayValue") for s in team.get("statistics", [])}
        third_conv, third_att = _split(stats.get("thirdDownEff"))
        penalties, penalty_yds = _split(stats.get("totalPenaltiesYards"))
        sacks, sack_yds = _split(stats.get("sacksYardsLost"))
        rows.append({
            "GameID": str(game_id),
            "TeamID": str(team["team"]["id"]),
            "Plays": _num(stats.get("totalOffensivePlays")),
            "Yds": _num(stats.get("totalYards")),
            "PassYds": _num(stats.get("netPassingYards")),
            "RushYds": _num(stats.get("rushingYards")),
            "Turnovers": _num(stats.get("turnovers")),
            "FirstDowns": _num(stats.get("firstDowns")),
            "ThirdConv": third_conv,
            "ThirdAtt": third_att,
            "Penalties": penalties,
            "PenaltyYds": penalty_yds,
            "SacksTaken": sacks,
            "HomeSpread": spread,
            "OverUnder": over_under,
        })
    return rows


def load_boxscores(season: int, games: pd.DataFrame, progress=None) -> pd.DataFrame:
    """Box scores for every completed game in `games`, downloading only missing ones."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"box_{season}.csv"
    cached = (pd.read_csv(path, dtype={"GameID": str, "TeamID": str})
              if path.exists() else pd.DataFrame(columns=["GameID", "TeamID"]))

    wanted = set(games.loc[games["Completed"], "GameID"].astype(str))
    missing = sorted(wanted - set(cached["GameID"]))
    if not missing:
        return cached

    new_rows, failed = [], []
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = {gid: pool.submit(fetch_boxscore, gid) for gid in missing}
        for i, (gid, fut) in enumerate(futures.items(), 1):
            try:
                new_rows += fut.result()
            except Exception:
                failed.append(gid)
            if progress:
                progress(i, len(missing))
    if failed:
        print(f"  {season}: {len(failed)} box scores failed to download (will retry next run)")

    box = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
    box.to_csv(path, index=False)
    return box


def load_history(first: int = FIRST_SEASON, last: int | None = None,
                 verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(games, boxscores) for every season in [first, last]; last defaults to current."""
    last = last or current_season()
    all_games, all_box = [], []
    for season in range(first, last + 1):
        games = load_games(season, refresh=(season == last))
        box = load_boxscores(season, games)
        if verbose:
            print(f"{season}: {games['Completed'].sum()} games played, "
                  f"{box['GameID'].nunique()} box scores")
        all_games.append(games)
        all_box.append(box)
    return pd.concat(all_games, ignore_index=True), pd.concat(all_box, ignore_index=True)


if __name__ == "__main__":
    first_arg = int(sys.argv[1]) if len(sys.argv) > 1 else FIRST_SEASON
    last_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    load_history(first_arg, last_arg)
