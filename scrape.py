"""Scrape NFL team stats from ESPN into the same layout as raw_data_csv.csv.

Usage:
    python3 scrape.py                 # current season -> raw_data_csv.csv
    python3 scrape.py 2025 out.csv    # a specific season / output file

Or from Python:
    from scrape import fetch_team_stats
    df = fetch_team_stats(season=2025)
"""
from __future__ import annotations

import json
import re
import sys

import pandas as pd
import requests

STATS_URL = "https://www.espn.com/nfl/stats/team/_/view/{view}"
STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/football/nfl/standings"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
REGULAR_SEASON = 2

# ESPN view -> (expected ESPN stat names in table order, our column names).
# Mapping is positional because some ESPN names repeat (e.g. "interceptions"
# appears for both takeaways and giveaways).
VIEWS = {
    "offense": (
        ["gamesPlayed", "netTotalYards", "netYardsPerGame", "netPassingYards",
         "netPassingYardsPerGame", "rushingYards", "rushingYardsPerGame",
         "totalPoints", "totalPointsPerGame"],
        ["Games", "OffTotalYds", "OffTotalYPG", "OffPassYds", "OffPassYPG",
         "OffRushYds", "OffRushYPG", "OffPoints", "OffPPG"],
    ),
    "defense": (
        ["gamesPlayed", "netTotalYards", "netYardsPerGame", "netPassingYards",
         "netPassingYardsPerGame", "rushingYards", "rushingYardsPerGame",
         "totalPoints", "totalPointsPerGame"],
        [None, "DefTotalYds", "DefTotalYPG", "DefPassYds", "DefPassYPG",
         "DefRushYds", "DefRushYPG", "DefPoints", "DefPPG"],
    ),
    "special": (
        ["gamesPlayed", "kickReturns", "kickReturnYards", "yardsPerKickReturn",
         "longKickReturn", "kickReturnTouchdowns", "puntReturns",
         "puntReturnYards", "yardsPerPuntReturn", "longPuntReturn",
         "puntReturnTouchdowns", "puntReturnFairCatches"],
        [None, "KickoffAtt", "KickoffYds", "KickoffAvg", "KickoffLong",
         "KickoffTD", "PuntAtt", "PuntYds", "PuntAvg", "PuntLong", "PuntTD",
         "PuntFC"],
    ),
    "turnovers": (
        ["gamesPlayed", "turnOverDifferential", "interceptions",
         "fumblesRecovered", "totalTakeaways", "interceptions", "fumblesLost",
         "totalGiveaways"],
        [None, "TurnoverDiff", "TakeawayInt", "TakeawayFum", "TakeawayTotal",
         "GiveawayInt", "GiveawayFum", "GiveawayTotal"],
    ),
}

COLUMN_ORDER = ["Team", "Games", "Wins"] + [
    c for _, cols in VIEWS.values() for c in cols if c and c != "Games"
]


def _get(url: str, params: dict | None = None) -> requests.Response:
    resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp


def _to_number(value: str) -> float:
    return float(str(value).replace(",", ""))


def _fetch_view(view: str, season: int | None, season_type: int) -> pd.DataFrame:
    url = STATS_URL.format(view=view)
    if season is not None:
        url += f"/season/{season}/seasontype/{season_type}"
    html = _get(url).text

    match = re.search(r"window\['__espnfitt__'\]=(\{.*?\});</script>", html, re.S)
    if not match:
        raise RuntimeError(f"Could not find stats data on {url} (page layout changed?)")
    team_stats = json.loads(match.group(1))["page"]["content"]["teamStats"]

    expected, columns = VIEWS[view]
    rows = []
    for entry in team_stats:
        names = [s["name"] for s in entry["stats"]]
        if names != expected:
            raise RuntimeError(f"Unexpected stat layout in '{view}' view: {names}")
        row = {"Team": entry["team"]["displayName"]}
        for col, stat in zip(columns, entry["stats"]):
            if col:
                row[col] = _to_number(stat["value"])
        rows.append(row)
    return pd.DataFrame(rows)


def _fetch_wins(season: int | None, season_type: int) -> pd.DataFrame:
    params = {"seasontype": season_type}
    if season is not None:
        params["season"] = season
    data = _get(STANDINGS_URL, params).json()

    rows = []
    for conference in data["children"]:
        for entry in conference["standings"]["entries"]:
            stats = {s["name"]: s.get("value") for s in entry["stats"]}
            rows.append({"Team": entry["team"]["displayName"], "Wins": stats["wins"]})
    return pd.DataFrame(rows)


def fetch_team_stats(season: int | None = None, season_type: int = REGULAR_SEASON) -> pd.DataFrame:
    """Return one row per NFL team with the columns of raw_data_csv.csv.

    season:      e.g. 2025 for the 2025-26 season; None = ESPN's current season.
    season_type: 2 = regular season (default), 3 = postseason.
    """
    df = _fetch_view("offense", season, season_type)
    for view in ("defense", "special", "turnovers"):
        df = df.merge(_fetch_view(view, season, season_type), on="Team", how="outer")
    df = df.merge(_fetch_wins(season, season_type), on="Team", how="left")

    df = df[COLUMN_ORDER].sort_values("Team").reset_index(drop=True)
    int_cols = [c for c in df.columns
                if c != "Team" and (df[c].dropna() % 1 == 0).all()]
    df[int_cols] = df[int_cols].astype("Int64")
    return df


def save_team_stats(path: str = "raw_data_csv.csv", season: int | None = None) -> pd.DataFrame:
    df = fetch_team_stats(season)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    season_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else "raw_data_csv.csv"
    result = save_team_stats(out_path, season_arg)
    print(f"Saved {len(result)} teams to {out_path}")
