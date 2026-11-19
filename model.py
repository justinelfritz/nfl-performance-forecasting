"""Learned NFL power rankings + game forecasts.

Approach
--------
* For every team and week, compute season-to-date stats from games played
  earlier that season (never later games), with three refinements:
    1. Opponent-adjusted ratings: least-squares team ratings that credit each
       team for the strength of the opponents it played (SRS-style), for point
       margin, points scored/allowed and yards per play.
    2. Recency weighting: a game played j weeks ago counts decay**j.
    3. Prior season for weeks 1-4 only: early-season values are blended with
       the team's final values from last season; from week 5 on, current-season
       data only.
* Center each stat on that week's league average, scale by its typical spread,
  and flip "lower is better" stats so higher = better. Each team's value is
      (n * current + p * prior) / (n + p + k)
  where n = recency-weighted games played, p = prior-season weight (weeks 1-4
  only, else 0) and k shrinks small samples toward league average.
* Model: home margin = HFA * home_field + sum_i w_i * (z_home_i - z_away_i),
  ridge regression with w_i >= 0. The w_i are the learned ranking weights;
  a team's rating sum_i w_i * z_i is in points vs an average team.
* Win probability = Phi(predicted margin / sigma).
* Walk-forward: week w of season S is predicted by a model fit only on games
  completed before week w of season S.

CLI:
    python3 model.py tune        # choose decay / k / p / alpha on 2015-2017
    python3 model.py backtest    # walk-forward test on 2018-2025
    python3 model.py ablation    # contribution of each improvement
    python3 model.py predict     # rankings + forecasts for the next week
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, Ridge

from history import DATA_DIR, FIRST_SEASON, load_history

BASE_FEATURES = [
    "WinPct", "OffPPG", "DefPPG", "OffTotalYPG", "DefTotalYPG",
    "OffPassYPG", "OffRushYPG", "DefPassYPG", "DefRushYPG",
    "OffYPP", "DefYPP", "TurnoverDiff", "SOS", "OppMargin",
]
ADJUSTED_FEATURES = ["AdjMargin", "AdjOff", "AdjDef", "AdjOffYPP", "AdjDefYPP"]
FEATURES = BASE_FEATURES + ADJUSTED_FEATURES
LOWER_IS_BETTER = {"DefPPG", "DefTotalYPG", "DefPassYPG", "DefRushYPG", "DefYPP",
                   "AdjDef", "AdjDefYPP"}

# The dashboard's hand-picked defaults, used as a benchmark
HAND_WEIGHTS = {"WinPct": 0.8, "OffTotalYPG": 0.5, "OffPPG": 0.7, "DefTotalYPG": 0.5,
                "DefPPG": 0.7, "TurnoverDiff": 0.3, "SOS": 0.5}

PRIOR_WEEKS = 4          # prior-season data is used only for weeks 1..PRIOR_WEEKS
RATING_RIDGE = 1.0       # regularization (in games) for the opponent-adjusted solves
VALIDATION_SEASONS = [2015, 2016, 2017]
TEST_SEASONS = list(range(2018, 2026))
PARAMS_PATH = DATA_DIR / "model_params.json"
DEFAULT_PARAMS = {"decay": 0.9, "k": 8.0, "p": 4.0, "alpha": 100.0}
# The first version of the model (no adjusted stats, no recency, no prior)
PREVIOUS_PARAMS = {"decay": 1.0, "k": 12.0, "p": 0.0, "alpha": 100.0}


# --------------------------------------------------------------------------
# Raw data
# --------------------------------------------------------------------------
def team_games(games: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """One row per team per completed game with own + opponent box-score stats."""
    g = games[games["Completed"].astype(bool)].copy()
    base = ["Season", "Week", "GameID"]
    home_flag = (~g["NeutralSite"].astype(bool)).astype(float)
    home = g[base].assign(TeamID=g["HomeID"], OppID=g["AwayID"], Team=g["HomeTeam"],
                          PF=g["HomeScore"], PA=g["AwayScore"], HomeFlag=home_flag)
    away = g[base].assign(TeamID=g["AwayID"], OppID=g["HomeID"], Team=g["AwayTeam"],
                          PF=g["AwayScore"], PA=g["HomeScore"], HomeFlag=-home_flag)
    tg = pd.concat([home, away], ignore_index=True)
    tg[["PF", "PA"]] = tg[["PF", "PA"]].astype(float)
    tg["Win"] = np.sign(tg["PF"] - tg["PA"]) * 0.5 + 0.5  # 1 / 0.5 / 0

    stats = ["Plays", "Yds", "PassYds", "RushYds", "Turnovers"]
    own = box[["GameID", "TeamID"] + stats]
    opp = own.rename(columns={"TeamID": "OppID", **{s: "Opp" + s for s in stats}})
    return tg.merge(own, on=["GameID", "TeamID"], how="left").merge(
        opp, on=["GameID", "OppID"], how="left")


@dataclass
class Raw:
    games: pd.DataFrame
    tg: pd.DataFrame


def load_raw(first: int = FIRST_SEASON, last: int | None = None, verbose: bool = False) -> Raw:
    games, box = load_history(first, last, verbose=verbose)
    spreads = box.groupby("GameID")["HomeSpread"].first()
    games = games.merge(spreads, left_on="GameID", right_index=True, how="left")
    return Raw(games, team_games(games, box))


# --------------------------------------------------------------------------
# Opponent-adjusted ratings
# --------------------------------------------------------------------------
def _ridge_solve(A: np.ndarray, y: np.ndarray, w: np.ndarray, penalized: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)[:, None]
    reg = np.sqrt(RATING_RIDGE) * np.diag(penalized.astype(float))[penalized]
    lhs = np.vstack([A * sw, reg])
    rhs = np.r_[y * sw[:, 0], np.zeros(len(reg))]
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def adjusted_ratings(prior: pd.DataFrame, wt: np.ndarray, teams: pd.Index) -> pd.DataFrame:
    """Opponent- and home-adjusted ratings from team-game rows (both sides of each game)."""
    T = len(teams)
    pos = pd.Series(np.arange(T), index=teams)
    i = pos.reindex(prior["TeamID"]).to_numpy()
    j = pos.reindex(prior["OppID"]).to_numpy()
    home = prior["HomeFlag"].to_numpy()
    rows = np.arange(len(prior))
    out = pd.DataFrame(index=teams)

    # Margin: PF - PA = r_i - r_j + h * home
    A = np.zeros((len(prior), T + 1))
    A[rows, i] += 1
    A[rows, j] -= 1
    A[:, T] = home
    sol = _ridge_solve(A, (prior["PF"] - prior["PA"]).to_numpy(), wt, np.r_[np.ones(T, bool), False])
    out["AdjMargin"] = sol[:T]

    # Offense/defense: stat = mu + off_i + def_j + h * home  (def = amount allowed above average)
    def off_def(y: np.ndarray, mask: np.ndarray):
        A = np.zeros((mask.sum(), 2 * T + 2))
        r = np.arange(mask.sum())
        A[:, 0] = 1
        A[r, 1 + i[mask]] = 1
        A[r, 1 + T + j[mask]] = 1
        A[:, -1] = home[mask]
        pen = np.r_[False, np.ones(2 * T, bool), False]
        s = _ridge_solve(A, y[mask], wt[mask], pen)
        return s[1:T + 1], s[T + 1:2 * T + 1]

    out["AdjOff"], out["AdjDef"] = off_def(prior["PF"].to_numpy(), np.ones(len(prior), bool))
    ypp = (prior["Yds"] / prior["Plays"]).to_numpy()
    out["AdjOffYPP"], out["AdjDefYPP"] = off_def(ypp, np.isfinite(ypp))
    return out


# --------------------------------------------------------------------------
# Weekly features
# --------------------------------------------------------------------------
def weekly_features(raw: Raw, decay: float) -> pd.DataFrame:
    """Recency-weighted, season-to-date features entering every week, plus a
    'final' row (week = last week + 1) holding each team's end-of-season values."""
    games, tg = raw.games, raw.tg
    out = []
    for season, sg in games.groupby("Season"):
        teams = pd.concat([
            sg[["HomeID", "HomeTeam"]].set_axis(["TeamID", "Team"], axis=1),
            sg[["AwayID", "AwayTeam"]].set_axis(["TeamID", "Team"], axis=1),
        ]).drop_duplicates("TeamID", keep="last").set_index("TeamID")["Team"]
        stg = tg[tg["Season"] == season]
        weeks = sorted(sg["Week"].unique())
        for week in weeks + [weeks[-1] + 1]:
            prior = stg[stg["Week"] < week]
            wt = decay ** (week - 1 - prior["Week"].to_numpy(float))
            cols = ["Win", "PF", "PA", "Yds", "PassYds", "RushYds", "Plays", "Turnovers",
                    "OppYds", "OppPassYds", "OppRushYds", "OppPlays", "OppTurnovers"]
            weighted = prior[cols].mul(wt, axis=0).assign(TeamID=prior["TeamID"].values, Wt=wt)
            agg = weighted.groupby("TeamID").sum().reindex(teams.index)
            agg["N"] = prior.groupby("TeamID").size().reindex(teams.index)
            n = agg["Wt"]

            f = pd.DataFrame(index=teams.index)
            f["WinPct"] = 100 * agg["Win"] / n
            f["OffPPG"], f["DefPPG"] = agg["PF"] / n, agg["PA"] / n
            f["OffTotalYPG"], f["DefTotalYPG"] = agg["Yds"] / n, agg["OppYds"] / n
            f["OffPassYPG"], f["OffRushYPG"] = agg["PassYds"] / n, agg["RushYds"] / n
            f["DefPassYPG"], f["DefRushYPG"] = agg["OppPassYds"] / n, agg["OppRushYds"] / n
            f["OffYPP"], f["DefYPP"] = agg["Yds"] / agg["Plays"], agg["OppYds"] / agg["OppPlays"]
            f["TurnoverDiff"] = (agg["OppTurnovers"] - agg["Turnovers"]) / n

            if len(prior):
                opp = prior[["TeamID", "OppID"]].merge(
                    agg[["Win", "Wt"]].assign(Margin=(agg["PF"] - agg["PA"]) / n),
                    left_on="OppID", right_index=True)
                grp = opp.groupby("TeamID")
                f["SOS"] = grp["Win"].sum() / grp["Wt"].sum()
                f["OppMargin"] = grp["Margin"].mean()
                f = f.join(adjusted_ratings(prior, wt, teams.index))
                f.loc[agg["N"].isna(), ADJUSTED_FEATURES] = np.nan
            else:
                f["SOS"] = f["OppMargin"] = np.nan
                f[ADJUSTED_FEATURES] = np.nan

            f.insert(0, "Neff", n.fillna(0.0))
            f.insert(0, "N", agg["N"].fillna(0).astype(int))
            f.insert(0, "Final", week > weeks[-1])
            f.insert(0, "Team", teams)
            f.insert(0, "Week", week)
            f.insert(0, "Season", season)
            out.append(f.reset_index())
    fw = pd.concat(out, ignore_index=True)

    # Center on that week's league average; flip lower-is-better so higher = better
    centered = fw.groupby(["Season", "Week"])[FEATURES].transform(lambda s: s - s.mean())
    for col in FEATURES:
        fw["c_" + col] = (-1.0 if col in LOWER_IS_BETTER else 1.0) * centered[col]

    # Last season's final (centered) values for the same franchise
    final = fw[fw["Final"]][["Season", "TeamID"] + ["c_" + c for c in FEATURES]].copy()
    final["Season"] += 1
    final.columns = ["Season", "TeamID"] + ["p_" + c for c in FEATURES]
    return fw.merge(final, on=["Season", "TeamID"], how="left")


# --------------------------------------------------------------------------
# Dataset (game-level arrays)
# --------------------------------------------------------------------------
@dataclass
class Dataset:
    games: pd.DataFrame
    fw: pd.DataFrame
    decay: float
    gm: pd.DataFrame = field(init=False)

    def __post_init__(self):
        c_cols = ["c_" + c for c in FEATURES]
        p_cols = ["p_" + c for c in FEATURES]
        f = self.fw[~self.fw["Final"]][["Season", "Week", "TeamID", "Neff"] + c_cols + p_cols]
        gm = self.games.merge(
            f.rename(columns={"TeamID": "HomeID"}), on=["Season", "Week", "HomeID"], how="left"
        ).merge(
            f.rename(columns={"TeamID": "AwayID"}), on=["Season", "Week", "AwayID"], how="left",
            suffixes=("_h", "_a"))
        self.H = gm[[c + "_h" for c in c_cols]].to_numpy(float)
        self.A = gm[[c + "_a" for c in c_cols]].to_numpy(float)
        self.HP = gm[[c + "_h" for c in p_cols]].to_numpy(float)
        self.AP = gm[[c + "_a" for c in p_cols]].to_numpy(float)
        self.HN = gm["Neff_h"].fillna(0).to_numpy(float)
        self.AN = gm["Neff_a"].fillna(0).to_numpy(float)
        gm = gm[self.games.columns].copy()
        gm["HomeField"] = (~gm["NeutralSite"].astype(bool)).astype(float)
        gm["Margin"] = gm["HomeScore"].astype(float) - gm["AwayScore"].astype(float)
        self.gm = gm.reset_index(drop=True)
        self.early = (self.gm["Week"] <= PRIOR_WEEKS).to_numpy()

    def scale(self, season: int) -> np.ndarray:
        """Typical spread of each centered feature (teams with >= 4 games), from
        seasons before `season` (or that season itself if none exist)."""
        rows = self.fw[(self.fw["Season"] < season) & (self.fw["N"] >= 4)]
        if rows.empty:
            rows = self.fw[(self.fw["Season"] == season) & (self.fw["N"] >= 4)]
        return rows[["c_" + c for c in FEATURES]].std().to_numpy()

    def design(self, k: float, p: float, scale: np.ndarray) -> np.ndarray:
        """[home_field, z_home - z_away] for every game."""
        zh = blend(self.H, self.HP, self.HN, k, p * self.early, scale)
        za = blend(self.A, self.AP, self.AN, k, p * self.early, scale)
        return np.column_stack([self.gm["HomeField"].to_numpy(), zh - za])


def blend(cur, prior, n, k, p, scale):
    """(n * current + p * prior) / (n + p + k), with p = 0 where there is no prior."""
    cur = np.nan_to_num(cur / scale)
    has_prior = np.isfinite(prior).all(axis=1)
    pw = np.where(has_prior, p, 0.0)
    prior = np.nan_to_num(prior / scale)
    denom = n + pw + k
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (n[:, None] * cur + pw[:, None] * prior) / denom[:, None]
    return np.nan_to_num(z)


@lru_cache(maxsize=8)
def _dataset_cached(decay: float, first: int, last: int | None) -> Dataset:
    raw = _raw_cached(first, last)
    return Dataset(raw.games, weekly_features(raw, decay), decay)


@lru_cache(maxsize=2)
def _raw_cached(first: int, last: int | None) -> Raw:
    return load_raw(first, last)


def build_dataset(decay: float | None = None, first: int = FIRST_SEASON,
                  last: int | None = None) -> Dataset:
    decay = load_params()["decay"] if decay is None else decay
    return _dataset_cached(float(decay), first, last)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class Fit:
    weights: pd.Series   # per-feature points per SD (>= 0)
    hfa: float
    sigma: float
    scale: np.ndarray
    k: float
    p: float

    @property
    def coef(self) -> np.ndarray:
        return np.r_[self.hfa, self.weights.to_numpy()]

    def win_prob(self, margin: np.ndarray) -> np.ndarray:
        return norm.cdf(margin / self.sigma)


def _fit_rows(X, y, alpha, use, scale, k, p) -> Fit:
    model = Ridge(alpha=alpha, positive=True, fit_intercept=False).fit(X[:, use], y)
    coef = np.zeros(X.shape[1])
    coef[use] = model.coef_
    resid = y - X @ coef
    return Fit(pd.Series(coef[1:], index=FEATURES), coef[0], float(resid.std()), scale, k, p)


def _use_mask(features) -> np.ndarray:
    return np.r_[True, [f in (features or FEATURES) for f in FEATURES]]


def train_index(ds: Dataset, season: int, week: int) -> np.ndarray:
    gm = ds.gm
    before = (gm["Season"] < season) | ((gm["Season"] == season) & (gm["Week"] < week))
    return gm.index[before & gm["Completed"].astype(bool)].to_numpy()


def fit_as_of(ds: Dataset, season: int, week: int, params: dict | None = None) -> Fit:
    p = params or load_params()
    scale = ds.scale(season)
    X = ds.design(p["k"], p["p"], scale)
    tr = train_index(ds, season, week)
    if len(tr) < 100:
        raise ValueError(f"Only {len(tr)} games are available before {season} week {week}; "
                         "the model needs at least 100 earlier games to train.")
    return _fit_rows(X[tr], ds.gm.loc[tr, "Margin"].to_numpy(), p["alpha"], _use_mask(None),
                     scale, p["k"], p["p"])


# --------------------------------------------------------------------------
# Walk-forward evaluation
# --------------------------------------------------------------------------
def walk_forward(ds: Dataset, seasons, k: float, p: float, alpha: float,
                 features=None, extra_models: bool = False, label: str = "Model") -> pd.DataFrame:
    """Predict every completed game in `seasons`, refitting before each week."""
    gm = ds.gm
    y_all = gm["Margin"].to_numpy()
    use = _use_mask(features)
    hand = np.array([HAND_WEIGHTS.get(f, 0.0) for f in FEATURES])
    hand /= hand.sum()
    rows = []
    for season in seasons:
        scale = ds.scale(season)
        X = ds.design(k, p, scale)
        for week in sorted(gm.loc[gm["Season"] == season, "Week"].unique()):
            test = gm.index[(gm["Season"] == season) & (gm["Week"] == week)
                            & gm["Completed"].astype(bool)].to_numpy()
            if not len(test):
                continue
            tr = train_index(ds, season, week)
            fits = {label: _fit_rows(X[tr], y_all[tr], alpha, use, scale, k, p)}
            if extra_models:
                Z = np.column_stack([X[:, 0], X[:, 1:] @ hand])
                lr = LinearRegression(fit_intercept=False).fit(Z[tr], y_all[tr])
                fits["Hand weights"] = Fit(pd.Series(hand * lr.coef_[1], index=FEATURES), lr.coef_[0],
                                           float((y_all[tr] - lr.predict(Z[tr])).std()), scale, k, p)
                fits["Points only"] = _fit_rows(X[tr], y_all[tr], alpha,
                                                _use_mask(["OffPPG", "DefPPG"]), scale, k, p)
                fits["Home field only"] = Fit(pd.Series(0.0, index=FEATURES), y_all[tr].mean(),
                                              float(y_all[tr].std()), scale, k, p)
            for name, f in fits.items():
                pred = X[test] @ f.coef
                rows.append(pd.DataFrame({
                    "Model": name, "Season": season, "Week": week,
                    "GameID": gm.loc[test, "GameID"].values,
                    "HomeTeam": gm.loc[test, "HomeTeam"].values,
                    "AwayTeam": gm.loc[test, "AwayTeam"].values,
                    "PredMargin": pred, "HomeWinProb": f.win_prob(pred),
                    "Margin": y_all[test],
                    "HomeSpread": gm.loc[test, "HomeSpread"].values,
                }))
    return pd.concat(rows, ignore_index=True)


def score(pred: pd.DataFrame) -> dict:
    m, p, y = pred["Margin"], pred["PredMargin"], pred["HomeWinProb"].clip(1e-6, 1 - 1e-6)
    outcome = np.where(m > 0, 1.0, np.where(m < 0, 0.0, 0.5))
    decided = m != 0
    return {
        "Games": len(pred),
        "Accuracy": float(((p > 0) == (m > 0))[decided].mean()),
        "MAE": float((p - m).abs().mean()),
        "Brier": float(((y - outcome) ** 2).mean()),
        "LogLoss": float(-(outcome * np.log(y) + (1 - outcome) * np.log(1 - y))[decided].mean()),
    }


def scorecard(pred: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: score(g) for name, g in pred.groupby("Model", sort=False)}).T


BREAKEVEN_ATS = 110 / 210  # win rate needed to profit at standard -110 odds (52.4%)


def ats_result(pred_margin, actual_margin, home_spread) -> pd.Series:
    """Model's pick against the betting line: 'W', 'L', 'Push' or NA (no line / not played).

    The model takes the home side when it predicts the home team beats the line
    (pred_margin > -home_spread), otherwise the away side.
    """
    pred_margin, actual, spread = (pd.Series(np.asarray(x, float))
                                   for x in (pred_margin, actual_margin, home_spread))
    line = -spread                      # expected home margin implied by the line
    cover = actual - line               # > 0: home covered
    out = pd.Series(pd.NA, index=pred_margin.index, dtype="object")
    ok = line.notna() & actual.notna() & (pred_margin != line)
    out[ok & (cover == 0)] = "Push"
    decided = ok & (cover != 0)
    out[decided] = np.where((cover > 0) == (pred_margin > line), "W", "L")[decided]
    return out


def spread_scorecard(pred: pd.DataFrame, model: str = "Model",
                     edges=(0, 1, 2, 3, 5, 7)) -> pd.DataFrame:
    """Against-the-spread record on games with a line, by size of disagreement with Vegas."""
    m = pred[(pred["Model"] == model) & pred["HomeSpread"].notna()].copy()
    if m.empty:
        return pd.DataFrame()
    m["ATS"] = ats_result(m["PredMargin"], m["Margin"], m["HomeSpread"]).values
    m["Edge"] = (m["PredMargin"] + m["HomeSpread"]).abs()
    rows = []
    for e in edges:
        g = m[(m["Edge"] >= e) & m["ATS"].isin(["W", "L"])]
        w = int((g["ATS"] == "W").sum())
        rows.append({"Min edge (pts)": e, "Games": len(g), "W": w, "L": len(g) - w,
                     "ATS %": w / len(g) if len(g) else float("nan")})
    return pd.DataFrame(rows)


PROB_BINS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0001)
PHASES = ([0, 1, 4, 9, 18], ["Wk 1", "Wk 2-4", "Wk 5-9", "Wk 10-18"])


def _favorites(pred: pd.DataFrame) -> pd.DataFrame:
    """Each played game from the favorite's side: its win prob and whether it won (tie = 0.5)."""
    g = pred[pred["Margin"].notna()].copy()
    home_fav = g["HomeWinProb"] >= 0.5
    g["FavProb"] = np.where(home_fav, g["HomeWinProb"], 1 - g["HomeWinProb"])
    fav_margin = np.where(home_fav, g["Margin"], -g["Margin"])
    g["FavWon"] = np.where(fav_margin > 0, 1.0, np.where(fav_margin < 0, 0.0, 0.5))
    return g


def calibration_table(pred: pd.DataFrame, model: str = "Model", bins=PROB_BINS) -> pd.DataFrame:
    """Predicted vs actual win rate of the favorite, bucketed by predicted probability."""
    g = _favorites(pred[pred["Model"] == model])
    labels = [f"{lo:.0%}–{min(hi, 1):.0%}" for lo, hi in zip(bins[:-1], bins[1:])]
    g["Bucket"] = pd.cut(g["FavProb"], list(bins), labels=labels, right=False)
    rows = []
    for b, x in g.groupby("Bucket", observed=True):
        n, pr, act = len(x), x["FavProb"].mean(), x["FavWon"].mean()
        rows.append({"Win prob": b, "Games": n, "Predicted": pr, "Actual": act,
                     "Gap": act - pr, "±95%": 1.96 * np.sqrt(pr * (1 - pr) / n)})
    return pd.DataFrame(rows)


def implied_sigma(pred: pd.DataFrame) -> pd.Series:
    """The σ each prediction used, recovered from margin and probability."""
    p = pred["HomeWinProb"].clip(1e-9, 1 - 1e-9)
    s = pred["PredMargin"] / norm.ppf(p)
    return s.where(pred["PredMargin"].abs() > 0.5)


def phase_calibration(pred: pd.DataFrame, model: str = "Model") -> pd.DataFrame:
    """By part of the season: is the assumed σ right, and are the probabilities well scaled?"""
    g = _favorites(pred[pred["Model"] == model])
    phase = pd.cut(g["Week"], PHASES[0], labels=PHASES[1])
    rows = []
    for ph, x in g.groupby(phase, observed=True):
        sc = score(x)
        rows.append({"Weeks": ph, "Games": len(x), "Accuracy": sc["Accuracy"], "Brier": sc["Brier"],
                     "Predicted": x["FavProb"].mean(), "Actual": x["FavWon"].mean(),
                     "σ assumed": implied_sigma(x).median(),
                     "σ actual": (x["Margin"] - x["PredMargin"]).std()})
    return pd.DataFrame(rows)


GRID = {
    "decay": [1.0, 0.985, 0.97, 0.95, 0.9],
    "k": [0, 2, 4, 8],
    "p": [0, 4, 8, 12, 16],
    "alpha": [100, 300, 1000, 3000],
}


def tune(seasons=VALIDATION_SEASONS, verbose: bool = True) -> dict:
    """Grid-search decay, shrinkage k, prior weight p and ridge alpha (by MAE)."""
    results = []
    for decay in GRID["decay"]:
        ds = build_dataset(decay)
        for k, p, alpha in product(GRID["k"], GRID["p"], GRID["alpha"]):
            s = score(walk_forward(ds, seasons, k, p, alpha))
            results.append({"decay": decay, "k": k, "p": p, "alpha": alpha, **s})
        if verbose:
            best_d = min((r for r in results if r["decay"] == decay), key=lambda r: r["MAE"])
            print(f"decay={decay}: best MAE {best_d['MAE']:.3f} (k={best_d['k']}, p={best_d['p']}, "
                  f"alpha={best_d['alpha']}, acc={best_d['Accuracy']:.3f})")
    best = min(results, key=lambda r: r["MAE"])
    params = {key: float(best[key]) for key in GRID}
    params["validation_seasons"] = list(seasons)
    params["validation_score"] = {m: best[m] for m in ("MAE", "Accuracy", "Brier", "LogLoss")}
    DATA_DIR.mkdir(exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    pd.DataFrame(results).to_csv(DATA_DIR / "tuning_results.csv", index=False)
    return params


def load_params() -> dict:
    if PARAMS_PATH.exists():
        params = json.loads(PARAMS_PATH.read_text())
        if all(key in params for key in DEFAULT_PARAMS):
            return params
    return DEFAULT_PARAMS


def backtest(ds: Dataset | None = None, seasons=TEST_SEASONS) -> pd.DataFrame:
    """Walk-forward test of the tuned model, benchmarks, and the previous model version."""
    prm = load_params()
    ds = ds if ds is not None and ds.decay == prm["decay"] else build_dataset(prm["decay"])
    pred = walk_forward(ds, seasons, prm["k"], prm["p"], prm["alpha"], extra_models=True)
    prev = PREVIOUS_PARAMS
    pred_prev = walk_forward(build_dataset(prev["decay"]), seasons, prev["k"], prev["p"],
                             prev["alpha"], features=BASE_FEATURES, label="Previous model")
    pred = pd.concat([pred, pred_prev], ignore_index=True)
    pred.to_csv(DATA_DIR / "backtest_predictions.csv", index=False)
    return pred


def ablation(seasons=TEST_SEASONS) -> pd.DataFrame:
    """Change one thing at a time, from the previous model to the tuned model."""
    prm = load_params()
    k, alpha, p = prm["k"], prm["alpha"], prm["p"]
    # Recency is tested at its best validation setting below 1.0 (if tuning picked 1.0)
    decay = prm["decay"]
    tuning_path = DATA_DIR / "tuning_results.csv"
    if decay == 1.0 and tuning_path.exists():
        tr = pd.read_csv(tuning_path)
        decay = float(tr[tr["decay"] < 1].sort_values("MAE").iloc[0]["decay"])
    prev = PREVIOUS_PARAMS
    steps = [
        ("Previous model", 1.0, prev["k"], 0.0, prev["alpha"], BASE_FEATURES),
        ("Previous + re-tuned k/alpha", 1.0, k, 0.0, alpha, BASE_FEATURES),
        ("+ opponent-adjusted", 1.0, k, 0.0, alpha, FEATURES),
        (f"+ recency (decay {decay})", decay, k, 0.0, alpha, FEATURES),
        ("+ prior season wk 1-4, no recency", 1.0, k, p, alpha, FEATURES),
        (f"+ prior season wk 1-4, decay {decay}", decay, k, p, alpha, FEATURES),
    ]
    preds = [walk_forward(build_dataset(d), seasons, kk, pp, a, features=feats, label=name)
             for name, d, kk, pp, a, feats in steps]
    return pd.concat(preds, ignore_index=True)


VEGAS_SIGMA_DEFAULT = 13.0  # used only when no earlier season has lines


def vegas_sigma(gm: pd.DataFrame | None, season: int) -> float:
    """σ of the betting line's misses over completed games in seasons before `season`."""
    if gm is None:
        return VEGAS_SIGMA_DEFAULT
    g = gm[(gm["Season"] < season) & gm["HomeSpread"].notna() & gm["Completed"].astype(bool)]
    if len(g) < 100:
        return VEGAS_SIGMA_DEFAULT
    return float((g["HomeScore"] - g["AwayScore"] + g["HomeSpread"]).std())


def vegas_predictions(pred: pd.DataFrame, gm: pd.DataFrame | None = None) -> pd.DataFrame:
    """The betting line as a forecast, on the model's games that have one.

    Margin = -spread; win prob = Phi(margin / σ), where σ is measured on earlier seasons'
    lines only, so the line gets no hindsight either.
    """
    model = pred[(pred["Model"] == "Model") & pred["HomeSpread"].notna()]
    vegas = model.assign(Model="Vegas spread", PredMargin=-model["HomeSpread"])
    sig = {s: vegas_sigma(gm, s) for s in vegas["Season"].unique()}
    vegas["HomeWinProb"] = norm.cdf(vegas["PredMargin"] / vegas["Season"].map(sig))
    return vegas


def vegas_comparison(pred: pd.DataFrame, gm: pd.DataFrame | None = None) -> pd.DataFrame:
    """Model vs betting spread on the games where ESPN has a spread."""
    model = pred[(pred["Model"] == "Model") & pred["HomeSpread"].notna()]
    if model.empty:
        return pd.DataFrame()
    return scorecard(pd.concat([model, vegas_predictions(pred, gm)]))


# --------------------------------------------------------------------------
# Rankings + forecasts
# --------------------------------------------------------------------------
def power_rankings(ds: Dataset, season: int, week: int, f: Fit | None = None) -> pd.DataFrame:
    """Ratings entering `week` (points vs an average team on a neutral field)."""
    f = f or fit_as_of(ds, season, week)
    fw = ds.fw[(ds.fw["Season"] == season) & (ds.fw["Week"] == week)].copy()
    p = f.p if week <= PRIOR_WEEKS else 0.0
    z = blend(fw[["c_" + c for c in FEATURES]].to_numpy(float),
              fw[["p_" + c for c in FEATURES]].to_numpy(float),
              fw["Neff"].to_numpy(float), f.k, np.full(len(fw), p), f.scale)
    fw["Rating"] = z @ f.weights.to_numpy()
    lo, hi = fw["Rating"].min(), fw["Rating"].max()
    fw["Score"] = 10 * (fw["Rating"] - lo) / (hi - lo) if hi > lo else 5.0
    fw = fw.sort_values("Rating", ascending=False).reset_index(drop=True)
    fw.index = fw.index + 1
    fw.index.name = "Rank"
    return fw[["Team", "TeamID", "N", "Rating", "Score"] + FEATURES]


def predict_week(ds: Dataset, season: int, week: int, f: Fit | None = None) -> pd.DataFrame:
    f = f or fit_as_of(ds, season, week)
    gm = ds.gm
    idx = gm.index[(gm["Season"] == season) & (gm["Week"] == week)].to_numpy()
    pred = ds.design(f.k, f.p, f.scale)[idx] @ f.coef
    out = gm.loc[idx, ["Season", "Week", "Date", "GameID", "HomeTeam", "AwayTeam", "HomeScore",
                       "AwayScore", "Completed", "NeutralSite", "HomeSpread"]].copy()
    out["PredMargin"] = pred
    out["HomeWinProb"] = f.win_prob(pred)
    return out.reset_index(drop=True)


def next_week(ds: Dataset) -> tuple[int, int]:
    """(season, week) of the earliest unplayed regular-season game, else last week."""
    gm = ds.gm
    season = int(gm["Season"].max())
    pending = gm[(gm["Season"] == season) & ~gm["Completed"].astype(bool)]
    week = int(pending["Week"].min()) if len(pending) else int(gm.loc[gm["Season"] == season, "Week"].max())
    return season, week


def _phase_table(pred: pd.DataFrame) -> pd.DataFrame:
    phase = pd.cut(pred["Week"], [0, 1, 4, 9, 18], labels=["Wk 1", "Wk 2-4", "Wk 5-9", "Wk 10-18"])
    return pd.DataFrame({
        (m, ph): {"Acc": score(g)["Accuracy"], "MAE": score(g)["MAE"]}
        for (m, ph), g in pred.groupby(["Model", phase], observed=True, sort=False)
    }).T.unstack().round(3)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predict"
    pd.set_option("display.width", 220)

    if cmd == "tune":
        print("Best:", tune())
    elif cmd == "backtest":
        preds = backtest()
        print(f"\nWalk-forward test {TEST_SEASONS[0]}-{TEST_SEASONS[-1]}, params {load_params()}")
        print(scorecard(preds).round(3).to_string())
        vc = vegas_comparison(preds, build_dataset().gm)
        if not vc.empty:
            print("\nOn games with an ESPN betting spread:")
            print(vc.round(3).to_string())
            print(f"\nAgainst the spread (break-even {BREAKEVEN_ATS:.1%}), by model-vs-line gap:")
            print(spread_scorecard(preds).round(3).to_string(index=False))
        print("\nBy season phase:")
        print(_phase_table(preds[preds["Model"].isin(["Model", "Previous model"])]).to_string())
    elif cmd == "ablation":
        preds = ablation()
        print(scorecard(preds).round(3).to_string())
        print()
        print(_phase_table(preds).to_string())
    else:
        dataset = build_dataset()
        s, w = next_week(dataset)
        f = fit_as_of(dataset, s, w)
        print(f"Learned weights (points per SD), HFA={f.hfa:.2f}, sigma={f.sigma:.1f}:")
        print(f.weights.round(2).to_string())
        print(f"\nPower rankings entering {s} week {w}:")
        print(power_rankings(dataset, s, w, f)[["Team", "N", "Rating", "Score", "AdjMargin"]]
              .round(2).to_string())
        print(f"\nForecasts for {s} week {w}:")
        print(predict_week(dataset, s, w, f)[["AwayTeam", "HomeTeam", "PredMargin", "HomeWinProb"]]
              .round(2).to_string(index=False))
