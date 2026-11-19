"""NFL team ranking dashboard.

Run with:  streamlit run dashboard.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

import model as ml
from history import fetch_lines
from games import REGULAR_SEASON, fetch_games, strength_of_schedule, team_game_log
from scrape import fetch_team_stats

SCHEDULE_COLS = ["SOS", "SOV", "OppMargin", "RemainingSOS"]

DATA_PATH = Path(__file__).parent / "raw_data_csv.csv"

# Defaults taken from process.py
DEFAULT_WEIGHTS = {
    "WinPct": 0.8,
    "OffTotalYPG": 0.5,
    "OffPPG": 0.7,
    "DefTotalYPG": 0.5,
    "DefPPG": 0.7,
    "TurnoverDiff": 0.3,
    "SOS": 0.5,  # strength of schedule (added when game results are loaded)
}
SCORE = "Score"  # final score, rescaled to 0-10
# Stats where a smaller number is better (their sign is flipped before scaling)
LOWER_IS_BETTER = {
    # Defense: yards and points allowed
    "DefTotalYds", "DefTotalYPG", "DefPassYds", "DefPassYPG",
    "DefRushYds", "DefRushYPG", "DefPoints", "DefPPG",
    # Turnovers committed
    "GiveawayInt", "GiveawayFum", "GiveawayTotal",
    # Kickoff returns happen after the opponent scores, so more = worse defense
    "KickoffAtt",
    # Fair catches give up return opportunities
    "PuntFC",
    # Future opponents' win %: an easier remaining schedule means more expected wins
    "RemainingSOS",
}
# Every other column is higher-is-better: Games, Wins, WinPct, all Off* stats,
# KickoffYds/Avg/Long/TD, PuntAtt/Yds/Avg/Long/TD, TurnoverDiff, Takeaway*,
# SOS, SOV, OppMargin.
# Fixed colors for the two compared teams (categorical slots 1 and 2)
COLOR_A, COLOR_B = "#2a78d6", "#eb6834"


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Team"]).reset_index(drop=True)
    df["WinPct"] = 100.0 * df["Wins"].astype(float) / df["Games"].astype(float)
    return df


@st.cache_data
def load_csv() -> pd.DataFrame:
    return _finish(pd.read_csv(DATA_PATH, index_col=False, encoding="utf-8-sig"))


@st.cache_data(ttl=3600, show_spinner="Fetching stats from ESPN...")
def load_espn(season: int | None) -> pd.DataFrame:
    return _finish(fetch_team_stats(season))


@st.cache_data(ttl=3600, show_spinner="Fetching game results from ESPN...")
def load_games(season: int | None) -> pd.DataFrame:
    games = fetch_games(season)
    return games[games["SeasonType"] == REGULAR_SEASON]


def schedule_metrics(games: pd.DataFrame) -> pd.DataFrame:
    sos = strength_of_schedule(games)
    sos["SOV"] = sos["SOV"].fillna(0.0)  # winless teams have no victories
    return sos


def lower_is_better(col: str) -> bool:
    return col in LOWER_IS_BETTER


st.set_page_config(page_title="NFL Team Rankings", layout="wide")
st.title("NFL Team Rankings")

VIEWS = ["Rankings", "Compare teams", "Model forecasts"]
SEASON_OPTIONS = ["Current"] + [str(y) for y in range(2025, 2001, -1)]
st.session_state.setdefault("view", VIEWS[0])
st.session_state.setdefault("last_view", VIEWS[0])


def forecast_click() -> None:
    """Forecast-table cell clicked: the chart column opens the margin chart, anything else
    loads that game's two teams + season into Compare teams."""
    cells = st.session_state[st.session_state["ml_table_key"]].selection.cells
    # New key next run so the table comes back unselected and the same cell can be clicked again.
    st.session_state["ml_table_n"] = st.session_state.get("ml_table_n", 0) + 1
    if not cells:
        return
    row, col = cells[0]
    if col == "Chart":
        st.session_state["plot_game"] = st.session_state["ml_games"][row]
        return
    away, home, season_, week_ = st.session_state["ml_matchups"][row]
    st.session_state["cmp_pending"] = (away, home, season_, week_)
    st.session_state["source"] = "Live from ESPN"
    st.session_state["season_choice"] = str(season_) if str(season_) in SEASON_OPTIONS else "Current"
    st.session_state["view"] = st.session_state["last_view"] = "Compare teams"


def keep_view() -> None:
    """Segmented controls can be un-selected; treat that as staying put."""
    if st.session_state["view"] is None:
        st.session_state["view"] = st.session_state["last_view"]
    st.session_state["last_view"] = st.session_state["view"]


def back_to_forecasts() -> None:
    st.session_state["view"] = st.session_state["last_view"] = "Model forecasts"

# ---------------- Sidebar: data source ----------------
games = None
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Live from ESPN", "Local CSV"], horizontal=True, key="source")
    if source == "Live from ESPN":
        season_choice = st.selectbox(
            "Season",
            SEASON_OPTIONS,
            key="season_choice",
            help="Regular-season stats. A year means that season (e.g. 2024 = 2024-25).",
        )
        season = None if season_choice == "Current" else int(season_choice)
        try:
            df = load_espn(season)
        except Exception as exc:  # network error or ESPN layout change
            st.error(f"ESPN fetch failed ({exc}); using local CSV.")
            df = load_csv()
        else:
            col_a, col_b = st.columns(2)
            if col_a.button("Refresh"):
                load_espn.clear()
                load_games.clear()
                st.rerun()
            if col_b.button("Save to CSV"):
                df.drop(columns="WinPct").to_csv(DATA_PATH, index=False)
                load_csv.clear()
                st.success(f"Saved to {DATA_PATH.name}")
    else:
        season = None  # local CSV is assumed to be the current season
        df = load_csv()

    if st.checkbox(
        "Load game results",
        value=True,
        help="Adds schedule-strength columns (SOS = opponents' combined win %, "
        "SOV = beaten opponents' win %, OppMargin = opponents' avg point margin, "
        "RemainingSOS = future opponents' win %) and game logs for team comparison.",
    ):
        try:
            games = load_games(season)
            sched = schedule_metrics(games)
            df = df.merge(
                sched[["Team"] + SCHEDULE_COLS].dropna(axis=1, how="all"),
                on="Team", how="left",
            )
        except Exception as exc:
            games = None
            st.error(f"Game-results fetch failed ({exc}).")

numeric_cols = [c for c in df.select_dtypes("number").columns]

# ---------------- Sidebar: column selection + weights ----------------
with st.sidebar:
    st.divider()
    st.header("Scoring inputs")
    selected = st.multiselect(
        "Columns used in the score",
        options=numeric_cols,
        default=[c for c in DEFAULT_WEIGHTS if c in numeric_cols],
    )

    weights = {}
    if selected:
        st.subheader("Weights")
        st.caption("↓ = lower is better (sign flipped before scoring)")
        for col in selected:
            weights[col] = st.slider(
                col + (" ↓" if lower_is_better(col) else ""),
                0.0, 1.0, DEFAULT_WEIGHTS.get(col, 0.5), 0.05, key=f"w_{col}",
            )

    if st.button("Reset to defaults"):
        for k in list(st.session_state):
            if k.startswith("w_"):
                del st.session_state[k]
        st.rerun()

# ---------------- Scoring ----------------
if not selected:
    st.info("Choose at least one column in the sidebar to score teams.")
    st.stop()

total_weight = sum(weights.values())
if total_weight == 0:
    st.warning("All weights are 0. Set at least one weight above 0.")
    st.stop()

data = df[["Team"] + selected].dropna().reset_index(drop=True)
signed = data[selected].copy()
for col in selected:
    if lower_is_better(col):
        signed[col] = -signed[col]

# z-scores, with "lower is better" columns flipped so higher always = better
scaled = pd.DataFrame(StandardScaler().fit_transform(signed), columns=selected)
w = pd.Series(weights)
weighted = (scaled[selected] * w).sum(axis=1) / total_weight

# Rescale the weighted z-score to 0-10: worst team = 0, best team = 10
spread = weighted.max() - weighted.min()
scaled[SCORE] = 10 * (weighted - weighted.min()) / spread if spread else 5.0
scaled.insert(0, "Team", data["Team"])

ranked = scaled.sort_values(SCORE, ascending=False).reset_index(drop=True)
ranked.index = ranked.index + 1
ranked.index.name = "Rank"
ranked = ranked[["Team", SCORE] + selected]

raw = data.set_index("Team").loc[ranked["Team"], selected].reset_index()
raw.index = ranked.index
raw.insert(1, SCORE, ranked[SCORE].round(2))

view = st.segmented_control("View", VIEWS, key="view", on_change=keep_view,
                            label_visibility="collapsed") or st.session_state["last_view"]

# =====================================================================
# Rankings tab
# =====================================================================
if view == "Rankings":
    c1, c2, c3 = st.columns(3)
    c1.metric("Top team", ranked.iloc[0]["Team"])
    c2.metric("Columns scored", len(selected))
    c3.metric("Teams ranked", len(ranked))
    if not any(c in selected for c in SCHEDULE_COLS):
        st.caption("Schedule strength isn't in the score. Add SOS/SOV/OppMargin in the sidebar.")

    chart_df = ranked.sort_values(SCORE)
    fig = px.bar(
        chart_df,
        x="Team",
        y=SCORE,
        color=SCORE,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=5,
        range_y=[0, 10],
        labels={SCORE: "Score (0-10)"},
    )
    fig.update_layout(xaxis_tickangle=-90, height=500, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    sub_raw, sub_scaled = st.tabs(["Raw values", "Standardized (z) inputs"])
    with sub_raw:
        st.dataframe(raw, width="stretch")
    with sub_scaled:
        st.dataframe(ranked.round(3), width="stretch")

    st.download_button(
        "Download rankings (CSV)",
        raw.to_csv().encode(),
        file_name="nfl_rankings.csv",
        mime="text/csv",
    )


# =====================================================================
# Compare tab
# =====================================================================
# =====================================================================
# Model tab (rendered before the compare tab, whose same-team check stops the script)
# =====================================================================
@st.cache_resource(ttl=3600, show_spinner="Loading game history (first run downloads ~15 seasons, a few minutes)...")
def load_model_dataset():
    # model.py memoizes datasets in-process; clear so an hourly reload picks up new games
    ml._raw_cached.cache_clear()
    ml._dataset_cached.cache_clear()
    return ml.build_dataset()


def nickname(team: str) -> str:
    """'San Francisco 49ers' -> '49ers' (keeps tables narrow)."""
    return "Washington" if team.endswith("Football Team") else team.split()[-1]


@st.cache_data(ttl=1800, show_spinner="Fetching betting lines...")
def live_lines(game_ids: tuple[str, ...]) -> dict[str, float]:
    return fetch_lines(game_ids)


def line_label(spread: float, home: str, away: str) -> str:
    """Home spread -> 'Giants -2.5' (favorite and points), 'PK' or ''."""
    if pd.isna(spread):
        return ""
    if spread == 0:
        return "PK"
    return f"{nickname(home if spread < 0 else away)} -{abs(spread):g}"


def _bump_table() -> None:
    """Rebuild the forecast table after the chart closes (a mouse-closed dialog otherwise
    leaves the grid ignoring its next click)."""
    st.session_state["ml_table_n"] = st.session_state.get("ml_table_n", 0) + 1


@st.dialog("Win probability by margin", width="large", on_dismiss=_bump_table)
def margin_dialog(g: dict) -> None:
    """P(pick wins by more than x points) for a range of x, from the model and the Vegas line."""
    pick, opp, m, sigma = g["pick"], g["opp"], g["margin"], g["sigma"]
    st.markdown(f"**{g['matchup']}** · {g['season']} week {g['week']}")
    lo = min(-21.0, np.floor((m - 2.5 * sigma) / 7) * 7)
    hi = max(28.0, np.ceil((m + 2.5 * sigma) / 7) * 7)
    x = np.linspace(lo, hi, 401)
    model_p = norm.cdf((m - x) / sigma)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=model_p, mode="lines", name="Model", line=dict(color=COLOR_A, width=2),
        hovertemplate=f"{nickname(pick)} win by more than %{{x:.1f}}: %{{y:.1%}}<extra>Model</extra>",
    ))
    vl = g["vegas_margin"]
    if vl is not None:
        fig.add_trace(go.Scatter(
            x=x, y=norm.cdf((vl - x) / g["vegas_sigma"]), mode="lines", name="Vegas line",
            line=dict(color=COLOR_B, width=2, dash="dash"),
            hovertemplate=f"{nickname(pick)} win by more than %{{x:.1f}}: %{{y:.1%}}"
                          "<extra>Vegas line</extra>",
        ))
    # Reference markers on the model curve.
    marks = [(0.0, "Win"), (m, "Predicted margin")]
    if vl is not None:
        marks.append((vl, "Covers Vegas line"))
    fig.add_trace(go.Scatter(
        x=[v for v, _ in marks], y=[norm.cdf((m - v) / sigma) for v, _ in marks],
        mode="markers", showlegend=False, text=[t for _, t in marks],
        marker=dict(size=10, color=COLOR_A, line=dict(width=2, color="white")),
        hovertemplate="%{text}: %{y:.1%}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#9a9a9a", width=1))
    fig.add_hline(y=0.5, line=dict(color="#9a9a9a", width=1, dash="dot"))
    if g["actual"] is not None:
        fig.add_vline(x=g["actual"], line=dict(color="#555", width=1.5, dash="dot"),
                      annotation_text=f"Actual: {g['actual']:+g}", annotation_position="top")
    fig.update_layout(
        height=420, margin=dict(t=30, b=10),
        xaxis=dict(title=f"Points: {nickname(pick)} margin of victory (negative = {nickname(opp)} win)",
                   zeroline=False, dtick=7),
        yaxis=dict(title=f"Chance the {nickname(pick)} win by more than this", tickformat=".0%",
                   range=[0, 1.02]),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    rows = {"Win": 0.0, **{f"Win by {k}+": k - 0.5 for k in (3, 7, 10, 14)}}
    if vl is not None:
        rows[f"Cover ({line_label(g['spread'], g['home'], g['away'])})"] = vl
    tbl = pd.DataFrame({"Outcome": list(rows),
                        "Model": [norm.cdf((m - v) / sigma) for v in rows.values()]})
    if vl is not None:
        tbl["Vegas line"] = [norm.cdf((vl - v) / g["vegas_sigma"]) for v in rows.values()]
    pct = st.column_config.NumberColumn(format="percent")
    st.dataframe(tbl, hide_index=True, width="stretch",
                 column_config={"Model": pct, "Vegas line": pct})
    st.caption(
        f"Model: {nickname(pick)} by {m:.1f} with σ = {sigma:.1f} pts (its typical miss), so "
        "the outcome is treated as a bell curve around the predicted margin. "
        + (f"Vegas: line as the expected margin ({vl:+g}) with σ = {g['vegas_sigma']:.1f} pts "
           "from earlier seasons' lines. " if vl is not None else "No Vegas line for this game. ")
        + "Real NFL margins cluster on 3 and 7, which a smooth curve doesn't capture, so "
        "probabilities right around those numbers are approximate."
    )


def calibration_section(bt: pd.DataFrame, cur: pd.DataFrame, cur_season, gm: pd.DataFrame) -> None:
    """Do games given X% actually go the favorite's way X% of the time?"""
    st.markdown("##### Win probability calibration")
    c1, c2 = st.columns([3, 2])
    sample = c1.radio(
        "Games", [f"Backtest {bt['Season'].min()}–{bt['Season'].max()}", f"{cur_season} so far"],
        horizontal=True, key="calib_sample", label_visibility="collapsed",
    )
    show_vegas = c2.toggle("Compare with Vegas line", value=True, key="calib_vegas")
    pred = bt if sample.startswith("Backtest") else cur.assign(Model="Model")
    pred = pred[pred["Model"] == "Model"]
    vegas = ml.vegas_predictions(pred, gm) if show_vegas else pd.DataFrame()
    if show_vegas and vegas["Margin"].notna().sum() == 0:
        st.caption("No completed games with a Vegas line in this sample; showing the model alone.")
        show_vegas = False
    if show_vegas:
        pred = pred[pred["HomeSpread"].notna()]
    cal = ml.calibration_table(pred)
    if cal.empty:
        st.info("No completed games to evaluate yet.")
        return
    sources = {"Model": cal}
    if show_vegas:
        sources["Vegas spread"] = ml.calibration_table(vegas, "Vegas spread")

    lo, hi = cal["Predicted"] - cal["±95%"], cal["Predicted"] + cal["±95%"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0.5, 1], y=[0.5, 1], mode="lines", line=dict(color="#9a9a9a", width=1, dash="dot"),
        name="Perfect calibration", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([cal["Predicted"], cal["Predicted"][::-1]]), y=pd.concat([hi, lo[::-1]]),
        mode="lines", fill="toself", fillcolor="rgba(154,154,154,0.18)", line=dict(width=0),
        name="Expected range for model (95%)", hoverinfo="skip",
    ))
    for name, color in [("Model", COLOR_A), ("Vegas spread", COLOR_B)]:
        if name not in sources:
            continue
        t = sources[name]
        fig.add_trace(go.Scatter(
            x=t["Predicted"], y=t["Actual"], mode="lines+markers",
            name="Vegas line" if name == "Vegas spread" else "Model",
            line=dict(color=color, width=2), marker=dict(size=9, line=dict(width=2, color="white")),
            error_y=dict(array=t["±95%"], thickness=1.5, width=0, color=color)
            if name == "Vegas spread" else None,
            customdata=t[["Win prob", "Games"]],
            hovertemplate=f"<b>{'Vegas line' if name == 'Vegas spread' else 'Model'}</b><br>"
                          "%{customdata[0]} bucket · %{customdata[1]} games<br>"
                          "Predicted %{x:.1%}<br>Actual %{y:.1%}<extra></extra>",
        ))
    fig.update_layout(
        title="Favorite's predicted vs actual win rate", height=400,
        xaxis=dict(title="Predicted win probability", tickformat=".0%", range=[0.5, 0.95]),
        yaxis=dict(title="Actual win rate", tickformat=".0%", range=[0.3, 1.05]),
        legend=dict(orientation="h", y=-0.22), margin=dict(t=50),
    )
    st.plotly_chart(fig, width="stretch")

    pct = st.column_config.NumberColumn(format="percent")
    table = pd.concat([t.assign(Source="Vegas line" if n == "Vegas spread" else n)
                       for n, t in sources.items()], ignore_index=True)
    table = table[["Source"] + [c for c in table.columns if c != "Source"]]
    st.dataframe(
        table.assign(Gap=table["Gap"] * 100, **{"±95%": table["±95%"] * 100}), hide_index=True,
        width="stretch",
        column_config={"Predicted": pct, "Actual": pct,
                       "Gap": st.column_config.NumberColumn("Gap (pts)", format="%+.1f"),
                       "±95%": st.column_config.NumberColumn("Expected noise ±", format="%.1f")},
    )
    outside = table[table["Gap"].abs() > table["±95%"]]
    n_games = int(pred["Margin"].notna().sum())
    vs = ""
    if show_vegas:
        both = pd.concat([pred, vegas])
        b = {m: ml.score(g[g["Margin"].notna()])["Brier"] for m, g in both.groupby("Model")}
        sig = sorted({round(ml.vegas_sigma(gm, int(x)), 1) for x in vegas["Season"].unique()})
        sig_txt = f"{sig[0]:g}–{sig[-1]:g}" if len(sig) > 1 else f"{sig[0]:g}"
        vs = (f"Compared on the {n_games} games with an ESPN line. Brier score (lower is better): "
              f"model {b['Model']:.3f}, Vegas {b['Vegas spread']:.3f}. Vegas probabilities use "
              f"Φ(line / σ) with σ = {sig_txt} pts, "
              "measured on earlier seasons' lines only. ")
    st.caption(
        vs + "Each game is counted from the favorite's side. Gap = actual minus predicted win "
        "rate, in percentage points; gaps inside the expected-noise band are luck. "
        + ("Every bucket is inside its band." if outside.empty else
           "Outside the band: " + ", ".join(f"{r.Source} {r['Win prob']}" for _, r in outside.iterrows())
           + (" — with this many buckets, one or two land outside by chance."
              if len(outside) <= 2 else "."))
    )

    ph = pd.concat(
        [ml.phase_calibration(pred).assign(Source="Model")]
        + ([ml.phase_calibration(vegas, "Vegas spread").assign(Source="Vegas line")] if show_vegas else []),
        ignore_index=True,
    )
    ph = ph[["Source"] + [c for c in ph.columns if c != "Source"]]
    st.dataframe(
        ph, hide_index=True, width="stretch",
        column_config={"Accuracy": pct, "Predicted": pct, "Actual": pct,
                       "Brier": st.column_config.NumberColumn(format="%.3f"),
                       "σ assumed": st.column_config.NumberColumn(format="%.1f pts"),
                       "σ actual": st.column_config.NumberColumn(format="%.1f pts")},
    )
    st.caption(
        "By part of the season. Predicted/Actual = favorites' average win probability vs their "
        "actual win rate. σ assumed is the spread of outcomes the model used to turn margins into "
        "probabilities; σ actual is the measured standard deviation of its misses. If σ actual "
        "were well above σ assumed, the probabilities would be overconfident."
    )


@st.cache_data(show_spinner="Scoring this season against the spread...")
def season_spread_record(_ds, season: int, data_version: int) -> pd.DataFrame:
    prm = ml.load_params()
    return ml.walk_forward(_ds, [season], prm["k"], prm["p"], prm["alpha"])


@st.cache_data(show_spinner="Fitting model...")
def model_outputs(_ds, season: int, week: int, data_version: int):
    f = ml.fit_as_of(_ds, season, week)
    return f, ml.power_rankings(_ds, season, week, f), ml.predict_week(_ds, season, week, f)


if view == "Model forecasts":
    st.caption(
        "Learned from game history. Team stats use only games played earlier that season, plus "
        f"(weeks 1-{ml.PRIOR_WEEKS} only) last season's final stats; they include opponent-adjusted "
        "ratings. Each week's model is trained only on games completed before that week."
    )
    try:
        ds = load_model_dataset()
    except Exception as exc:
        st.error(f"Could not load game history ({exc}).")
        ds = None

    if ds is not None:
        cur_season, cur_week = ml.next_week(ds)
        seasons = sorted(ds.gm["Season"].unique(), reverse=True)
        pick_s, pick_w = st.columns(2)
        m_season = pick_s.selectbox("Season", seasons, index=seasons.index(cur_season), key="ml_season")
        weeks = sorted(ds.gm.loc[ds.gm["Season"] == m_season, "Week"].unique())
        default_w = cur_week if m_season == cur_season else weeks[-1]
        m_week = pick_w.selectbox("Week", weeks, index=weeks.index(default_w), key="ml_week")

        try:
            fit_, rankings, preds = model_outputs(ds, int(m_season), int(m_week), len(ds.gm))
        except ValueError as exc:
            st.info(f"{exc} Pick a later week or season.")
            ds = None

    if ds is not None:

        # ---- Forecasts ----
        st.markdown(f"#### Week {m_week} forecasts")
        preds = preds.copy()
        played = preds["Completed"].astype(bool)
        need = preds.loc[preds["HomeSpread"].isna() & ~played, "GameID"]
        if len(need):
            fetched = live_lines(tuple(need.astype(str)))
            preds.loc[need.index, "HomeSpread"] = need.astype(str).map(fetched).values
        fav_home = preds["PredMargin"] >= 0
        actual = preds["HomeScore"].astype(float) - preds["AwayScore"].astype(float)
        vegas_home = -preds["HomeSpread"]                     # line as expected home margin
        ats = ml.ats_result(preds["PredMargin"], actual.where(played), preds["HomeSpread"])
        ats_side = preds["HomeTeam"].where(preds["PredMargin"] > vegas_home, preds["AwayTeam"])
        table = pd.DataFrame({
            "Chart": "📈",
            "Matchup": preds["AwayTeam"] + preds["NeutralSite"].map({True: " vs ", False: " @ "})
                       + preds["HomeTeam"],
            "Pick": preds["HomeTeam"].where(fav_home, preds["AwayTeam"]),
            "By (pts)": preds["PredMargin"].abs().round(1),
            "Win prob": preds["HomeWinProb"].where(fav_home, 1 - preds["HomeWinProb"]),
            "Vegas line": [line_label(sp, h, a) for sp, h, a in
                           zip(preds["HomeSpread"], preds["HomeTeam"], preds["AwayTeam"])],
            "Spread pick": [nickname(t) if pd.notna(sp) and pm != -sp else ""
                            for t, sp, pm in zip(ats_side, preds["HomeSpread"], preds["PredMargin"])],
            "Result": [
                f"{a} {int(asc)}-{int(hsc)} {h}" if c else "—"
                for a, h, asc, hsc, c in zip(preds["AwayTeam"], preds["HomeTeam"],
                                            preds["AwayScore"], preds["HomeScore"], played)
            ],
            "Correct": [("✓" if (p >= 0) == (m > 0) else "✗") if c and m != 0 else ""
                        for p, m, c in zip(preds["PredMargin"], actual, played)],
            "Spread error": (preds["PredMargin"] - actual).abs().round(1).where(played),
            "ATS": ats.map({"W": "✓", "L": "✗", "Push": "Push"}).fillna("").values,
        })
        st.caption("Click 📈 for a game's win probability by margin; click anywhere else in a "
                   "row to compare the two teams.")
        matchups = [(a, h, int(m_season), int(m_week))
                    for a, h in zip(preds["AwayTeam"], preds["HomeTeam"])]
        st.session_state["ml_matchups"] = matchups
        v_sigma = ml.vegas_sigma(ds.gm, int(m_season))
        games_info = []
        for i in range(len(preds)):
            r = preds.iloc[i]
            home_pick = bool(fav_home.iloc[i])
            sgn = 1 if home_pick else -1
            sp = r["HomeSpread"]
            games_info.append({
                "matchup": table["Matchup"].iloc[i], "season": int(m_season), "week": int(m_week),
                "home": r["HomeTeam"], "away": r["AwayTeam"],
                "pick": r["HomeTeam"] if home_pick else r["AwayTeam"],
                "opp": r["AwayTeam"] if home_pick else r["HomeTeam"],
                "margin": float(abs(r["PredMargin"])), "sigma": float(fit_.sigma),
                "spread": float(sp) if pd.notna(sp) else None,
                "vegas_margin": float(-sp * sgn) if pd.notna(sp) else None,
                "vegas_sigma": v_sigma,
                "actual": float(actual.iloc[i] * sgn) if played.iloc[i] else None,
            })
        st.session_state["ml_games"] = games_info
        st.session_state["ml_table_key"] = f"ml_forecast_table_{st.session_state.get('ml_table_n', 0)}"
        st.dataframe(
            table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3,
            key=st.session_state["ml_table_key"], on_select=forecast_click,
            selection_mode="single-cell",
            column_config={
                "Chart": st.column_config.TextColumn(
                    "", width=40, help="Win probability by margin of victory"),
                "Win prob": st.column_config.ProgressColumn(
                    "Win prob", format="percent", min_value=0.5, max_value=1.0),
                "Spread error": st.column_config.NumberColumn(
                    "Spread error", format="%.1f pts",
                    help="How far the predicted margin was from the actual margin."),
                "Spread pick": st.column_config.TextColumn(
                    help="The side the model takes against the Vegas line."),
                "ATS": st.column_config.TextColumn(
                    help="Did the model's spread pick cover the Vegas line?"),
            },
        )
        if "plot_game" in st.session_state:
            margin_dialog(st.session_state.pop("plot_game"))
        if played.any():
            decided = played & (actual != 0)
            hits = ((preds["PredMargin"] >= 0) == (actual > 0))[decided]
            err = (preds["PredMargin"] - actual)[played].abs()
            line_err = (vegas_home - actual)[played & vegas_home.notna()].abs()
            w, l = int((ats == "W").sum()), int((ats == "L").sum())
            st.caption(
                f"Week {m_week}: {int(hits.sum())}/{int(decided.sum())} picks correct; "
                f"average spread error {err.mean():.1f} pts"
                + (f" (Vegas {line_err.mean():.1f})" if len(line_err) else "")
                + (f"; {w}-{l} against the spread." if w + l else ".")
            )
        if not played.all():
            st.caption("Vegas lines for unplayed games are current ESPN lines (refreshed every 30 min).")
        if m_week <= ml.PRIOR_WEEKS:
            st.caption(
                f"Weeks 1-{ml.PRIOR_WEEKS}: each team's stats are blended with last season's final "
                f"stats (worth {fit_.p:g} games). From week {ml.PRIOR_WEEKS + 1} on, current season only."
            )

        # ---- Power rankings + weights ----
        rank_col, weight_col = st.columns([3, 2])
        with rank_col:
            st.markdown(f"#### Power rankings entering week {m_week}")
            st.dataframe(
                rankings[["Team", "N", "Rating", "Score", "AdjMargin", "WinPct", "OffPPG", "DefPPG", "OffYPP"]]
                .rename(columns={"N": "GP", "Rating": "Rating (pts)", "Score": "Score (0-10)",
                                 "AdjMargin": "Adj margin"})
                .round(2),
                width="stretch", height=35 * 13,
            )
            st.caption("Rating = expected margin vs an average team on a neutral field. "
                       "Adj margin = point margin per game adjusted for opponents and home field.")
        with weight_col:
            st.markdown("#### Learned weights")
            wts = fit_.weights.sort_values()
            wts = wts[wts > 0.005]
            fig = px.bar(x=wts.values, y=wts.index, orientation="h",
                         labels={"x": "Points per standard deviation", "y": ""})
            fig.update_traces(marker_color=COLOR_A, marker_line_width=0,
                              hovertemplate="%{y}: %{x:.2f} pts per SD<extra></extra>")
            fig.update_layout(height=120 + 32 * len(wts), margin=dict(l=0, r=10, t=10, b=0))
            st.plotly_chart(fig, width="stretch")
            zero = [f for f in ml.FEATURES if fit_.weights[f] <= 0.005]
            st.caption(
                f"Home field: {fit_.hfa:+.2f} pts. Typical miss (σ): {fit_.sigma:.1f} pts. "
                + (f"Dropped (weight 0): {', '.join(zero)}." if zero else "")
            )

        # ---- Backtest ----
        st.markdown("#### Backtest")
        bt_path = ml.DATA_DIR / "backtest_predictions.csv"
        if st.button("Run walk-forward backtest" if not bt_path.exists() else "Re-run backtest"):
            with st.spinner("Backtesting (refits the model before every week)..."):
                ml.backtest(ds)
        if bt_path.exists():
            bt = pd.read_csv(bt_path)
            sc = ml.scorecard(bt)
            st.dataframe(
                sc.assign(Games=sc["Games"].astype(int)).round(3), width="stretch",
                column_config={"Accuracy": st.column_config.NumberColumn(format="percent")},
            )
            st.caption(
                f"Seasons {bt['Season'].min()}–{bt['Season'].max()}, every game predicted using only "
                "earlier data. Accuracy = winner picked correctly; MAE = average miss in points; "
                "Brier and LogLoss score the win probabilities (lower is better)."
            )
            by_season = pd.DataFrame([
                {"Season": s, "Model": m, "Accuracy": ml.score(g)["Accuracy"]}
                for (s, m), g in bt[bt["Model"].isin(["Model", "Previous model"])].groupby(["Season", "Model"])
            ])
            fig = px.line(by_season, x="Season", y="Accuracy", color="Model", markers=True,
                          color_discrete_map={"Model": COLOR_A, "Previous model": COLOR_B},
                          category_orders={"Model": ["Model", "Previous model"]},
                          title="Pick accuracy by season")
            fig.update_traces(line_width=2, marker_size=8,
                              hovertemplate="%{x}: %{y:.1%}<extra>%{fullData.name}</extra>")
            fig.update_layout(height=340, yaxis_tickformat=".0%", legend_title_text="")
            st.plotly_chart(fig, width="stretch")
            vc = ml.vegas_comparison(bt, ds.gm)
            if not vc.empty:
                st.caption(
                    f"On the {int(vc.loc['Model', 'Games'])} test games with an ESPN betting spread: "
                    f"model {vc.loc['Model', 'Accuracy']:.1%} accuracy / {vc.loc['Model', 'MAE']:.2f} MAE, "
                    f"Vegas {vc.loc['Vegas spread', 'Accuracy']:.1%} / {vc.loc['Vegas spread', 'MAE']:.2f}."
                )

            st.markdown("##### Against the spread")
            sc_ats = ml.spread_scorecard(bt)
            if not sc_ats.empty:
                st.dataframe(
                    sc_ats, hide_index=True, width="stretch",
                    column_config={"ATS %": st.column_config.NumberColumn(format="percent")},
                )
                st.caption(
                    "Each row counts games where the model's margin differed from the Vegas line by at "
                    f"least that many points. Pushes excluded. Break-even at standard -110 odds is "
                    f"{ml.BREAKEVEN_ATS:.1%}."
                )
            cur = season_spread_record(ds, int(cur_season), len(ds.gm))
            cur_l = cur[cur["HomeSpread"].notna()]
            if len(cur_l):
                res = ml.ats_result(cur_l["PredMargin"], cur_l["Margin"], cur_l["HomeSpread"])
                w, l, pu = int((res == "W").sum()), int((res == "L").sum()), int((res == "Push").sum())
                st.caption(
                    f"{cur_season} so far (walk-forward, not part of the test above): {w}-{l}"
                    + (f"-{pu}" if pu else "") + f" against the spread"
                    + (f" ({w / (w + l):.1%})" if w + l else "")
                    + f"; spread error {(cur_l['PredMargin'] - cur_l['Margin']).abs().mean():.2f} pts "
                    f"vs Vegas {(-cur_l['HomeSpread'] - cur_l['Margin']).abs().mean():.2f}."
                )

            calibration_section(bt, cur, cur_season, ds.gm)


def game_table(log: pd.DataFrame, team: str, records: pd.Series, opp_scores: pd.Series) -> pd.DataFrame:
    t = log[log["Team"] == team].sort_values("Week")
    return pd.DataFrame({
        "Wk": t["Week"].values,
        "Date": t["Date"].dt.tz_convert("US/Eastern").dt.strftime("%b %d").values,
        "Opponent": [("vs " if h else "@ ") + nickname(o) for h, o in zip(t["Home"], t["Opponent"])],
        "Result": [
            f"{r} {pf}-{pa}" if c else "—"
            for r, pf, pa, c in zip(t["Result"], t["PointsFor"], t["PointsAgainst"], t["Completed"])
        ],
        "Opp record": t["Opponent"].map(records).values,
        "Opp score": t["Opponent"].map(opp_scores).round(1).values,
    })


def match_team(name: str, teams: list[str]) -> str | None:
    """Game-feed name -> name used in the stats table (handles renames like Washington)."""
    if name in teams:
        return name
    same = [t for t in teams if nickname(t) == nickname(name)]
    if len(same) == 1:
        return same[0]
    city = [t for t in teams if t.split()[0] == name.split()[0]]
    return city[0] if len(city) == 1 else None


def record_str(results: pd.Series) -> str:
    w, l, t = (results == "W").sum(), (results == "L").sum(), (results == "T").sum()
    return f"{w}-{l}" + (f"-{t}" if t else "")


if view == "Compare teams":
    teams = ranked["Team"].tolist()
    pending = st.session_state.pop("cmp_pending", None)
    if pending:
        away, home, p_season, p_week = pending
        found = [match_team(t, teams) for t in (away, home)]
        if all(found):
            st.session_state["cmp_a"], st.session_state["cmp_b"] = found
            st.session_state["cmp_from"] = (away, home, p_season, p_week)
        else:
            st.warning(f"Couldn't find {away} / {home} in the {p_season} team stats.")
    if st.session_state.get("cmp_a") not in teams:
        st.session_state["cmp_a"] = teams[0]
    if st.session_state.get("cmp_b") not in teams:
        st.session_state["cmp_b"] = teams[min(1, len(teams) - 1)]
    origin = st.session_state.get("cmp_from")
    if origin and {match_team(origin[0], teams), match_team(origin[1], teams)} == {
            st.session_state["cmp_a"], st.session_state["cmp_b"]}:
        note, back = st.columns([4, 1])
        note.caption(f"From the forecasts: {origin[0]} @ {origin[1]}, {origin[2]} week {origin[3]}. "
                     "Stats and schedule below are for the full season in the sidebar.")
        back.button("← Back to forecasts", on_click=back_to_forecasts)
    pick_a, pick_b = st.columns(2)
    team_a = pick_a.selectbox("Team A", teams, key="cmp_a")
    team_b = pick_b.selectbox("Team B", teams, key="cmp_b")
    if team_a == team_b:
        st.warning("Pick two different teams.")
        st.stop()
    color_map = {team_a: COLOR_A, team_b: COLOR_B}

    log = team_game_log(games) if games is not None and not games.empty else None
    records = scores_by_team = None
    if log is not None:
        played = log[log["Completed"]]
        records = played.groupby("Team")["Result"].apply(record_str)
    scores_by_team = ranked.set_index("Team")[SCORE]
    rank_by_team = pd.Series(ranked.index, index=ranked["Team"])

    # ---- Header cards ----
    head_a, head_b = st.columns(2)
    for col, team in ((head_a, team_a), (head_b, team_b)):
        with col:
            st.subheader(team)
            m1, m2, m3 = st.columns(3)
            m1.metric("Rank", f"#{rank_by_team[team]}")
            m2.metric("Score", f"{scores_by_team[team]:.1f}")
            rec = records.get(team) if records is not None else (
                f"{int(df.set_index('Team').at[team, 'Wins'])} W")
            m3.metric("Record", rec)

    # ---- Stat comparison table ----
    st.markdown("#### Statistics")
    show_all = st.toggle("Show all stats (not just scoring inputs)", value=False)
    stat_cols = numeric_cols if show_all else selected
    by_team = df.set_index("Team")
    rows = []
    for stat in stat_cols:
        low_good = lower_is_better(stat)
        ranks = by_team[stat].rank(ascending=low_good, method="min")
        va, vb = by_team.at[team_a, stat], by_team.at[team_b, stat]
        if pd.isna(va) or pd.isna(vb) or va == vb:
            edge = "Even"
        else:
            edge = team_a if (va < vb) == low_good else team_b
        rows.append({
            "Stat": stat + (" ↓" if low_good else ""),
            team_a: round(float(va), 3) if pd.notna(va) else None,
            team_b: round(float(vb), 3) if pd.notna(vb) else None,
            f"{team_a} rank": int(ranks[team_a]) if pd.notna(ranks[team_a]) else None,
            f"{team_b} rank": int(ranks[team_b]) if pd.notna(ranks[team_b]) else None,
            "Edge": edge,
        })
    stats_table = pd.DataFrame(rows)
    st.dataframe(stats_table, width="stretch", hide_index=True)
    edges = stats_table["Edge"].value_counts()
    st.caption(
        f"Edge count: {team_a} {edges.get(team_a, 0)}, {team_b} {edges.get(team_b, 0)}, "
        f"even {edges.get('Even', 0)}. ↓ = lower is better. Rank is out of {len(by_team)} teams."
    )

    # ---- Standardized profile chart ----
    n_teams = int(by_team[selected].notna().any(axis=1).sum())
    z = ranked.set_index("Team").loc[[team_a, team_b], selected]
    z_long = z.reset_index().melt(id_vars="Team", var_name="Stat", value_name="z")
    rank_lbl, raw_val = [], []
    for team, stat in zip(z_long["Team"], z_long["Stat"]):
        col_ = by_team[stat]
        r = col_.rank(ascending=lower_is_better(stat), method="min")
        rk = r[team]
        tied = pd.notna(rk) and int((r == rk).sum()) > 1
        rank_lbl.append("—" if pd.isna(rk) else f"{'T-' if tied else '#'}{int(rk)}")
        v = col_[team]
        raw_val.append("—" if pd.isna(v) else (f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.3g}"))
    z_long["RankLbl"], z_long["Raw"] = rank_lbl, raw_val
    z_long["SD"] = [("—" if pd.isna(v) else f"{v:+.3g}") for v in z_long["z"]]
    fig = px.bar(
        z_long, y="Stat", x="z", color="Team", orientation="h", barmode="group",
        color_discrete_map=color_map, text="RankLbl", custom_data=["RankLbl", "Raw", "SD"],
        category_orders={"Stat": selected[::-1], "Team": [team_a, team_b]},
        labels={"z": "Standard deviations from league average (right = better)", "Stat": ""},
        title=f"Scoring inputs vs league average (labels = league rank of {n_teams})",
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_traces(
        marker_line_width=0, textposition="outside", cliponaxis=False,
        textfont=dict(size=13), constraintext="none",
        hovertemplate="<b>%{y}</b><br>Rank %{customdata[0]} of " + str(n_teams)
                      + "<br>Value %{customdata[1]}<br>%{customdata[2]} SD from average"
                      "<extra>%{fullData.name}</extra>",
    )
    span = float(z_long["z"].abs().max() or 1)
    fig.update_layout(height=120 + 50 * len(selected), legend_title_text="", bargap=0.3,
                      bargroupgap=0.08,
                      xaxis=dict(range=[-span * 1.25 - 0.3, span * 1.25 + 0.3]))
    st.plotly_chart(fig, width="stretch")
    st.caption("Rank 1 = best in the league for that stat, using the same direction as the table "
               "above (↓ stats rank lowest first). T- marks a tie.")

    # ---- Schedule & game results ----
    st.markdown("#### Schedule and results")
    if log is None:
        st.info("Turn on “Load game results” in the sidebar to see schedules and game results.")
    else:
        sched_cols = [c for c in SCHEDULE_COLS if c in by_team.columns]
        g_a, g_b = st.columns(2)
        for col, team in ((g_a, team_a), (g_b, team_b)):
            with col:
                st.markdown(f"**{team}**")
                if sched_cols:
                    for tile, stat in zip(st.columns(len(sched_cols)), sched_cols):
                        val = by_team.at[team, stat]
                        tile.metric(stat, "—" if pd.isna(val) else
                                    (f"{val:+.1f}" if stat == "OppMargin" else f"{val:.3f}"))
                table = game_table(log, team, records, scores_by_team)
                st.dataframe(table, width="stretch", hide_index=True,
                             height=35 * (len(table) + 1) + 3)

        # Point margin by week
        margins = log[log["Team"].isin([team_a, team_b]) & log["Completed"]].copy()
        margins["Score"] = margins["PointsFor"].astype(str) + "-" + margins["PointsAgainst"].astype(str)
        fig = px.bar(
            margins, x="Week", y="Margin", color="Team", barmode="group",
            color_discrete_map=color_map, category_orders={"Team": [team_a, team_b]},
            custom_data=["Opponent", "Score", "Result"],
            labels={"Margin": "Point margin"}, title="Point margin by week",
        )
        fig.update_traces(
            marker_line_width=0,
            hovertemplate="Week %{x}: %{customdata[2]} %{customdata[1]} vs %{customdata[0]}"
                          "<extra>%{fullData.name}</extra>",
        )
        fig.add_hline(y=0, line_width=1, line_color="gray")
        fig.update_layout(height=380, legend_title_text="", xaxis_dtick=1, bargap=0.25, bargroupgap=0.08)
        st.plotly_chart(fig, width="stretch")

        # Head-to-head
        h2h = log[(log["Team"] == team_a) & (log["Opponent"] == team_b)]
        st.markdown("#### Head-to-head")
        if h2h.empty:
            st.caption("These teams don't meet this regular season.")
        for _, g in h2h.iterrows():
            where = "home" if g["Home"] else "away"
            if g["Completed"]:
                winner = team_a if g["Result"] == "W" else team_b if g["Result"] == "L" else "Tie"
                st.write(f"Week {g['Week']} ({team_a} {where}): {team_a} {g['PointsFor']}, "
                         f"{team_b} {g['PointsAgainst']}. **{winner}**")
            else:
                st.write(f"Week {g['Week']} ({team_a} {where}): not played yet.")

        # Common opponents
        st.markdown("#### Common opponents")
        played = log[log["Completed"]]
        opp_a = played[played["Team"] == team_a]
        opp_b = played[played["Team"] == team_b]
        common = sorted((set(opp_a["Opponent"]) & set(opp_b["Opponent"])) - {team_a, team_b})
        if not common:
            st.caption("No common opponents played yet.")
        else:
            def results_vs(t: pd.DataFrame, opp: str) -> str:
                g = t[t["Opponent"] == opp]
                return ", ".join(f"{r} {pf}-{pa}" for r, pf, pa in
                                 zip(g["Result"], g["PointsFor"], g["PointsAgainst"]))

            common_tbl = pd.DataFrame({
                "Opponent": common,
                team_a: [results_vs(opp_a, o) for o in common],
                team_b: [results_vs(opp_b, o) for o in common],
            })
            st.dataframe(common_tbl, width="stretch", hide_index=True)
            ca_ = opp_a[opp_a["Opponent"].isin(common)]
            cb_ = opp_b[opp_b["Opponent"].isin(common)]
            st.caption(
                f"Vs common opponents: {team_a} {record_str(ca_['Result'])} "
                f"(avg margin {ca_['Margin'].mean():+.1f}), "
                f"{team_b} {record_str(cb_['Result'])} (avg margin {cb_['Margin'].mean():+.1f})."
            )
