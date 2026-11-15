import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIG + THEME
# ============================================================

st.set_page_config(
    page_title="MLBL | League Analytics",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#14263D"
BLUE = "#2F6B9A"
GOLD = "#E2B45B"
MUTED = "#718096"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Oswald:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #F4F6F9; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px; }
    h1, h2, h3 { color: #14263D; }
    h1 { font-family: 'Oswald', sans-serif !important; letter-spacing: .5px; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #E5EAF0; padding: 16px 18px;
        border-radius: 12px; box-shadow: 0 2px 8px rgba(20,38,61,.04);
    }
    div[data-testid="stMetricLabel"] { color: #718096; }
    div[data-testid="stMetricValue"] { color: #14263D; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    .hero {
        background: linear-gradient(115deg, #14263D 0%, #234B70 100%);
        padding: 26px 30px; border-radius: 16px; color: white; margin-bottom: 22px;
    }
    .hero h1 { color: white !important; font-size: 2.25rem; margin: 0; }
    .hero p { color: #D7E3EF; margin: 6px 0 0 0; }
    .eyebrow { color: #E2B45B; font-size: .75rem; font-weight: 700; letter-spacing: 2px; }
    .small-note { color: #718096; font-size: .82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING + CLEANING
# ============================================================

PLAYER_REQUIRED = [
    "Number", "Name", "Team", "G", "AB", "R", "H", "2B", "3B",
    "HR", "RBI", "BB", "SO", "HBP", "SB", "CS", "ROE",
    "AVG", "OBP", "SLG", "OPS",
]
TEAM_REQUIRED = [
    "Team", "G", "AB", "R", "H", "2B", "3B", "HR", "RBI",
    "BB", "SO", "HBP", "SB", "CS", "ROE",
    "AVG", "OBP", "SLG", "OPS",
]
COUNT_COLS = [
    "Number", "G", "AB", "R", "H", "2B", "3B", "HR", "RBI",
    "BB", "SO", "HBP", "SB", "CS", "ROE",
]
RATE_COLS = ["AVG", "OBP", "SLG", "OPS"]


def clean_frame(df: pd.DataFrame, required: list[str], label: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Remove blank export columns such as "Unnamed: 3".
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    # Trim text fields.
    for col in ["Name", "Team"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} CSV is missing required columns: {', '.join(missing)}")
    numeric_cols = [c for c in COUNT_COLS + RATE_COLS if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in RATE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def read_csv_source(uploaded_file, default_path: str, required: list[str], label: str):
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name
    elif Path(default_path).exists():
        raw = pd.read_csv(default_path)
        source_name = Path(default_path).name
    else:
        return None, None
    return clean_frame(raw, required, label), source_name


def add_derived_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TB"] = out["H"] + out["2B"] + 2 * out["3B"] + 3 * out["HR"]
    out["PA_est"] = out["AB"] + out["BB"] + out["HBP"]
    out["ISO"] = out["SLG"] - out["AVG"]
    out["XBH"] = out["2B"] + out["3B"] + out["HR"]
    out["BB_pct"] = np.where(out["PA_est"] > 0, out["BB"] / out["PA_est"], np.nan)
    out["K_pct"] = np.where(out["PA_est"] > 0, out["SO"] / out["PA_est"], np.nan)
    out["BB_K"] = np.where(out["SO"] > 0, out["BB"] / out["SO"], np.nan)
    out["HR_per_G"] = np.where(out["G"] > 0, out["HR"] / out["G"], np.nan)
    out["RBI_per_G"] = np.where(out["G"] > 0, out["RBI"] / out["G"], np.nan)
    out["R_per_G"] = np.where(out["G"] > 0, out["R"] / out["G"], np.nan)
    attempts = out["SB"] + out["CS"]
    out["SB_pct"] = np.where(attempts > 0, out["SB"] / attempts, np.nan)
    out["Net_SB"] = out["SB"] - out["CS"]
    babip_den = out["AB"] - out["SO"] - out["HR"]
    out["BABIP_est"] = np.where(
        babip_den > 0, (out["H"] - out["HR"]) / babip_den, np.nan
    )
    return out


def calculate_league_baseline(player_df: pd.DataFrame):
    # Mirrors the notebook's approach: use all player records for the league baseline.
    total_ab = player_df["AB"].sum()
    total_pa = (player_df["AB"] + player_df["BB"] + player_df["HBP"]).sum()
    if total_ab <= 0 or total_pa <= 0:
        raise ValueError("Cannot calculate league OPS+ baseline: league AB/PA totals are zero.")
    lg_obp = (
        player_df["H"].sum() + player_df["BB"].sum() + player_df["HBP"].sum()
    ) / total_pa
    total_tb = (
        player_df["H"].sum()
        + player_df["2B"].sum()
        + 2 * player_df["3B"].sum()
        + 3 * player_df["HR"].sum()
    )
    lg_slg = total_tb / total_ab
    return float(lg_obp), float(lg_slg)


def add_ops_plus(df: pd.DataFrame, lg_obp: float, lg_slg: float) -> pd.DataFrame:
    out = df.copy()
    out["OPS_plus"] = 100 * (
        out["OBP"] / lg_obp + out["SLG"] / lg_slg - 1
    )
    return out


# ============================================================
# SIDEBAR: INPUTS + FILTERS
# ============================================================

st.sidebar.markdown("## ⚾ MLBL Analytics")
st.sidebar.caption("Load the two league CSV exports")

players_upload = st.sidebar.file_uploader(
    "Player statistics CSV", type=["csv"], key="players_csv"
)
teams_upload = st.sidebar.file_uploader(
    "Team statistics CSV", type=["csv"], key="teams_csv"
)

st.sidebar.divider()
min_ab = st.sidebar.number_input(
    "Minimum at-bats for player leaderboard",
    min_value=0, max_value=1000, value=10, step=1,
)
st.sidebar.caption(
    "The qualification threshold controls who appears in player rankings. "
    "The league OPS+ baseline uses all player records, matching the notebook's latest calculation."
)

try:
    players_raw, players_source = read_csv_source(
        players_upload, "Book1.csv", PLAYER_REQUIRED, "Player"
    )
    teams_raw, teams_source = read_csv_source(
        teams_upload, "Book2.csv", TEAM_REQUIRED, "Team"
    )
except Exception as exc:
    st.error(f"Could not read the uploaded data: {exc}")
    st.stop()

if players_raw is None or teams_raw is None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">MLBL · SEMI-PRO DIVISION</div>
          <h1>LEAGUE ANALYTICS</h1>
          <p>Upload the player and team CSV files in the sidebar to open the dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Expected player file: Book1.csv. Expected team file: Book2.csv. "
        "You can upload files directly here; alternatively place both files beside app.py."
    )
    st.stop()

players = add_derived_stats(players_raw)
teams = add_derived_stats(teams_raw)

try:
    lg_obp, lg_slg = calculate_league_baseline(players)
except Exception as exc:
    st.error(str(exc))
    st.stop()

players = add_ops_plus(players, lg_obp, lg_slg)
teams = add_ops_plus(teams, lg_obp, lg_slg)

qualified = players.loc[players["AB"] >= min_ab].copy()
qualified["OPS+ Rank"] = (
    qualified["OPS_plus"].rank(method="min", ascending=False).astype(int)
)
qualified = qualified.sort_values(
    ["OPS_plus", "AB"], ascending=[False, False]
).reset_index(drop=True)

team_ranked = teams.copy()
team_ranked["OPS+ Rank"] = (
    team_ranked["OPS_plus"].rank(method="min", ascending=False).astype(int)
)
team_ranked = team_ranked.sort_values("OPS_plus", ascending=False).reset_index(drop=True)

team_options = sorted(players["Team"].dropna().astype(str).unique().tolist())
st.sidebar.divider()
selected_teams = st.sidebar.multiselect(
    "Filter players by team",
    options=team_options,
    default=[],
    help="Leave empty to include every team.",
)
if selected_teams:
    qualified_view = qualified[qualified["Team"].isin(selected_teams)].copy()
else:
    qualified_view = qualified.copy()

# ============================================================
# HEADER + KPI STRIP
# ============================================================

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">MLBL · SEMI-PRO DIVISION</div>
      <h1>LEAGUE ANALYTICS</h1>
      <p>Player leaderboards, team performance and offensive trends · Data: {players_source} + {teams_source}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Players in dataset", f"{len(players):,}")
k2.metric("Qualified players", f"{len(qualified):,}", f"≥ {min_ab} AB")
k3.metric("Teams", f"{teams['Team'].nunique():,}")
k4.metric("League OPS", f"{(lg_obp + lg_slg):.3f}")

st.caption(
    f"League baseline: OBP {lg_obp:.3f} · SLG {lg_slg:.3f} · "
    "OPS+ is a simplified league-adjusted measure without park factors."
)

# ============================================================
# TABS
# ============================================================

tab_overview, tab_players, tab_teams, tab_distributions, tab_analysis, tab_data = st.tabs(
    ["Overview", "Player leaderboard", "Team rankings", "Distributions", "Analysis", "Raw data"]
)

# ------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------
with tab_overview:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Top offensive performers")
        top = qualified_view.head(10).copy()
        if not top.empty:
            fig = px.bar(
                top.sort_values("OPS_plus", ascending=True),
                x="OPS_plus",
                y="Name",
                color="Team",
                orientation="h",
                text=top.sort_values("OPS_plus", ascending=True)["OPS_plus"].round(0).astype(int),
                hover_data={"AB": True, "AVG": ":.3f", "OBP": ":.3f", "SLG": ":.3f", "OPS_plus": ":.1f"},
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig.add_vline(x=100, line_dash="dash", line_color="#C58B2A", annotation_text="League avg")
            fig.update_layout(
                height=430, template="plotly_white", margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="OPS+", yaxis_title="", legend_title="Team",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No players meet the current at-bat threshold and team filter.")

    with right:
        st.subheader("Team OPS+")
        team_fig = px.bar(
            team_ranked.sort_values("OPS_plus", ascending=True),
            x="OPS_plus", y="Team", orientation="h",
            text=team_ranked.sort_values("OPS_plus", ascending=True)["OPS_plus"].round(0).astype(int),
            color="OPS_plus", color_continuous_scale=["#DCE6EF", "#2F6B9A", "#14263D"],
            hover_data={"AVG": ":.3f", "OBP": ":.3f", "SLG": ":.3f", "OPS": ":.3f"},
        )
        team_fig.add_vline(x=100, line_dash="dash", line_color="#C58B2A")
        team_fig.update_layout(
            height=430, template="plotly_white", coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10), xaxis_title="OPS+", yaxis_title="",
        )
        st.plotly_chart(team_fig, use_container_width=True)

    st.subheader("League offensive profile")
    metric_cols = st.columns(4)
    for col, metric, label in zip(metric_cols, ["AVG", "OBP", "SLG", "OPS"], ["Batting average", "On-base %", "Slugging %", "OPS"]):
        col.metric(label, f"{players[metric].mean():.3f}")

    st.markdown(
        '<p class="small-note">Rate-stat averages shown here are unweighted player averages for descriptive context. '
        'OPS+ baseline uses aggregate player totals.</p>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# PLAYER LEADERBOARD
# ------------------------------------------------------------
with tab_players:
    st.subheader("Player OPS+ leaderboard")
    search = st.text_input(
        "Search player",
        placeholder="Search by player name, team or jersey number...",
        key="player_search",
    )
    sort_options = {
        "OPS+": "OPS_plus",
        "Batting average": "AVG",
        "On-base percentage": "OBP",
        "Slugging percentage": "SLG",
        "OPS": "OPS",
        "Home runs": "HR",
        "RBI": "RBI",
        "Stolen bases": "SB",
        "Walk rate": "BB_pct",
        "ISO (power)": "ISO",
    }
    sort_label = st.selectbox("Rank by", list(sort_options.keys()), index=0)
    sort_col = sort_options[sort_label]
    ascending = False  # Higher values rank first for all selected metrics.
    result = qualified_view.copy()

    if search.strip():
        q = search.strip().lower()
        mask = (
            result["Name"].astype(str).str.lower().str.contains(q, na=False)
            | result["Team"].astype(str).str.lower().str.contains(q, na=False)
            | result["Number"].astype(str).str.lower().str.contains(q, na=False)
        )
        result = result.loc[mask]

    result = result.sort_values(sort_col, ascending=ascending, na_position="last")
    result["Display Rank"] = result[sort_col].rank(
        ascending=ascending, method="min"
    ).astype("Int64")

    display_cols = [
        "Display Rank", "Number", "Name", "Team", "G", "AB", "H",
        "HR", "RBI", "BB", "SO", "SB", "AVG", "OBP", "SLG", "OPS", "OPS_plus",
    ]
    display_df = result[display_cols].rename(columns={"OPS_plus": "OPS+"})
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=560,
        column_config={
            "Display Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Number": st.column_config.NumberColumn("#", format="%d"),
            "AVG": st.column_config.NumberColumn("AVG", format="%.3f"),
            "OBP": st.column_config.NumberColumn("OBP", format="%.3f"),
            "SLG": st.column_config.NumberColumn("SLG", format="%.3f"),
            "OPS": st.column_config.NumberColumn("OPS", format="%.3f"),
            "OPS+": st.column_config.NumberColumn("OPS+", format="%.0f"),
        },
    )

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered leaderboard as CSV",
        data=csv_bytes,
        file_name="mlbl_player_leaderboard.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Player profile")
    profile_options = result["Name"].dropna().astype(str).tolist()
    if profile_options:
        chosen_name = st.selectbox("Select a player", profile_options)
        p = result[result["Name"].astype(str) == chosen_name].iloc[0]
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("OPS+", f"{p['OPS_plus']:.0f}")
        p2.metric("AVG", f"{p['AVG']:.3f}")
        p3.metric("OBP", f"{p['OBP']:.3f}")
        p4.metric("SLG", f"{p['SLG']:.3f}")
        p5.metric("HR", f"{int(p['HR'])}")
        st.caption(
            f"#{int(p['Number'])} · {p['Team']} · {int(p['G'])} games · {int(p['AB'])} AB"
        )
    else:
        st.info("No players match the current search and filters.")

# ------------------------------------------------------------
# TEAM RANKINGS
# ------------------------------------------------------------
with tab_teams:
    st.subheader("Team offensive rankings")
    team_table = team_ranked[
        ["OPS+ Rank", "Team", "G", "AB", "R", "H", "HR", "RBI",
         "AVG", "OBP", "SLG", "OPS", "OPS_plus"]
    ].rename(columns={"OPS_plus": "OPS+"})
    st.dataframe(
        team_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "AVG": st.column_config.NumberColumn(format="%.3f"),
            "OBP": st.column_config.NumberColumn(format="%.3f"),
            "SLG": st.column_config.NumberColumn(format="%.3f"),
            "OPS": st.column_config.NumberColumn(format="%.3f"),
            "OPS+": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    team_compare = teams.melt(
        id_vars="Team", value_vars=["AVG", "OBP", "SLG"],
        var_name="Metric", value_name="Rate",
    )
    fig = px.bar(
        team_compare, x="Team", y="Rate", color="Metric", barmode="group",
        title="Team batting profile: AVG, OBP and SLG",
        color_discrete_sequence=["#2F6B9A", "#E2B45B", "#14263D"],
    )
    fig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# DISTRIBUTIONS
# ------------------------------------------------------------
with tab_distributions:
    st.subheader("League-wide batting distributions")
    dist_source = qualified_view if not qualified_view.empty else qualified
    if dist_source.empty:
        st.warning("No qualified players available for distributions.")
    else:
        selected_metrics = st.multiselect(
            "Select statistics",
            ["AVG", "OBP", "SLG", "OPS", "OPS_plus", "ISO", "BB_pct", "K_pct", "HR", "RBI", "SB"],
            default=["AVG", "OBP", "SLG", "OPS"],
        )
        if selected_metrics:
            dist_long = dist_source[selected_metrics].melt(
                var_name="Statistic", value_name="Value"
            )
            fig = px.histogram(
                dist_long, x="Value", facet_col="Statistic",
                facet_col_wrap=2, facet_col_spacing=0.10,
                facet_row_spacing=0.14, color="Statistic",
                nbins=18, opacity=0.82,
                color_discrete_sequence=["#2F6B9A"],
                title="Player statistic distributions",
            )
            fig.update_layout(
                template="plotly_white", height=max(450, 260 * ((len(selected_metrics) + 1) // 2)),
                showlegend=False, margin=dict(l=15, r=15, t=65, b=15),
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Choose at least one statistic.")

        st.markdown("#### OPS+ distribution")
        fig2 = px.histogram(
            dist_source, x="OPS_plus", nbins=16, marginal="box",
            color_discrete_sequence=["#2F6B9A"],
        )
        fig2.add_vline(x=100, line_dash="dash", line_color="#C58B2A", annotation_text="League average")
        fig2.update_layout(template="plotly_white", height=350, xaxis_title="OPS+")
        st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# RELATIONSHIPS / ANALYSIS
# ------------------------------------------------------------
with tab_analysis:
    st.subheader("Offensive profile and relationships")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(
            qualified_view, x="AVG", y="ISO", color="Team",
            size="HR", hover_name="Name",
            hover_data={"OPS_plus": ":.1f", "AB": True, "OBP": ":.3f", "SLG": ":.3f"},
            title="Batting average vs. isolated power",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_layout(template="plotly_white", height=440)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            qualified_view, x="BB_pct", y="K_pct", color="Team",
            size="PA_est", hover_name="Name",
            hover_data={"OPS_plus": ":.1f", "BB": True, "SO": True},
            title="Plate discipline: walk rate vs. strikeout rate",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_layout(template="plotly_white", height=440)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(
            qualified_view, x="HR", y="RBI", color="Team",
            size="R", hover_name="Name",
            hover_data={"OPS_plus": ":.1f", "AB": True},
            title="Home runs vs. RBI production",
        )
        fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.scatter(
            qualified_view, x="AB", y="OPS_plus", color="Team",
            size="H", hover_name="Name",
            hover_data={"AVG": ":.3f", "OBP": ":.3f", "SLG": ":.3f"},
            title="OPS+ vs. at-bat volume",
        )
        fig.add_hline(y=100, line_dash="dash", line_color="#C58B2A")
        fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Team offensive production")
    team_metrics = teams.melt(
        id_vars="Team", value_vars=["R", "RBI", "HR", "XBH"],
        var_name="Metric", value_name="Total",
    )
    fig = px.bar(
        team_metrics, x="Team", y="Total", color="Metric", barmode="group",
        title="Runs, RBI, home runs and extra-base hits by team",
    )
    fig.update_layout(template="plotly_white", height=430)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# RAW DATA
# ------------------------------------------------------------
with tab_data:
    st.subheader("Source data")
    st.caption(f"Player source: {players_source} · Team source: {teams_source}")
    with st.expander("Player CSV", expanded=False):
        st.dataframe(players_raw, use_container_width=True, hide_index=True)
        st.download_button(
            "Download cleaned player data",
            players_raw.to_csv(index=False).encode("utf-8"),
            "players_cleaned.csv",
            "text/csv",
        )
    with st.expander("Team CSV", expanded=False):
        st.dataframe(teams_raw, use_container_width=True, hide_index=True)
        st.download_button(
            "Download cleaned team data",
            teams_raw.to_csv(index=False).encode("utf-8"),
            "teams_cleaned.csv",
            "text/csv",
        )

st.divider()
st.markdown(
    """
    <p class="small-note">
    OPS+ methodology: 100 × (player OBP / league OBP + player SLG / league SLG − 1).
    League OBP and SLG are calculated from aggregate player totals. This is a simplified,
    non-park-adjusted league comparison, not the official MLB park-adjusted OPS+ formula.
    PA is estimated as AB + BB + HBP because sacrifice flies and intentional walks are not
    present in the source data. BABIP and other derived measures should be treated as estimates.
    </p>
    """,
    unsafe_allow_html=True,
)
