
import streamlit as st
import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -------------------------------------------------------------------
# Constants for Google Sheets
# -------------------------------------------------------------------
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE = 'token_statsperform.pickle'
SPREADSHEET_ID = '14KDGCP2t6n8Sm096Kf0YQz6KHYSvhiKKnC0Q409rp2Q'
RANGE_NAME = 'Feuille 1'

# Physical metrics
DISTANCE_COLS = [
    "TotalDistanceRun",
    "WalkDist",
    "JogDist",
    "RunDist",
    "HiSpeedRunDist",
    "SprintDist",
]

COUNT_COLS = [
    "WalkCount",
    "JogCount",
    "RunCount",
    "HiSpeedRunCount",
    "SprintCount",
]

# -------------------------------------------------------------------
# Page config & header
# -------------------------------------------------------------------
st.set_page_config(page_title="StatsPerform | FC Versailles", layout='wide')

logo_url = 'https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png'
col1, col2 = st.columns([9, 1])
with col1:
    st.title("StatsPerform | FC Versailles")
with col2:
    st.image(logo_url, use_container_width=True)

st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Google Sheets auth
# -------------------------------------------------------------------
def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    return creds


def fetch_google_sheet(spreadsheet_id, range_name):
    creds = get_credentials()
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])
    if not values:
        st.error("No data found in the specified range.")
        return pd.DataFrame()
    header = values[0]
    data = values[1:]
    max_columns = len(header)
    adjusted_data = [
        row + [None] * (max_columns - len(row)) if len(row) < max_columns else row[:max_columns]
        for row in data
    ]
    return pd.DataFrame(adjusted_data, columns=header)


@st.cache_data(show_spinner="Loading data...")
def load_data(ttl=60):
    return fetch_google_sheet(SPREADSHEET_ID, RANGE_NAME)


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def detect_player_col(df: pd.DataFrame):
    candidates = ["playerName", "PlayerName", "athleteName", "AthleteName", "fullName", "FullName"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_position_col(df: pd.DataFrame):
    candidates = ["position", "Position", "positionName", "PositionName", "role", "Role", "PlayingPositionName"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def format_metric_name(col: str) -> str:
    # Simple human-readable label from column name
    return col.replace("_per90", "").replace("_", " ")


# -------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------
@st.cache_data(show_spinner="Preprocessing data...")
def preprocess_data(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Core identifiers
    if "teamName" not in df.columns:
        raise ValueError("Missing required column 'teamName' in the dataset.")

    if "teamShortName" not in df.columns:
        df["teamShortName"] = df["teamName"]

    # League
    if "newestLeague" not in df.columns:
        league_col = None
        for cand in ["league", "League", "newestLeagueId"]:
            if cand in df.columns:
                league_col = cand
                break
        if league_col is not None:
            df["newestLeague"] = df[league_col].astype(str)
        else:
            df["newestLeague"] = "Unknown"
    df["newestLeague"] = df["newestLeague"].astype(str)

    # Match ID / game ID
    if "matchId" not in df.columns:
        if "gameId" in df.columns:
            df["matchId"] = df["gameId"]
        else:
            df["matchId"] = np.nan

    # Match date
    if "matchDate" not in df.columns:
        if "date" in df.columns:
            df["matchDate"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["matchDate"] = pd.NaT
    else:
        df["matchDate"] = pd.to_datetime(df["matchDate"], errors="coerce")

    # Opponent
    if "opponentTeamName" not in df.columns:
        opp_col = None
        for cand in ["opponentTeamName", "opponentFullName", "opponent"]:
            if cand in df.columns:
                opp_col = cand
                break
        if opp_col is not None:
            df["opponentTeamName"] = df[opp_col].astype(str)
        else:
            df["opponentTeamName"] = ""

    # Match key (robust ID)
    if df["matchId"].notna().any():
        df["match_key"] = df["matchId"].astype(str)
    else:
        df["match_key"] = (
            df["teamName"].astype(str)
            + "_"
            + df["matchDate"].astype(str)
            + "_"
            + df["opponentTeamName"].astype(str)
        )

    # Minutes
    if "Min" not in df.columns:
        raise ValueError("Missing required column 'Min' (minutes played) in the dataset.")
    df["Min"] = pd.to_numeric(df["Min"], errors="coerce").fillna(0)

    # Physical metrics
    metric_cols = []
    for col in DISTANCE_COLS + COUNT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            metric_cols.append(col)

    if not metric_cols:
        raise ValueError("No known physical metric columns found in the dataset.")

    # Create per90 metrics (respecting rules: ignore Min <= 0 and metric == 0)
    per90_cols = []
    for col in metric_cols:
        per90_name = f"{col}_per90"
        df[per90_name] = np.where(
            (df["Min"] > 0) & (df[col] > 0),
            df[col] / df["Min"] * 90,
            np.nan,  # we use NaN so that means ignore these lines in averages
        )
        per90_cols.append(per90_name)

    distance_present = [c for c in DISTANCE_COLS if c in metric_cols]
    count_present = [c for c in COUNT_COLS if c in metric_cols]
    distance_per90_cols = [f"{c}_per90" for c in distance_present]
    count_per90_cols = [f"{c}_per90" for c in count_present]

    player_col = detect_player_col(df)
    position_col = detect_position_col(df)

    return {
        "players": df,
        "metric_cols": metric_cols,
        "per90_cols": per90_cols,
        "distance_cols": distance_present,
        "count_cols": count_present,
        "distance_per90_cols": distance_per90_cols,
        "count_per90_cols": count_per90_cols,
        "player_col": player_col,
        "position_col": position_col,
    }


# -------------------------------------------------------------------
# Load & preprocess
# -------------------------------------------------------------------
data = load_data()

try:
    data_dict = preprocess_data(data)
except Exception as e:
    st.error(f"Error while preprocessing data: {e}")
    st.stop()

df_players = data_dict["players"]
metric_cols = data_dict["metric_cols"]
per90_cols = data_dict["per90_cols"]
distance_cols = data_dict["distance_cols"]
count_cols = data_dict["count_cols"]
distance_per90_cols = data_dict["distance_per90_cols"]
count_per90_cols = data_dict["count_per90_cols"]
player_col = data_dict["player_col"]
position_col = data_dict["position_col"]

# We will always work on rows where Min > 0 (player a joué)
df_players_min = df_players[df_players["Min"] > 0].copy()

# -------------------------------------------------------------------
# Sidebar: ONLY league / saison filters (no performance filters)
# -------------------------------------------------------------------
st.sidebar.header("Filtres league")

league_options = sorted(df_players_min["newestLeague"].dropna().astype(str).unique())
selected_leagues = st.sidebar.multiselect(
    "League (newestLeague)",
    options=league_options,
    default=league_options,
)

df_league = df_players_min[
    df_players_min["newestLeague"].astype(str).isin(selected_leagues)
].copy()

if df_league.empty:
    st.warning("Aucune donnée pour les leagues sélectionnées.")
    st.stop()

# Optional season filter if column exists (still contextual)
season_col = None
for cand in ["season", "Season", "seasonName", "SeasonName", "newestSeason"]:
    if cand in df_league.columns:
        season_col = cand
        break

if season_col is not None:
    season_options = sorted(df_league[season_col].dropna().astype(str).unique())
    if len(season_options) > 1:
        selected_seasons = st.sidebar.multiselect(
            "Saison",
            options=season_options,
            default=season_options,
        )
        df_league = df_league[
            df_league[season_col].astype(str).isin(selected_seasons)
        ].copy()

if df_league.empty:
    st.warning("Aucune donnée après filtrage league / saison.")
    st.stop()

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_overview, tab_team, tab_player = st.tabs(
    ["Overview (Benchmark league)", "Team Analytics", "Player / Match Drilldown"]
)

# -------------------------------------------------------------------
# TAB 1 – Overview (Benchmark league)
# -------------------------------------------------------------------
with tab_overview:
    st.subheader("Benchmark league – équipes & joueurs")

    # Team-level aggregation for selected leagues
    team_group_cols = ["newestLeague", "teamShortName", "teamName"]
    for col in team_group_cols:
        if col not in df_league.columns:
            df_league[col] = ""

    # Matches played per team
    match_counts = (
        df_league.groupby(team_group_cols, dropna=False)["match_key"]
        .nunique()
        .reset_index()
        .rename(columns={"match_key": "MatchesPlayed"})
    )

    # Mean per90 metrics per team (players ayant joué & métriques > 0 -> déjà géré via NaN)
    team_means = (
        df_league.groupby(team_group_cols, dropna=False)[per90_cols]
        .mean()
        .reset_index()
    )

    df_team_agg = team_means.merge(match_counts, on=team_group_cols, how="left")
    df_team_agg["MatchesPlayed"] = df_team_agg["MatchesPlayed"].fillna(0).astype(int)

    # KPIs league
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Nombre d'équipes", len(df_team_agg))
    with kpi2:
        st.metric("Nombre total de matchs", int(df_team_agg["MatchesPlayed"].sum()))
    with kpi3:
        if "TotalDistanceRun_per90" in df_team_agg.columns:
            league_avg_td = df_team_agg["TotalDistanceRun_per90"].mean()
            st.metric(
                "Distance moyenne league (TotalDistanceRun_per90)",
                f"{league_avg_td:.1f}",
            )
        else:
            st.metric("Distance moyenne league", "N/A")

    st.markdown("---")

    # Choix niveau d'analyse
    view_level = st.radio(
        "Niveau d'analyse",
        ["Équipes", "Joueurs par poste"],
        horizontal=True,
    )

    # ---------------------------
    # Vue ÉQUIPES
    # ---------------------------
    if view_level == "Équipes":
        if not per90_cols:
            st.warning("Aucune métrique _per90 disponible.")
        else:
            default_metric_team = (
                "TotalDistanceRun_per90" if "TotalDistanceRun_per90" in per90_cols else per90_cols[0]
            )
            metric_team = st.selectbox(
                "Métrique par 90 à comparer (équipes)",
                options=per90_cols,
                index=per90_cols.index(default_metric_team),
                format_func=format_metric_name,
            )

            max_teams = max(5, len(df_team_agg))
            top_n_teams = st.slider(
                "Top N équipes à afficher",
                min_value=5,
                max_value=max(5, min(20, max_teams)),
                value=min(10, max_teams),
            )

            df_plot = (
                df_team_agg
                .sort_values(metric_team, ascending=False)
                .head(top_n_teams)
            )

            fig = px.bar(
                df_plot,
                x="teamShortName",
                y=metric_team,
                color="newestLeague",
                hover_data=["teamName", "MatchesPlayed"],
                labels={
                    "teamShortName": "Équipe",
                    metric_team: format_metric_name(metric_team),
                    "newestLeague": "League",
                },
            )
            fig.update_layout(
                title=f"Top {top_n_teams} équipes – {format_metric_name(metric_team)}",
                xaxis_title="Équipe",
                yaxis_title=format_metric_name(metric_team),
                margin=dict(l=10, r=10, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Vue JOUEURS PAR POSTE
    # ---------------------------
    else:
        if player_col is None or position_col is None:
            st.info("Impossible d'afficher la vue joueurs par poste (colonnes joueur ou poste manquantes).")
        else:
            if not per90_cols:
                st.warning("Aucune métrique _per90 disponible.")
            else:
                default_metric_player = (
                    "HiSpeedRunDist_per90"
                    if "HiSpeedRunDist_per90" in per90_cols
                    else per90_cols[0]
                )
                metric_player = st.selectbox(
                    "Métrique par 90 (joueurs)",
                    options=per90_cols,
                    index=per90_cols.index(default_metric_player),
                    format_func=format_metric_name,
                )

                positions = sorted(
                    df_league[position_col].dropna().astype(str).unique()
                )
                if not positions:
                    st.info("Aucun poste disponible dans les données.")
                else:
                    selected_pos = st.selectbox(
                        "Poste",
                        options=positions,
                    )
                    top_n_players = st.slider(
                        "Top N joueurs à afficher",
                        min_value=5,
                        max_value=30,
                        value=10,
                    )

                    df_pos = df_league[
                        df_league[position_col].astype(str) == selected_pos
                    ].copy()

                    # Moyenne par joueur (et équipe) pour la métrique choisie
                    group_cols = [player_col, "teamShortName"]
                    df_players_pos = (
                        df_pos.groupby(group_cols, dropna=False)[metric_player]
                        .mean()
                        .reset_index()
                        .dropna(subset=[metric_player])
                    )

                    if df_players_pos.empty:
                        st.info("Aucun joueur pour ce poste / cette métrique.")
                    else:
                        df_players_pos = df_players_pos.sort_values(
                            metric_player, ascending=False
                        ).head(top_n_players)

                        fig = px.bar(
                            df_players_pos,
                            x=metric_player,
                            y=player_col,
                            orientation="h",
                            hover_data=["teamShortName"],
                            labels={
                                player_col: "Joueur",
                                metric_player: format_metric_name(metric_player),
                                "teamShortName": "Équipe",
                            },
                        )
                        fig.update_layout(
                            title=f"Top {top_n_players} joueurs ({selected_pos}) – {format_metric_name(metric_player)}",
                            xaxis_title=format_metric_name(metric_player),
                            yaxis_title="Joueur",
                            margin=dict(l=10, r=10, t=40, b=40),
                        )
                        fig.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# TAB 2 – Team Analytics
# -------------------------------------------------------------------
with tab_team:
    st.subheader("Analytics détaillée par équipe")

    teams_in_league = sorted(df_league["teamShortName"].dropna().astype(str).unique())
    if not teams_in_league:
        st.warning("Aucune équipe disponible.")
    else:
        default_team = "Versailles" if "Versailles" in teams_in_league else teams_in_league[0]
        selected_team = st.selectbox(
            "Sélectionner une équipe (teamShortName)",
            options=teams_in_league,
            index=teams_in_league.index(default_team),
        )

        df_team = df_league[df_league["teamShortName"] == selected_team].copy()

        if df_team.empty:
            st.warning("Aucune donnée pour cette équipe.")
        else:
            # -----------------------
            # Part 1 – KPIs équipe
            # -----------------------
            st.markdown("### KPIs équipe (par 90 min)")

            matches_played_team = df_team["match_key"].nunique()

            kpi_cols = st.columns(4)
            with kpi_cols[0]:
                st.metric("Matchs joués", matches_played_team)

            kpi_metrics = [
                "TotalDistanceRun_per90",
                "HiSpeedRunDist_per90",
                "SprintDist_per90",
            ]
            idx = 1
            for m in kpi_metrics:
                if m in per90_cols and idx < len(kpi_cols):
                    team_mean = df_team[m].mean(skipna=True)
                    league_mean = df_league[m].mean(skipna=True)
                    if pd.isna(league_mean) or league_mean == 0:
                        delta_str = "N/A"
                    else:
                        delta_pct = (team_mean - league_mean) / league_mean * 100
                        delta_str = f"{delta_pct:+.1f} %"
                    with kpi_cols[idx]:
                        st.metric(
                            format_metric_name(m),
                            f"{team_mean:.1f}",
                            delta=delta_str,
                        )
                    idx += 1

            st.markdown("---")

            # -----------------------
            # Part 2 – Match-by-match
            # -----------------------
            st.markdown("### Évolution match par match (équipe)")

            # Moyenne par match pour l'équipe (sur les joueurs ayant joué)
            match_group_cols = ["match_key", "matchDate", "opponentTeamName"]
            for c in match_group_cols:
                if c not in df_team.columns:
                    df_team[c] = np.nan

            df_team_matches = (
                df_team.groupby(match_group_cols, dropna=False)[per90_cols]
                .mean()
                .reset_index()
                .sort_values("matchDate")
            )

            if df_team_matches.empty:
                st.info("Aucune donnée match par match pour cette équipe.")
            else:
                default_metric_line = (
                    "TotalDistanceRun_per90"
                    if "TotalDistanceRun_per90" in per90_cols
                    else per90_cols[0]
                )
                metric_line = st.selectbox(
                    "Métrique à suivre match par match",
                    options=per90_cols,
                    index=per90_cols.index(default_metric_line),
                    format_func=format_metric_name,
                )

                league_mean_metric = df_league[metric_line].mean(skipna=True)

                fig_line = go.Figure()

                fig_line.add_trace(
                    go.Scatter(
                        x=df_team_matches["matchDate"],
                        y=df_team_matches[metric_line],
                        mode="lines+markers",
                        name=selected_team,
                        hovertext=df_team_matches["opponentTeamName"],
                        hovertemplate=(
                            "<b>%{x|%Y-%m-%d}</b><br>vs %{hovertext}<br>"
                            + f"{format_metric_name(metric_line)}: %{{y:.1f}}<extra></extra>"
                        ),
                    )
                )

                fig_line.add_trace(
                    go.Scatter(
                        x=df_team_matches["matchDate"],
                        y=[league_mean_metric] * len(df_team_matches),
                        mode="lines",
                        name="Moyenne league",
                        line=dict(dash="dash"),
                        hovertemplate=f"Moyenne league: {league_mean_metric:.1f}<extra></extra>",
                    )
                )

                fig_line.update_layout(
                    title=f"{format_metric_name(metric_line)} – {selected_team} vs moyenne league",
                    xaxis_title="Date du match",
                    yaxis_title=format_metric_name(metric_line),
                    margin=dict(l=10, r=10, t=40, b=40),
                )
                st.plotly_chart(fig_line, use_container_width=True)

            st.markdown("---")

            # -----------------------
            # Part 3 – Joueurs de l'équipe
            # -----------------------
            st.markdown("### Profil joueurs de l'équipe (par 90 min)")

            if player_col is None:
                st.info("Impossible d'agréger par joueur (colonne joueur manquante).")
            else:
                group_cols_players = [player_col]
                if position_col is not None:
                    group_cols_players.append(position_col)

                # Aggregation joueur: minutes totales + moyennes per90
                agg_dict = {"Min": "sum"}
                for c in per90_cols:
                    agg_dict[c] = "mean"

                df_team_players_agg = (
                    df_team.groupby(group_cols_players, dropna=False)
                    .agg(agg_dict)
                    .reset_index()
                    .rename(columns={"Min": "MinutesPlayed"})
                )

                if df_team_players_agg.empty:
                    st.info("Aucun joueur pour cette équipe.")
                else:
                    max_minutes = int(df_team_players_agg["MinutesPlayed"].max())
                    if max_minutes <= 0:
                        st.info("Tous les joueurs ont 0 minute.")
                    else:
                        default_min_val = min(300, max_minutes)
                        min_minutes = st.slider(
                            "Filtrer les joueurs par minutes jouées (≥)",
                            min_value=0,
                            max_value=max_minutes,
                            value=default_min_val,
                        )

                        df_team_players_agg = df_team_players_agg[
                            df_team_players_agg["MinutesPlayed"] >= min_minutes
                        ]

                        if df_team_players_agg.empty:
                            st.info("Aucun joueur ne dépasse le seuil de minutes choisi.")
                        else:
                            # Filtre par poste optionnel
                            if position_col is not None:
                                positions_team = sorted(
                                    df_team_players_agg[position_col]
                                    .dropna()
                                    .astype(str)
                                    .unique()
                                )
                                position_options = ["Tous postes"] + positions_team
                                selected_position = st.selectbox(
                                    "Filtrer par poste",
                                    options=position_options,
                                )
                                if selected_position != "Tous postes":
                                    df_team_players_agg = df_team_players_agg[
                                        df_team_players_agg[position_col].astype(str)
                                        == selected_position
                                    ]

                            if df_team_players_agg.empty:
                                st.info("Aucun joueur après filtrage par poste.")
                            else:
                                default_metric_team_players = (
                                    "TotalDistanceRun_per90"
                                    if "TotalDistanceRun_per90" in per90_cols
                                    else per90_cols[0]
                                )
                                metric_team_players = st.selectbox(
                                    "Métrique par 90 (joueurs de l'équipe)",
                                    options=per90_cols,
                                    index=per90_cols.index(default_metric_team_players),
                                    format_func=format_metric_name,
                                )

                                df_plot_players = df_team_players_agg[
                                    [player_col, "MinutesPlayed", metric_team_players]
                                    + ([position_col] if position_col is not None else [])
                                ].dropna(subset=[metric_team_players])

                                df_plot_players = df_plot_players.sort_values(
                                    metric_team_players, ascending=False
                                )

                                fig_players = px.bar(
                                    df_plot_players,
                                    x=player_col,
                                    y=metric_team_players,
                                    hover_data=["MinutesPlayed"]
                                    + ([position_col] if position_col is not None else []),
                                    labels={
                                        player_col: "Joueur",
                                        metric_team_players: format_metric_name(metric_team_players),
                                    },
                                )
                                fig_players.update_layout(
                                    title=f"{selected_team} – joueurs triés par {format_metric_name(metric_team_players)}",
                                    xaxis_title="Joueur",
                                    yaxis_title=format_metric_name(metric_team_players),
                                    margin=dict(l=10, r=10, t=40, b=40),
                                )
                                st.plotly_chart(fig_players, use_container_width=True)


# -------------------------------------------------------------------
# TAB 3 – Player / Match Drilldown
# -------------------------------------------------------------------
with tab_player:
    st.subheader("Drilldown joueur / match")

    if player_col is None:
        st.info("Impossible de faire le drill-down joueur (colonne joueur manquante).")
    else:
        teams_in_league = sorted(df_league["teamShortName"].dropna().astype(str).unique())
        if not teams_in_league:
            st.warning("Aucune équipe disponible.")
        else:
            default_team_drill = "Versailles" if "Versailles" in teams_in_league else teams_in_league[0]
            selected_team_drill = st.selectbox(
                "Équipe",
                options=teams_in_league,
                index=teams_in_league.index(default_team_drill),
            )

            df_team_drill = df_league[df_league["teamShortName"] == selected_team_drill].copy()

            players_in_team = sorted(
                df_team_drill[player_col].dropna().astype(str).unique()
            )
            if not players_in_team:
                st.info("Aucun joueur pour cette équipe.")
            else:
                selected_player = st.selectbox(
                    "Joueur",
                    options=players_in_team,
                )

                if not per90_cols:
                    st.warning("Aucune métrique _per90 disponible.")
                else:
                    default_metric_drill = (
                        "TotalDistanceRun_per90"
                        if "TotalDistanceRun_per90" in per90_cols
                        else per90_cols[0]
                    )
                    selected_metric_drill = st.selectbox(
                        "Métrique par 90 à analyser",
                        options=per90_cols,
                        index=per90_cols.index(default_metric_drill),
                        format_func=format_metric_name,
                    )

                    df_player_drill = df_team_drill[
                        df_team_drill[player_col].astype(str) == selected_player
                    ].copy()

                    if df_player_drill.empty:
                        st.info("Aucune donnée pour ce joueur.")
                    else:
                        # KPIs joueur
                        mean_metric_player = df_player_drill[selected_metric_drill].mean(skipna=True)
                        total_minutes_player = df_player_drill["Min"].sum()

                        if position_col is not None:
                            pos_series = df_player_drill[position_col].dropna().astype(str)
                            if not pos_series.empty:
                                player_position = pos_series.mode().iloc[0]
                            else:
                                player_position = None
                        else:
                            player_position = None

                        if player_position is not None and position_col is not None:
                            df_league_same_pos = df_league[
                                df_league[position_col].astype(str) == player_position
                            ]
                        else:
                            df_league_same_pos = df_league

                        league_avg_metric_pos = df_league_same_pos[selected_metric_drill].mean(skipna=True)

                        if pd.isna(league_avg_metric_pos) or league_avg_metric_pos == 0:
                            delta_pos_str = "N/A"
                        else:
                            delta_pos_pct = (mean_metric_player - league_avg_metric_pos) / league_avg_metric_pos * 100
                            delta_pos_str = f"{delta_pos_pct:+.1f} %"

                        kpi_cols_player = st.columns(3)
                        with kpi_cols_player[0]:
                            st.metric(
                                f"Moyenne {format_metric_name(selected_metric_drill)}",
                                f"{mean_metric_player:.1f}",
                            )
                        with kpi_cols_player[1]:
                            st.metric(
                                "Minutes totales jouées",
                                int(total_minutes_player),
                            )
                        with kpi_cols_player[2]:
                            label_pos = (
                                f"vs league ({player_position})"
                                if player_position is not None
                                else "vs league (tous postes)"
                            )
                            st.metric(
                                f"% {label_pos}",
                                delta_pos_str,
                            )

                        st.markdown("---")

                        # Line chart match by match (joueur)
                        st.markdown("### Évolution match par match (joueur)")

                        match_group_cols = ["match_key", "matchDate", "opponentTeamName"]
                        for c in match_group_cols:
                            if c not in df_player_drill.columns:
                                df_player_drill[c] = np.nan

                        df_player_matches = (
                            df_player_drill.groupby(match_group_cols, dropna=False)[selected_metric_drill]
                            .mean()
                            .reset_index()
                            .sort_values("matchDate")
                        )

                        if df_player_matches.empty:
                            st.info("Aucune donnée match par match pour ce joueur.")
                        else:
                            fig_player_line = go.Figure()
                            fig_player_line.add_trace(
                                go.Scatter(
                                    x=df_player_matches["matchDate"],
                                    y=df_player_matches[selected_metric_drill],
                                    mode="lines+markers",
                                    name=selected_player,
                                    hovertext=df_player_matches["opponentTeamName"],
                                    hovertemplate=(
                                        "<b>%{x|%Y-%m-%d}</b><br>vs %{hovertext}<br>"
                                        + f"{format_metric_name(selected_metric_drill)}: %{{y:.1f}}<extra></extra>"
                                    ),
                                )
                            )
                            fig_player_line.update_layout(
                                title=f"{selected_player} – {format_metric_name(selected_metric_drill)} match par match",
                                xaxis_title="Date du match",
                                yaxis_title=format_metric_name(selected_metric_drill),
                                margin=dict(l=10, r=10, t=40, b=40),
                            )
                            st.plotly_chart(fig_player_line, use_container_width=True)

                        st.markdown("---")

                        # Barplot comparaison joueur vs autres au même poste
                        st.markdown("### Comparaison vs autres joueurs du même poste dans la league")

                        if position_col is None:
                            st.info("Comparaison par poste impossible (colonne poste manquante).")
                        elif player_position is None:
                            st.info("Poste du joueur non défini, comparaison par poste impossible.")
                        else:
                            df_pos_league = df_league[
                                df_league[position_col].astype(str) == player_position
                            ].copy()

                            if df_pos_league.empty:
                                st.info("Aucun joueur pour ce poste dans la league.")
                            else:
                                group_cols_pos = [player_col, "teamShortName"]
                                df_pos_agg = (
                                    df_pos_league.groupby(group_cols_pos, dropna=False)[selected_metric_drill]
                                    .mean()
                                    .reset_index()
                                    .dropna(subset=[selected_metric_drill])
                                )

                                if df_pos_agg.empty:
                                    st.info("Aucun joueur pour ce poste / cette métrique.")
                                else:
                                    top_n_comp = st.slider(
                                        "Top N joueurs du poste à afficher",
                                        min_value=10,
                                        max_value=50,
                                        value=25,
                                    )

                                    df_pos_agg_sorted = df_pos_agg.sort_values(
                                        selected_metric_drill, ascending=False
                                    )

                                    # Assurer que le joueur sélectionné est visible
                                    mask_sel = (
                                        (df_pos_agg_sorted[player_col].astype(str) == selected_player)
                                        & (df_pos_agg_sorted["teamShortName"].astype(str) == selected_team_drill)
                                    )
                                    df_top = df_pos_agg_sorted.head(top_n_comp)
                                    if not mask_sel.head(top_n_comp).any():
                                        df_sel_row = df_pos_agg_sorted[mask_sel]
                                        if not df_sel_row.empty:
                                            df_top = pd.concat([df_top, df_sel_row])

                                    df_top["is_selected"] = (
                                        (df_top[player_col].astype(str) == selected_player)
                                        & (df_top["teamShortName"].astype(str) == selected_team_drill)
                                    )

                                    fig_comp = px.bar(
                                        df_top,
                                        x=selected_metric_drill,
                                        y=player_col,
                                        orientation="h",
                                        color="is_selected",
                                        color_discrete_map={True: "crimson", False: "lightgray"},
                                        hover_data=["teamShortName"],
                                        labels={
                                            player_col: "Joueur",
                                            selected_metric_drill: format_metric_name(selected_metric_drill),
                                            "teamShortName": "Équipe",
                                            "is_selected": "Joueur sélectionné",
                                        },
                                    )
                                    fig_comp.update_layout(
                                        title=f"Comparaison {format_metric_name(selected_metric_drill)} – poste {player_position}",
                                        xaxis_title=format_metric_name(selected_metric_drill),
                                        yaxis_title="Joueur",
                                        margin=dict(l=10, r=10, t=40, b=40),
                                    )
                                    fig_comp.update_yaxes(autorange="reversed")
                                    st.plotly_chart(fig_comp, use_container_width=True)
