#########################################
# NCAAB MARCH MADNESS 2025 - MAIN APP  #
#########################################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import altair as alt
from PIL import Image
import os
import math
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# 1) PATH TO DATA & RESOURCES
#    (Adjust if your CSV path or location changes)
# ----------------------------------------------------------------------------
abs_path = r'https://raw.githubusercontent.com/nehat312/march-madness-2025/main'
mm_database_csv = abs_path + '/data/mm_2025_database.csv'

# ----------------------------------------------------------------------------
# 2) DATA LOADING AND PREPROCESSING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load and preprocess the March Madness dataset"""
    mm_database_2025 = pd.read_csv(mm_database_csv, index_col=0)
    mm_database_2025.index.name = "TEAM"  # rename for clarity
    
    # Fill NA values for numeric columns to enable better visualizations
    numeric_cols = mm_database_2025.select_dtypes(include=['float64']).columns
    mm_database_2025[numeric_cols] = mm_database_2025[numeric_cols].fillna(mm_database_2025[numeric_cols].mean())
    
    return mm_database_2025

mm_database_2025 = load_data()

# ----------------------------------------------------------------------------
# 3) SELECT RELEVANT COLUMNS FOR DIFFERENT ANALYSES
# ----------------------------------------------------------------------------
core_cols = [
    "KP_Rank", "WIN_25", "LOSS_25", "WIN% ALL GM", "WIN% CLOSE GM",
    "KP_AdjEM", "KP_AdjO", "KP_AdjD", "KP_SOS_AdjEM", "NET_24",
    "OFF EFF", "DEF EFF", "AVG MARGIN", "PTS/GM", "OPP PTS/GM",
    "eFG%", "OPP eFG%", "TS%", "OPP TS%", 
    "AST/TO%", "STOCKS/GM", "STOCKS-TOV/GM",
    "3PT%", "2PT%", "FT%", "TTL REB/GM"
]

# Path columns for treemap
path_cols = ["CONFERENCE", "TM_KP"]

# Additional metrics for radar charts
radar_metrics = [
    'OFF EFF', 'DEF EFF', 'OFF REB/GM', 'DEF REB/GM',
    'BLKS/GM', 'STL/GM', 'AST/GM', 'TO/GM'
]

# Create a main dataframe with the required columns
all_required_cols = list(set(core_cols + path_cols + radar_metrics))
actual_cols = [c for c in all_required_cols if c in mm_database_2025.columns]
df_main = mm_database_2025[actual_cols].copy()

# Remove rows with missing values in required path columns for treemap
treemap_required = [c for c in path_cols + ["KP_AdjEM"] if c in df_main.columns]
df_treemap = df_main.dropna(subset=treemap_required).copy()

# ----------------------------------------------------------------------------
# 4) COMPUTED STATISTICS & METRICS
# ----------------------------------------------------------------------------
# Compute tournament averages & standard deviations for radar charts
tournament_avgs = {
    m: df_main[m].mean() for m in radar_metrics if m in df_main.columns
}
tournament_stdevs = {
    m: df_main[m].std() for m in radar_metrics if m in df_main.columns
}

# Create a Top 25 teams dataframe
if "KP_Rank" in df_main.columns:
    df_top25 = df_main.sort_values("KP_Rank").head(25).copy()
else:
    # Fallback if KP_Rank is missing
    ranking_col = next((c for c in df_main.columns if "Rank" in c), None)
    if ranking_col:
        df_top25 = df_main.sort_values(ranking_col).head(25).copy()
    else:
        # If no ranking available, sort by KP_AdjEM if available
        if "KP_AdjEM" in df_main.columns:
            df_top25 = df_main.sort_values("KP_AdjEM", ascending=False).head(25).copy()
        else:
            df_top25 = df_main.head(25).copy()

# Determine popular conferences (with at least 3 teams)
if "CONFERENCE" in df_main.columns:
    conf_counts = df_main["CONFERENCE"].value_counts()
    popular_conferences = conf_counts[conf_counts >= 3].index.tolist()
else:
    popular_conferences = []

# ----------------------------------------------------------------------------
# 5) HELPER FUNCTIONS FOR VISUALIZATIONS 
# ----------------------------------------------------------------------------
def create_treemap(df):
    """Create a conference-based treemap colored by KP_AdjEM."""
    if all(c in df.columns for c in ["CONFERENCE", "TM_KP", "KP_AdjEM"]):
        treemap_data = df.reset_index()  # 'TEAM' becomes a column
        treemap_data = treemap_data.sort_values("KP_AdjEM")

        treemap = px.treemap(
            data_frame=treemap_data,
            path=["CONFERENCE", "TM_KP"],
            values="KP_AdjEM",
            color="KP_AdjEM",
            color_continuous_scale=px.colors.diverging.RdYlGn,
            hover_data=["TEAM", "KP_Rank", "WIN_25", "LOSS_25"],
            template="plotly_dark",
            title="2025 KenPom AdjEM by Conference"
        )
        treemap.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            coloraxis_colorbar=dict(
                title="KP_AdjEM",
                thicknessmode="pixels", thickness=15,
                lenmode="pixels", len=300
            )
        )
        return treemap
    return None

def create_conference_comparison(df, popular_conferences):
    """Create a multi-metric conference comparison chart."""
    if "CONFERENCE" not in df.columns or len(popular_conferences) == 0:
        return None
    
    # Select metrics for comparison
    comparison_metrics = ["KP_AdjEM", "OFF EFF", "DEF EFF", "AVG MARGIN"]
    available_metrics = [m for m in comparison_metrics if m in df.columns]
    
    if len(available_metrics) == 0:
        return None
    
    # Aggregate by conference
    conf_stats = df.groupby("CONFERENCE")[available_metrics].mean().reset_index()
    conf_stats = conf_stats[conf_stats["CONFERENCE"].isin(popular_conferences)]
    
    # Create subplots: one bar per metric
    fig = make_subplots(
        rows=1, 
        cols=len(available_metrics),
        subplot_titles=[m for m in available_metrics],
        shared_yaxes=True
    )
    
    colors = px.colors.qualitative.Plotly
    
    for i, metric in enumerate(available_metrics):
        conf_stats_sorted = conf_stats.sort_values(metric, ascending=False)
        fig.add_trace(
            go.Bar(
                x=conf_stats_sorted["CONFERENCE"],
                y=conf_stats_sorted[metric],
                marker_color=colors[i % len(colors)],
                name=metric
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        height=400,
        title="Conference Comparison by Key Metrics",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=50, r=20, t=80, b=80)
    )
    
    # Rotate x-axis if many conferences
    if len(popular_conferences) > 5:
        for j in range(len(available_metrics)):
            fig.update_xaxes(tickangle=45, row=1, col=j+1)
    
    return fig

def create_offensive_defensive_scatter(df):
    """Create an offensive vs defensive efficiency scatter plot with quadrant lines."""
    if "OFF EFF" not in df.columns or "DEF EFF" not in df.columns:
        return None
    
    plot_data = df.reset_index()
    size_col = "AVG MARGIN" if "AVG MARGIN" in df.columns else None
    color_col = "KP_AdjEM" if "KP_AdjEM" in df.columns else None
    
    fig = px.scatter(
        plot_data,
        x="OFF EFF",
        y="DEF EFF",
        size=size_col,
        color=color_col,
        hover_name="TEAM",
        hover_data=["CONFERENCE", "KP_Rank", "WIN_25", "LOSS_25"],
        color_continuous_scale="RdYlGn",
        template="plotly_white",
        title="Offensive vs Defensive Efficiency",
        labels={
            "OFF EFF": "Offensive Efficiency",
            "DEF EFF": "Defensive Efficiency (lower is better)",
            color_col: "KenPom AdjEM" if color_col == "KP_AdjEM" else color_col
        }
    )
    
    x_avg = df["OFF EFF"].mean()
    y_avg = df["DEF EFF"].mean()
    
    fig.add_shape(
        type="line", line=dict(dash="dash", color="gray"),
        x0=x_avg, y0=fig.data[0].y.min(), x1=x_avg, y1=fig.data[0].y.max()
    )
    fig.add_shape(
        type="line", line=dict(dash="dash", color="gray"),
        x0=fig.data[0].x.min(), y0=y_avg, x1=fig.data[0].x.max(), y1=y_avg
    )
    
    # Quadrant annotations
    fig.add_annotation(x=df["OFF EFF"].max()-5, y=df["DEF EFF"].min()+5,
                       text="Elite Offense, Elite Defense",
                       showarrow=False, font=dict(color="green"))
    fig.add_annotation(x=df["OFF EFF"].min()+5, y=df["DEF EFF"].min()+5,
                       text="Poor Offense, Elite Defense",
                       showarrow=False)
    fig.add_annotation(x=df["OFF EFF"].max()-5, y=df["DEF EFF"].max()-5,
                       text="Elite Offense, Poor Defense",
                       showarrow=False)
    fig.add_annotation(x=df["OFF EFF"].min()+5, y=df["DEF EFF"].max()-5,
                       text="Poor Offense, Poor Defense",
                       showarrow=False, font=dict(color="red"))
    
    fig.update_layout(height=600)
    return fig

def compute_performance_text(team_row):
    """Return a simple performance category: 'Elite', 'Above Average', etc. based on avg z-score."""
    available_metrics = [m for m in radar_metrics if m in team_row.index]
    if not available_metrics:
        return "N/A"
    
    z_vals = []
    for m in available_metrics:
        std = tournament_stdevs.get(m, 1.0)
        if std <= 0:
            std = 1.0
        z = (team_row[m] - tournament_avgs.get(m, 0)) / std
        # invert where lower is better
        if m in ['DEF EFF', 'TO/GM']:
            z = -z
        z_vals.append(z)
    
    avg_z = sum(z_vals) / len(z_vals)
    if avg_z > 1.5:
        return "Elite"
    elif avg_z > 0.5:
        return "Above Average"
    elif avg_z > -0.5:
        return "Average"
    elif avg_z > -1.5:
        return "Below Average"
    else:
        return "Poor"

def get_radar_traces(team_row, show_legend=False):
    """Generate the 3-layer radar chart traces (TEAM, NCAAM AVG, CONF) for a single team."""
    available_metrics = [m for m in radar_metrics if m in team_row.index]
    if not available_metrics:
        return []
    
    z_scores = []
    for m in available_metrics:
        std = tournament_stdevs.get(m, 1.0)
        if std <= 0:
            std = 1.0
        z = (team_row[m] - tournament_avgs.get(m, 0)) / std
        if m in ['DEF EFF', 'TO/GM']:
            z = -z
        z_scores.append(z)
    
    scale_factor = 1.5
    scaled_team = [min(max(5 + z*scale_factor, 0), 10) for z in z_scores]

    # NCAAM avg => 5
    scaled_ncaam = [5]*len(available_metrics)

    # Conference avg
    if "CONFERENCE" in team_row.index and team_row["CONFERENCE"] in df_main["CONFERENCE"].values:
        conf = team_row["CONFERENCE"]
        conf_df = df_main[df_main["CONFERENCE"] == conf]
        conf_vals = []
        for m in available_metrics:
            conf_avg = conf_df[m].mean()
            std = tournament_stdevs.get(m, 1.0)
            if std <= 0:
                std = 1.0
            z = (conf_avg - tournament_avgs.get(m, 0)) / std
            if m in ['DEF EFF', 'TO/GM']:
                z = -z
            conf_vals.append(z)
        scaled_conf = [min(max(5 + z*scale_factor, 0), 10) for z in conf_vals]
    else:
        scaled_conf = scaled_ncaam.copy()

    # Close the circle
    metrics_circ = available_metrics + [available_metrics[0]]
    team_scaled_circ = scaled_team + [scaled_team[0]]
    ncaam_scaled_circ = scaled_ncaam + [scaled_ncaam[0]]
    conf_scaled_circ = scaled_conf + [scaled_conf[0]]

    # Build the 3 polar traces
    trace_team = go.Scatterpolar(
        r=team_scaled_circ,
        theta=metrics_circ,
        fill='toself',
        fillcolor='rgba(30,144,255,0.3)',
        name='TEAM',
        line=dict(color='dodgerblue', width=2),
        showlegend=show_legend,
        hovertemplate="%{theta}: %{r:.1f}<extra>" + f"{team_row.name}</extra>"
    )
    trace_ncaam = go.Scatterpolar(
        r=ncaam_scaled_circ,
        theta=metrics_circ,
        fill='toself',
        fillcolor='rgba(255,99,71,0.2)',
        name='NCAAM AVG',
        line=dict(color='tomato', width=2, dash='dash'),
        showlegend=show_legend,
        hoverinfo='skip'
    )
    trace_conf = go.Scatterpolar(
        r=conf_scaled_circ,
        theta=metrics_circ,
        fill='toself',
        fillcolor='rgba(50,205,50,0.2)',
        name='CONFERENCE',
        line=dict(color='limegreen', width=2, dash='dot'),
        showlegend=show_legend,
        hoverinfo='skip'
    )
    
    return [trace_team, trace_ncaam, trace_conf]

def create_team_radars(teams_df, title="Team Radar Analysis"):
    """Given a subset of teams, create multi-subplot radar charts."""
    if teams_df.empty:
        return None
    
    n_teams = len(teams_df)
    if n_teams <= 3:
        rows, cols = 1, n_teams
    elif n_teams <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 2, 4  # cap at 8
        teams_df = teams_df.head(8)
        n_teams = 8
    
    # Build subplot titles
    subplot_titles = []
    for idx, (team_name, row) in enumerate(teams_df.iterrows()):
        conf = row.get("CONFERENCE", "")
        rank = int(row.get("KP_Rank", idx+1)) if "KP_Rank" in row else (idx+1)
        subplot_titles.append(f"#{rank} {team_name} ({conf})")
    
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'polar'}]*cols for _ in range(rows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.07,
        vertical_spacing=0.15
    )
    
    fig.update_layout(
        height=500 if rows == 1 else 900,
        title=title,
        template='plotly_dark',
        font=dict(size=12),
        showlegend=True
    )
    # Custom radial ticks
    fig.update_polars(
        radialaxis=dict(
            tickmode='array',
            tickvals=[0,2,4,6,8,10],
            ticktext=['0','2','4','6','8','10'],
            tickfont=dict(size=10),
            showline=False,
            gridcolor='lightgrey'
        ),
        angularaxis=dict(
            tickfont=dict(size=10),
            showline=False,
            gridcolor='lightgrey'
        )
    )
    
    for i, (team_name, team_row) in enumerate(teams_df.iterrows()):
        r = i // cols + 1
        c = i % cols + 1
        show_legend = (i == 0)
        
        # Add radar traces
        traces = get_radar_traces(team_row, show_legend=show_legend)
        for tr in traces:
            fig.add_trace(tr, row=r, col=c)
        
        # Add performance annotation
        perf_text = compute_performance_text(team_row)
        polar_idx = (r-1)*cols + c
        polar_key = "polar" if polar_idx == 1 else f"polar{polar_idx}"
        if polar_key in fig.layout:
            domain_x = fig.layout[polar_key].domain.x
            domain_y = fig.layout[polar_key].domain.y
            x_annot = domain_x[0] + 0.03
            y_annot = domain_y[1] - 0.03
        else:
            x_annot = 0.1
            y_annot = 0.9
        
        fig.add_annotation(
            x=x_annot,
            y=y_annot,
            xref="paper",
            yref="paper",
            text=f"<b>{perf_text}</b>",
            showarrow=False,
            font=dict(size=12, color="gold")
        )
    
    return fig

def create_shooting_efficiency_chart(df):
    """Example: A 2-row plot: top eFG% scatter vs TS%, then a bar chart with 3PT%, 2PT%, FT%."""
    if not all(c in df.columns for c in ["eFG%", "TS%", "3PT%", "2PT%", "FT%"]):
        return None
    
    plot_data = df.reset_index()
    plot_data = plot_data.sort_values("eFG%", ascending=False).head(20)
    
    # We'll show the top 20 teams by eFG%
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Shooting Efficiency Comparison (Top 20 eFG%)",
                        "Shooting Breakdown by Type"],
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Scatter plot of eFG% vs TS%
    fig.add_trace(
        go.Scatter(
            x=plot_data["eFG%"],
            y=plot_data["TS%"],
            mode='markers+text',
            text=plot_data["TEAM"],
            textposition="top center",
            marker=dict(
                size=10,
                color=plot_data["3PT%"],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="3PT%")
            ),
            name='Teams'
        ),
        row=1, col=1
    )
    
    # Bar chart for top 10 teams by eFG%
    for i, row in plot_data.head(10).iterrows():
        # X = shot types, Y = each shot's percentage
        team_name = row["TEAM"]
        fg3 = row["3PT%"]
        fg2 = row["2PT%"]
        ft = row["FT%"]
        
        fig.add_trace(
            go.Bar(
                x=["3PT%", "2PT%", "FT%"],
                y=[fg3, fg2, ft],
                name=team_name,
                hovertemplate=f"{team_name}<br>3PT={{x}}%<br>Value={{y}}"
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=800,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(title="Effective FG% (horizontal axis)"),
        yaxis=dict(title="True Shooting %"),
        xaxis2=dict(title="Shot Type"),
        yaxis2=dict(title="Percentage", range=[0, 100])
    )
    return fig

def create_correlation_heatmap(df):
    """Create a correlation heatmap of selected numeric columns for interpretability."""
    # Pick some columns that are typically interesting 
    num_cols = [
        "KP_AdjEM", "OFF EFF", "DEF EFF", "AVG MARGIN", "PTS/GM", "OPP PTS/GM",
        "AST/TO%", "eFG%", "TS%", "3PT%", "2PT%", "FT%", "TTL REB/GM"
    ]
    # Filter for columns that exist
    num_cols = [c for c in num_cols if c in df.columns]
    if len(num_cols) < 2:
        return None
    
    corr_df = df[num_cols].corr()
    
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap (Selected Metrics)",
    )
    fig.update_layout(
        template="plotly_white",
        width=700, height=700
    )
    return fig

def create_stat_distribution_chart(df, stat="KP_AdjEM"):
    """Create a distribution (histogram + boxplot) for a chosen numeric stat to see overall field distribution."""
    if stat not in df.columns:
        return None
    
    # Drop NA
    series = df[stat].dropna()
    if series.empty:
        return None
    
    # Plotly histogram
    hist_fig = px.histogram(
        series,
        nbins=30,
        template="plotly_white",
        title=f"Distribution of {stat}",
        labels={stat: stat}
    )
    hist_fig.update_layout(height=400)
    
    # Plotly boxplot
    box_fig = px.box(
        series,
        template="plotly_white",
        title=f"Boxplot of {stat}",
        labels={stat: stat}
    )
    box_fig.update_layout(height=300)
    
    return hist_fig, box_fig

# ----------------------------------------------------------------------------
# 6) STREAMLIT APP LAYOUT
# ----------------------------------------------------------------------------
st.title("NCAA March Madness 2025 - Pre-Tournament Analysis")

# Create a sidebar for navigation
page = st.sidebar.radio(
    "Select Page",
    (
        "Home",
        "Treemap (Conferences)",
        "Conference Comparison",
        "Offense vs. Defense",
        "Team Radar (Multi-Select)",
        "Distribution & Boxplots",
        "Correlation Heatmap",
        "Shooting Efficiency",
    )
)

if page == "Home":
    st.header("Welcome to the 2025 NCAA Men's Basketball Pre-Tourney Dashboard")
    st.markdown("""
        This application provides an early look at the Division I field before the bracket is set. 
        Explore team performance, conference strength, and more. Data courtesy of KenPom and related 2025 metrics.
    """)
    
    st.subheader("Top 25 Snapshot")
    st.write(df_top25.head(25))
    
    st.markdown("Use the sidebar to explore more visual analyses.")
    
elif page == "Treemap (Conferences)":
    st.header("Treemap of Teams by Conference (Colored by KP_AdjEM)")
    treemap_fig = create_treemap(df_treemap)
    if treemap_fig:
        st.plotly_chart(treemap_fig, use_container_width=True)
    else:
        st.write("Treemap not available. Required columns missing.")
    
elif page == "Conference Comparison":
    st.header("Multi-Metric Conference Comparison")
    conf_fig = create_conference_comparison(df_main, popular_conferences)
    if conf_fig:
        st.plotly_chart(conf_fig, use_container_width=True)
    else:
        st.write("No valid conference data to display.")
    
elif page == "Offense vs. Defense":
    st.header("Offensive vs Defensive Efficiency")
    od_fig = create_offensive_defensive_scatter(df_main)
    if od_fig:
        st.plotly_chart(od_fig, use_container_width=True)
    else:
        st.write("Unable to create Offense-Defense scatter. Required columns missing.")

elif page == "Team Radar (Multi-Select)":
    st.header("Interactive Team Radar Charts")
    team_list = list(df_main.index.unique())
    selected_teams = st.multiselect(
        "Select up to 8 teams to compare:",
        options=team_list,
        default=team_list[:4]  # default picks just for demonstration
    )
    if selected_teams:
        subset = df_main.loc[selected_teams].copy()
        # Create a radar figure
        radar_fig = create_team_radars(subset, title="Radar Dashboards for Selected Teams")
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.write("No radar chart available for selected teams.")
    else:
        st.write("Please select at least one team.")

elif page == "Distribution & Boxplots":
    st.header("Overall Distribution of a Metric")
    numeric_cols = df_main.select_dtypes(include=[np.number]).columns.tolist()
    selected_metric = st.selectbox("Select a metric to visualize:", options=numeric_cols, index=0)
    
    dist_charts = create_stat_distribution_chart(df_main, stat=selected_metric)
    if dist_charts:
        hist_fig, box_fig = dist_charts
        st.plotly_chart(hist_fig, use_container_width=True)
        st.plotly_chart(box_fig, use_container_width=True)
    else:
        st.write("Unable to generate distribution charts.")

elif page == "Correlation Heatmap":
    st.header("Correlation Heatmap (Key Metrics)")
    corr_fig = create_correlation_heatmap(df_main)
    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.write("Correlation heatmap unavailable. Check numeric columns.")

elif page == "Shooting Efficiency":
    st.header("Shooting Efficiency Analysis")
    se_fig = create_shooting_efficiency_chart(df_main)
    if se_fig:
        st.plotly_chart(se_fig, use_container_width=True)
    else:
        st.write("Unable to generate shooting efficiency chart.")


# ----------------------------------------------------------------------------
# 8D) FUTURE TABS
# ----------------------------------------------------------------------------
# with tab_tbd1:
#     st.subheader("TBD Tab 1")
#     st.write("Placeholder for future bracket or advanced analytics.")
#     if NCAA_logo:
#         st.image(NCAA_logo, width=120)

# with tab_tbd2:
#     st.subheader("TBD Tab 2")
#     st.write("Placeholder for additional expansions or data merges.")
#     if NCAA_logo:
#         st.image(NCAA_logo, width=120)

# ----------------------------------------------------------------------------
# 9) FOOTER / STOP
# ----------------------------------------------------------------------------
st.markdown("---")
github_link = "[GitHub: 2025 Repo](https://github.com/nehat312/march-madness-2025)"
kenpom_link = "[KENPOM](https://kenpom.com/)"
st.write(github_link + " | " + kenpom_link)

st.stop()
