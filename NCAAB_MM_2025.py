import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os, math

# --- Streamlit Setup ---
st.set_page_config(page_title="NCAA BASKETBALL -- MARCH MADNESS 2025", layout="wide", initial_sidebar_state="auto")
hide_menu_style = """<style>#MainMenu {visibility: hidden; } footer {visibility: hidden;}</style>"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 1) Load Data from GitHub CSV
abs_path = r'https://raw.githubusercontent.com/nehat312/march-madness-2025/main'
mm_database_csv = abs_path + '/data/mm_2025_database.csv'

@st.cache_data
def load_data():
    df = pd.read_csv(mm_database_csv, index_col=0)
    df.index.name = "TEAM"
    return df

mm_database_2025 = load_data()

# ----------------------------------------------------------------------------
# 2) Select Relevant Columns (including radar metrics)
core_cols = [
    "KP_Rank", "WIN_25", "LOSS_25", "WIN% ALL GM", "WIN% CLOSE GM", 
    "KP_AdjEM", "KP_SOS_AdjEM", "OFF EFF", "DEF EFF", "OFF REB/GM", "DEF REB/GM",
    "BLKS/GM", "STL/GM", "AST/GM", "TO/GM", "AVG MARGIN", "PTS/GM", "OPP PTS/GM",
    "eFG%", "OPP eFG%", "TS%", "OPP TS%", "AST/TO%", "STOCKS/GM", "STOCKS-TOV/GM"
]
extra_cols_for_treemap = ["CONFERENCE", "TM_KP"]
all_desired_cols = core_cols + extra_cols_for_treemap
actual_cols = [c for c in all_desired_cols if c in mm_database_2025.columns]
df_main = mm_database_2025[actual_cols].copy()

# Ensure team label exists (if "TM_KP" is missing, use index)
if "TM_KP" not in df_main.columns:
    df_main["TM_KP"] = df_main.index

# ----------------------------------------------------------------------------
# 3) Clean Data for Treemap
required_path_cols = ["CONFERENCE", "TM_KP", "KP_AdjEM"]
cols_that_exist = [c for c in required_path_cols if c in df_main.columns]
if len(cols_that_exist) == len(required_path_cols):
    df_main_notnull = df_main.dropna(subset=cols_that_exist, how="any").copy()
else:
    df_main_notnull = df_main.copy()

# ----------------------------------------------------------------------------
# 4) Logo Loading
logo_path = "images/NCAA_logo1.png"
NCAA_logo = Image.open(logo_path) if os.path.exists(logo_path) else None

# ----------------------------------------------------------------------------
# 5) Radar Chart Functions
def get_default_metrics():
    return ['OFF EFF', 'DEF EFF', 'OFF REB/GM', 'DEF REB/GM', 'BLKS/GM', 'STL/GM', 'AST/GM', 'TO/GM']

def compute_tournament_stats(df):
    metrics = get_default_metrics()
    avgs = {m: df[m].mean() for m in metrics if m in df.columns}
    stdevs = {m: df[m].std() for m in metrics if m in df.columns}
    return avgs, stdevs

def compute_performance_text(team_row, t_avgs, t_stdevs):
    metrics = get_default_metrics()
    z_vals = []
    for m in metrics:
        if m in team_row and m in t_avgs and m in t_stdevs:
            std = t_stdevs[m] if t_stdevs[m] > 0 else 1.0
            z = (team_row[m] - t_avgs[m]) / std
            if m in ['DEF EFF', 'TO/GM']:
                z = -z
            z_vals.append(z)
    if not z_vals:
        return "No Data"
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

def get_radar_traces(team_row, t_avgs, t_stdevs, conf_df, show_legend=False):
    metrics = get_default_metrics()
    available_metrics = [m for m in metrics if m in team_row.index and m in t_avgs]
    if not available_metrics:
        return []
    z_scores = []
    for m in available_metrics:
        std = t_stdevs[m] if t_stdevs[m] > 0 else 1.0
        z = (team_row[m] - t_avgs[m]) / std
        if m in ['DEF EFF', 'TO/GM']:
            z = -z
        z_scores.append(z)
    scale_factor = 1.5
    scaled_team = [min(max(5 + z * scale_factor, 0), 10) for z in z_scores]
    scaled_ncaam = [5] * len(available_metrics)
    conf = team_row['CONFERENCE'] if 'CONFERENCE' in team_row else None
    if conf is not None and not conf_df.empty:
        conf_vals = []
        for m in available_metrics:
            if m in conf_df.columns:
                conf_avg = conf_df[m].mean()
                std = t_stdevs[m] if t_stdevs[m] > 0 else 1.0
                z = (conf_avg - t_avgs[m]) / std
                if m in ['DEF EFF', 'TO/GM']:
                    z = -z
                conf_vals.append(z)
            else:
                conf_vals.append(0)
        scaled_conf = [min(max(5 + z * scale_factor, 0), 10) for z in conf_vals]
    else:
        scaled_conf = [5] * len(available_metrics)
    metrics_circ = available_metrics + [available_metrics[0]]
    team_scaled_circ = scaled_team + [scaled_team[0]]
    ncaam_scaled_circ = scaled_ncaam + [scaled_ncaam[0]]
    conf_scaled_circ = scaled_conf + [scaled_conf[0]]
    team_name = team_row.name if hasattr(team_row, 'name') else "Team"
    trace_team = go.Scatterpolar(
        r=team_scaled_circ, theta=metrics_circ, fill='toself',
        fillcolor='rgba(30,144,255,0.3)', name='TEAM',
        line=dict(color='dodgerblue', width=2), showlegend=show_legend,
        hovertemplate="%{theta}: %{r:.1f}<extra>" + f"{team_name}</extra>"
    )
    trace_ncaam = go.Scatterpolar(
        r=ncaam_scaled_circ, theta=metrics_circ, fill='toself',
        fillcolor='rgba(255,99,71,0.2)', name='NCAAM AVG',
        line=dict(color='tomato', width=2, dash='dash'), showlegend=show_legend,
        hoverinfo='skip'
    )
    trace_conf = go.Scatterpolar(
        r=conf_scaled_circ, theta=metrics_circ, fill='toself',
        fillcolor='rgba(50,205,50,0.2)', name='CONFERENCE',
        line=dict(color='limegreen', width=2, dash='dot'), showlegend=show_legend,
        hoverinfo='skip'
    )
    return [trace_team, trace_ncaam, trace_conf]

def create_radar_chart(selected_teams, full_df):
    if not selected_teams:
        return None
    team_mask = full_df['TM_KP'].isin(selected_teams)
    subset = full_df[team_mask].copy().reset_index()
    if subset.empty:
        return None
    t_avgs, t_stdevs = compute_tournament_stats(full_df)
    n_teams = len(subset)
    if n_teams <= 4:
        rows, cols = 1, n_teams
    else:
        rows, cols = 2, min(4, math.ceil(n_teams / 2))
    subplot_titles = []
    for i, row in subset.iterrows():
        team_name = row['TM_KP'] if 'TM_KP' in row else f"Team {i+1}"
        conf = row['CONFERENCE'] if 'CONFERENCE' in row else "N/A"
        subplot_titles.append(f"{i+1}) {team_name} ({conf})")
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'polar'}] * cols for _ in range(rows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.07, vertical_spacing=0.15
    )
    fig.update_layout(
        height=400 if rows == 1 else 800,
        title="Radar Dashboards for Selected Teams",
        template='plotly_dark', font=dict(size=12), showlegend=True
    )
    # Adjust angular axis to reduce label spillover
    fig.update_polars(
        radialaxis=dict(
            tickmode='array', tickvals=[0, 2, 4, 6, 8, 10],
            ticktext=['0', '2', '4', '6', '8', '10'],
            tickfont=dict(size=10),
            showline=False, gridcolor='lightgrey'
        ),
        angularaxis=dict(
            tickfont=dict(size=8),
            tickangle=45,
            showline=False, gridcolor='lightgrey'
        )
    )
    for idx, team_row in subset.iterrows():
        r = idx // cols + 1
        c = idx % cols + 1
        show_legend = (idx == 0)
        conf = team_row['CONFERENCE'] if 'CONFERENCE' in team_row else None
        conf_df = full_df[full_df['CONFERENCE'] == conf] if conf else pd.DataFrame()
        traces = get_radar_traces(team_row, t_avgs, t_stdevs, conf_df, show_legend=show_legend)
        for tr in traces:
            fig.add_trace(tr, row=r, col=c)
        perf_text = compute_performance_text(team_row, t_avgs, t_stdevs)
        polar_idx = (r - 1) * cols + c
        polar_key = "polar" if polar_idx == 1 else f"polar{polar_idx}"
        if polar_key in fig.layout:
            domain_x = fig.layout[polar_key].domain.x
            domain_y = fig.layout[polar_key].domain.y
            x_annot = domain_x[0] + 0.03
            y_annot = domain_y[1] - 0.03
        else:
            x_annot, y_annot = 0.1, 0.9
        fig.add_annotation(
            x=x_annot, y=y_annot, xref="paper", yref="paper",
            text=f"<b>{perf_text}</b>", showarrow=False,
            font=dict(size=12, color="gold")
        )
    return fig

# ----------------------------------------------------------------------------
# 6) Create Enhanced Treemap
def create_treemap(df_notnull):
    if all(c in df_notnull.columns for c in ["CONFERENCE", "TM_KP", "KP_AdjEM"]):
        treemap_data = df_notnull.reset_index()
        treemap_data["KP_AdjEM"] = pd.to_numeric(treemap_data["KP_AdjEM"], errors='coerce')
        if "TM_KP" not in treemap_data.columns:
            treemap_data["TM_KP"] = treemap_data["TEAM"]
        treemap_data['hover_text'] = treemap_data.apply(
            lambda x: (
                f"<b>{x['TM_KP']}</b><br>"
                f"KP Rank: {x['KP_Rank']:.0f}<br>"
                f"Record: {x['WIN_25']:.0f}-{x['LOSS_25']:.0f}<br>"
                f"AdjEM: {x['KP_AdjEM']:.1f}<br>"
                f"OFF EFF: {x.get('OFF EFF', 'N/A')}<br>"
                f"DEF EFF: {x.get('DEF EFF', 'N/A')}"
            ),
            axis=1
        )
        treemap = px.treemap(
            data_frame=treemap_data,
            path=["CONFERENCE", "TM_KP"],
            values="KP_AdjEM",
            color="KP_AdjEM",
            color_continuous_scale=px.colors.diverging.RdYlGn,
            hover_name="TM_KP",
            hover_data={"TEAM": True, "KP_Rank": ':.0f', "KP_AdjEM": ':.1f', "hover_text": False, "TM_KP": False},
            custom_data=["hover_text"],
            template="plotly_dark",
            title="2025 KenPom AdjEM by Conference"
        )
        treemap.update_traces(
            hovertemplate='%{customdata[0]}',
            texttemplate='<b>%{label}</b><br>%{value:.1f}',
            textfont=dict(size=11)
        )
        treemap.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            coloraxis_colorbar=dict(
                title="AdjEM", thicknessmode="pixels", thickness=15,
                lenmode="pixels", len=300, yanchor="top", y=1, ticks="outside"
            )
        )
        return treemap
    return None

# ----------------------------------------------------------------------------
# 7) App Header & Tabs
st.title("NCAA BASKETBALL -- MARCH MADNESS 2025")
st.write("2025 MARCH MADNESS RESEARCH HUB")
col1, col2 = st.columns([6, 1])
with col2:
    if NCAA_logo:
        st.image(NCAA_logo, width=150)
treemap = create_treemap(df_main_notnull)
tab_home, tab_eda, tab_radar, tab_regions, tab_tbd = st.tabs([
    "Home", "EDA & Plots", "Team Radar Charts", "Regional Heatmaps", "TBD"
])

# --- Home Tab ---
with tab_home:
    st.subheader("Conference Treemap Overview")
    if treemap is not None:
        st.plotly_chart(treemap, use_container_width=True)
    else:
        st.warning("Treemap not available due to missing data in required columns.")
    st.markdown("### Key Insights")
    st.write("Use the tabs above to explore teams, stats, and visualizations for March Madness 2025.")
    if "CONFERENCE" in df_main.columns:
        conf_counts = df_main["CONFERENCE"].value_counts().reset_index()
        conf_counts.columns = ["Conference", "# Teams"]
        if "KP_AdjEM" in df_main.columns:
            conf_stats = df_main.groupby("CONFERENCE")["KP_AdjEM"].agg(["mean", "min", "max", "count"]).reset_index()
            conf_stats.columns = ["Conference", "Avg AdjEM", "Min AdjEM", "Max AdjEM", "Count"]
            conf_stats = conf_stats.sort_values("Avg AdjEM", ascending=False)
            st.markdown("### Conference Power Ratings")
            st.dataframe(conf_stats.style.format({
                "Avg AdjEM": "{:.2f}", "Min AdjEM": "{:.2f}", "Max AdjEM": "{:.2f}"
            }), use_container_width=True)

# --- EDA & Plots Tab ---
with tab_eda:
    st.header("Exploratory Data Analysis")
    plot_type = st.selectbox("Select Plot Type", ["Histogram", "Correlation Heatmap", "Conference Comparison", "Team Metrics Comparison"])
    numeric_cols = [c for c in core_cols if c in df_main.columns and pd.api.types.is_numeric_dtype(df_main[c])]
    if plot_type == "Histogram":
        hist_metric = st.selectbox("Select Metric", numeric_cols, index=numeric_cols.index("KP_AdjEM") if "KP_AdjEM" in numeric_cols else 0)
        fig_hist = px.histogram(df_main, x=hist_metric, nbins=25, marginal="box", color_discrete_sequence=["dodgerblue"],
                                template="plotly_dark", title=f"Distribution of {hist_metric} (All Teams)")
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
    elif plot_type == "Correlation Heatmap":
        default_corr_metrics = [m for m in ["KP_AdjEM", "OFF EFF", "DEF EFF", "PTS/GM", "OPP PTS/GM"] if m in numeric_cols]
        selected_corr_metrics = st.multiselect("Select Metrics for Correlation Analysis", options=numeric_cols, default=default_corr_metrics)
        if len(selected_corr_metrics) >= 2:
            df_for_corr = df_main[selected_corr_metrics].dropna()
            corr_mat = df_for_corr.corr().round(2)
            fig_corr = px.imshow(corr_mat, text_auto=True, color_continuous_scale="RdBu_r",
                                 title="Correlation Matrix", template="plotly_dark")
            fig_corr.update_layout(width=800, height=700)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Please select at least 2 metrics for correlation analysis")
    elif plot_type == "Conference Comparison":
        if "CONFERENCE" in df_main.columns:
            conf_metric = st.selectbox("Select Metric for Conference Comparison", numeric_cols, index=numeric_cols.index("KP_AdjEM") if "KP_AdjEM" in numeric_cols else 0)
            conf_group = df_main.groupby("CONFERENCE")[conf_metric].mean().dropna().sort_values(ascending=False)
            fig_conf = px.bar(conf_group, y=conf_group.index, x=conf_group.values, orientation='h',
                              title=f"Average {conf_metric} by Conference", labels={"y": "Conference", "x": conf_metric},
                              color=conf_group.values, color_continuous_scale="Viridis", template="plotly_dark")
            for conf in conf_group.index:
                teams = df_main[df_main["CONFERENCE"] == conf]
                fig_conf.add_trace(go.Scatter(x=teams[conf_metric], y=[conf] * len(teams),
                                              mode="markers", marker=dict(color="white", size=6, opacity=0.7),
                                              name=f"{conf} Teams"))
            fig_conf.update_layout(showlegend=False)
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.warning("Conference data not available")
    elif plot_type == "Team Metrics Comparison":
        x_metric = st.selectbox("Select X-Axis Metric", numeric_cols, index=numeric_cols.index("OFF EFF") if "OFF EFF" in numeric_cols else 0)
        y_metric = st.selectbox("Select Y-Axis Metric", numeric_cols, index=numeric_cols.index("DEF EFF") if "DEF EFF" in numeric_cols else 0)
        if "CONFERENCE" in df_main.columns:
            fig_scatter = px.scatter(df_main.reset_index(), x=x_metric, y=y_metric, color="CONFERENCE",
                                     hover_name="TEAM", size="KP_AdjEM" if "KP_AdjEM" in df_main.columns else None,
                                     size_max=15, opacity=0.8, template="plotly_dark", title=f"{y_metric} vs {x_metric}", height=700)
        else:
            fig_scatter = px.scatter(df_main.reset_index(), x=x_metric, y=y_metric, hover_name="TEAM", opacity=0.8,
                                     template="plotly_dark", title=f"{y_metric} vs {x_metric}")
        if (("OFF" in x_metric and "DEF" in y_metric) or ("DEF" in x_metric and "OFF" in y_metric)):
            x_avg, y_avg = df_main[x_metric].mean(), df_main[y_metric].mean()
            fig_scatter.add_hline(y=y_avg, line_dash="dash", line_color="white", opacity=0.4)
            fig_scatter.add_vline(x=x_avg, line_dash="dash", line_color="white", opacity=0.4)
            quadrants = [
                {"x": x_avg * 0.9, "y": y_avg * 0.9, "text": "Poor Both"},
                {"x": x_avg * 1.1, "y": y_avg * 0.9, "text": "Good X Only"},
                {"x": x_avg * 0.9, "y": y_avg * 1.1, "text": "Good Y Only"},
                {"x": x_avg * 1.1, "y": y_avg * 1.1, "text": "Elite Both"}
            ]
            for q in quadrants:
                fig_scatter.add_annotation(x=q["x"], y=q["y"], text=q["text"], showarrow=False,
                                           font=dict(color="white", size=10), opacity=0.7)
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- Radar Charts Tab ---
with tab_radar:
    st.header("Team Radar Performance Charts")
    radar_metrics = get_default_metrics()
    available_radar_metrics = [m for m in radar_metrics if m in df_main.columns]
    if len(available_radar_metrics) < 3:
        st.warning(f"Not enough radar metrics available. Need at least 3 of these: {', '.join(radar_metrics)}")
    else:
        if "TM_KP" in df_main.columns:
            all_teams = df_main["TM_KP"].dropna().unique().tolist()
            default_teams = []
            if "KP_AdjEM" in df_main.columns:
                top_teams = df_main.sort_values("KP_AdjEM", ascending=False).head(6)
                if "TM_KP" in top_teams.columns:
                    default_teams = top_teams["TM_KP"].tolist()
            if not default_teams and all_teams:
                default_teams = all_teams[:min(6, len(all_teams))]
            selected_teams = st.multiselect("Select Teams to Compare (4-8 recommended)",
                                            options=sorted(all_teams), default=default_teams[:min(6, len(default_teams))])
            if selected_teams:
                radar_fig = create_radar_chart(selected_teams, df_main)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
                else:
                    st.warning("Could not create radar chart with selected teams.")
            else:
                st.info("Please select at least one team to display radar charts.")
            with st.expander("About Radar Charts"):
                st.markdown("""
These radar charts show team performance across 8 key metrics compared to:
- **NCAAM Average** (red dashed line)
- **Conference Average** (green dotted line)

Each metric is scaled so that 5 represents the NCAA average. Values above 5 are better, and those below 5 are worse.  
The overall performance rating is calculated from the team's average z-score across all metrics.
                """)
        else:
            st.warning("Team names (TM_KP column) not available in dataset.")

# --- Regional Heatmaps Tab ---
with tab_regions:
    st.header("Regional Analysis & Heatmaps")
    st.write("Regional analysis for East, West, South, and Midwest.")
    df_heat = df_main.copy()
    df_heat.loc["TOURNEY AVG"] = df_heat.mean(numeric_only=True)
    df_heat_T = df_heat.T  # Transpose so that index = metric names, columns = teams
    # Define region team lists (update these with actual 2025 team names)
    east_teams_2025 = ["Alabama", "Houston", "Duke", "Tennessee", "TOURNEY AVG"]
    west_teams_2025 = ["Kansas", "UCLA", "Gonzaga", "Connecticut", "TOURNEY AVG"]
    south_teams_2025 = ["Arizona", "Baylor", "Virginia", "Florida", "TOURNEY AVG"]
    midwest_teams_2025 = ["Texas", "Xavier", "Indiana", "Michigan St", "TOURNEY AVG"]
    regions = {
        "East Region": east_teams_2025,
        "West Region": west_teams_2025,
        "South Region": south_teams_2025,
        "Midwest Region": midwest_teams_2025
    }
    # Enhanced safe formatting function for region heatmap cells
    def safe_format(x):
        try:
            val = float(x)
            # If the value is a fraction (0 to 1), format as percentage; otherwise, two decimals.
            if 0 <= val < 1:
                return f"{val*100:.1f}%"
            else:
                return f"{val:.2f}"
        except (ValueError, TypeError):
            return x
    for region_name, team_list in regions.items():
        teams_found = [tm for tm in team_list if tm in df_heat_T.columns]
        if teams_found:
            region_df = df_heat_T[teams_found].copy()
            st.subheader(region_name)
            region_styler = region_df.style
            for row_label, cmap in {
                "KP_Rank": "Spectral_r",
                "WIN_25": "YlGn",
                "LOSS_25": "YlOrRd_r",
                "KP_AdjEM": "RdYlGn",
                "KP_SOS_AdjEM": "RdBu",
                "OFF EFF": "Blues",
                "DEF EFF": "Reds_r",
                "AVG MARGIN": "RdYlGn",
                "TS%": "YlGn",
                "OPP TS%": "YlOrRd_r",
                "AST/TO%": "Greens",
                "STOCKS/GM": "Purples"
            }.items():
                region_styler = region_styler.background_gradient(cmap=cmap, subset=pd.IndexSlice[[row_label], :])
            region_styler = region_styler.format(safe_format)
            st.dataframe(region_styler, use_container_width=True)
        else:
            st.info(f"No data available for {region_name}.")

# --- TBD Tab ---
with tab_tbd:
    st.header("To Be Determined")
    st.info("More visualizations and analysis coming soon.")

# ----------------------------------------------------------------------------
# 9) GitHub Link & App Footer
st.markdown("---")
st.markdown("App code available on [GitHub](https://github.com/nehat312/march-madness-2025)")

st.stop()
