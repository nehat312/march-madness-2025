import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os, math

# --- Streamlit Setup ---
st.set_page_config(page_title="NCAA BASKETBALL -- MARCH MADNESS 2025",
                   layout="wide", initial_sidebar_state="auto")
hide_menu_style = """
 <style>
 #MainMenu {visibility: hidden; }
 footer {visibility: hidden;}
 </style>
"""
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
# Add columns needed for Treemap and new seed-based hover text
extra_cols_for_treemap = ["CONFERENCE", "TM_KP", "SEED_25"]
all_desired_cols = core_cols + extra_cols_for_treemap
actual_cols = [c for c in all_desired_cols if c in mm_database_2025.columns]
df_main = mm_database_2025[actual_cols].copy()

# Ensure team label (if "TM_KP" is missing, use index)
if "TM_KP" not in df_main.columns:
    df_main["TM_KP"] = df_main.index

# Create new column with absolute value of KP_AdjEM to avoid negative sizing issues
if "KP_AdjEM" in df_main.columns:
    df_main["KP_AdjEM_Pos"] = df_main["KP_AdjEM"].abs()

# ----------------------------------------------------------------------------
# 3) Clean Data for Treemap
required_path_cols = ["CONFERENCE", "TM_KP", "KP_AdjEM"]  # must have these for the treemap
cols_that_exist = [c for c in required_path_cols if c in df_main.columns]
if len(cols_that_exist) == len(required_path_cols):
    df_main_notnull = df_main.dropna(subset=cols_that_exist, how="any").copy()
else:
    df_main_notnull = df_main.copy()

# ----------------------------------------------------------------------------
# 4) Logo Loading / Syntax Configuration
logo_path = "images/NCAA_logo1.png"
FinalFour25_logo_path = "images/ncaab_mens_finalfour2025_logo.png"
Conferences25_logo_path = "images/ncaab_conferences_2025.png"

NCAA_logo = Image.open(logo_path) if os.path.exists(logo_path) else None
FinalFour25_logo = Image.open(FinalFour25_logo_path) if os.path.exists(FinalFour25_logo_path) else None
Conferences25_logo = Image.open(Conferences25_logo_path) if os.path.exists(Conferences25_logo_path) else None

### GLOBAL VISUALIZATION SETTINGS ###
viz_margin_dict = dict(l=20, r=20, t=50, b=20)
viz_bg_color = '#0360CE'
viz_font_dict = dict(size=12, color='#FFFFFF')
RdYlGn = px.colors.diverging.RdYlGn

# ----------------------------------------------------------------------------
# ENHANCED TABLE STYLING (ported from 2023)
header = {
    'selector': 'th',
    'props': [
        ('background-color', '#0360CE'),
        ('color', 'white'),
        ('text-align', 'center'),
        ('vertical-align', 'middle'),
        ('font-weight', 'bold'),
        ('border-bottom', '2px solid #000000')
    ]
}
header_level0 = {
    'selector': 'th.col_heading.level0',
    'props': [('font-size', '12px')]
}
index_style = {
    'selector': 'th.row_heading',
    'props': [
        ('background-color', '#000000'),
        ('color', 'white'),
        ('text-align', 'center'),
        ('vertical-align', 'middle'),
        ('font-weight', 'bold'),
        ('font-size', '12px')
    ]
}
numbers = {
    'selector': 'td.data',
    'props': [
        ('text-align', 'center'),
        ('vertical-align', 'center'),
        ('font-weight', 'bold')
    ]
}
borders_right = {
    'selector': '.row_heading.level1',
    'props': [('border-right', '1px solid #FFFFFF')]
}
top_row = {
    'selector': 'td.data.row0',
    'props': [
        ('border-bottom', '2px dashed #000000'),
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('font-size', '12px')
    ]
}
table_row0 = {
    'selector': '.row0',
    'props': [
        ('border-bottom', '2px dashed #000000'),
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('font-size', '12px')
    ]
}
table_row1 = {
    'selector': '.row1',
    'props': [
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('font-size', '12px')
    ]
}
table_row2 = {
    'selector': '.row2',
    'props': [
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('font-size', '12px')
    ]
}
table_row3 = {
    'selector': '.row3',
    'props': [
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('font-size', '12px')
    ]
}
table_row4 = {
    'selector': '.row4',
    'props': [
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('font-size', '12px')
    ]
}
table_row5 = {
    'selector': '.row5',
    'props': [
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('font-size', '12px')
    ]
}
table_row6 = {
    'selector': '.row6',
    'props': [
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('font-size', '12px')
    ]
}
table_row7 = {
    'selector': '.row7',
    'props': [
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('font-size', '12px')
    ]
}
table_row8 = {
    'selector': '.row8',
    'props': [
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('font-size', '12px')
    ]
}
table_row9 = {
    'selector': '.row9',
    'props': [
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('font-size', '12px')
    ]
}
table_col0 = {
    'selector': '.row0',
    'props': [
        ('border-left', '3px solid #000000'),
        ('min-width', '75px'),
        ('max-width', '75px'),
        ('column-width', '75px')
    ]
}
detailed_table_styles = [
    header,
    header_level0,
    index_style,
    numbers,
    borders_right,
    top_row,
    table_row0,
    table_row1,
    table_row2,
    table_row3,
    table_row4,
    table_row5,
    table_row6,
    table_row7,
    table_row8,
    table_row9,
    table_col0
]

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
    # Incorporate team seed if it exists
    seed_info = ""
    if 'SEED_25' in team_row and pd.notna(team_row['SEED_25']):
        seed_info = f"(Seed {int(team_row['SEED_25'])})"
    team_name = f"{team_row.name} {seed_info}".strip()
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
    # Use the comprehensive radar tab logic from the original version:
    radar_metrics = get_default_metrics()
    available_radar_metrics = [m for m in radar_metrics if m in full_df.columns]
    if len(available_radar_metrics) < 3:
        st.warning(f"Not enough radar metrics available. Need at least 4: {', '.join(radar_metrics)}")
        return None
    if "TM_KP" in full_df.columns:
        all_teams = full_df["TM_KP"].dropna().unique().tolist()
        default_teams = ['Duke', 'Kansas', 'Auburn', 'Houston']
        if "KP_AdjEM" in full_df.columns:
            top_teams = full_df.sort_values("KP_AdjEM", ascending=False).head(4)
            if "TM_KP" in top_teams.columns:
                default_teams = top_teams["TM_KP"].tolist()
        if not default_teams and all_teams:
            default_teams = all_teams[:min(4, len(all_teams))]
        # Here we assume selected_teams is already provided via multiselect
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
            seed_text = ""
            if "SEED_25" in row and not pd.isna(row["SEED_25"]):
                seed_text = f" - Seed {int(row['SEED_25'])}"
            subplot_titles.append(f"{i+1}) {team_name} ({conf}){seed_text}")
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
        # Adjust polar axes
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
    else:
        st.warning("Team names not available in dataset.")
        return None

# ----------------------------------------------------------------------------
# 6) Treemap Function
def create_treemap(df_notnull):
    try:
        # Limit to top 100 teams based on KP_Rank, if available
        if "KP_Rank" in df_notnull.columns:
            top_100_teams = df_notnull.sort_values(by="KP_Rank").head(100)
        else:
            top_100_teams = df_notnull.copy()
        if top_100_teams.empty:
            st.warning("No data to display in treemap.")
            return None
        required_columns = ["CONFERENCE", "TM_KP", "KP_AdjEM", "KP_Rank", "WIN_25", "LOSS_25"]
        if not all(col in top_100_teams.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in top_100_teams.columns]
            st.error(f"Missing required columns for treemap: {missing_cols}")
            return None
        treemap_data = top_100_teams.copy()  # Avoid modifying original df
        treemap_data["KP_AdjEM"] = pd.to_numeric(treemap_data["KP_AdjEM"], errors='coerce')
        treemap_data = treemap_data.dropna(subset=["KP_AdjEM"])
        if "TM_KP" not in treemap_data.columns:
            treemap_data["TM_KP"] = treemap_data["TEAM"]
        # Create advanced hover text
        def hover_text_func(x):
            base = (
                f"<b>{x['TM_KP']}</b><br>"
                f"KP Rank: {int(x['KP_Rank'])}<br>"
                f"Record: {int(x['WIN_25'])}-{int(x['LOSS_25'])}<br>"
                f"AdjEM: {x['KP_AdjEM']:.1f}<br>"
            )
            if "OFF EFF" in x and "DEF EFF" in x:
                base += f"OFF EFF: {x['OFF EFF']:.1f}<br>DEF EFF: {x['DEF EFF']:.1f}<br>"
            # Add seed if present
            if "SEED_25" in x and not pd.isna(x["SEED_25"]):
                base += f"Seed: {int(x['SEED_25'])}"
            return base
        treemap_data['hover_text'] = treemap_data.apply(hover_text_func, axis=1)
        treemap = px.treemap(
            treemap_data,
            path=["CONFERENCE", "TM_KP"],
            values="KP_AdjEM",
            color="KP_AdjEM",
            color_continuous_scale=px.colors.diverging.RdYlGn,
            hover_data=["hover_text"],
            title="<b>2025 KenPom AdjEM by Conference (Top 100)</b>"
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
            ),
            template="plotly_dark"
        )
        return treemap
    except Exception as e:
        st.error(f"An error occurred while generating treemap: {e}")
        return None

# ----------------------------------------------------------------------------
# 7) App Header & Tabs
st.title("NCAA BASKETBALL -- MARCH MADNESS 2025")
st.write("2025 MARCH MADNESS RESEARCH HUB")
st.write("Toggle tabs above to explore March Madness 2025 brackets, stats, visualizations")
col1, col2 = st.columns([6, 1])
with col1:
    if FinalFour25_logo:
        st.image(FinalFour25_logo, width=250)
    if NCAA_logo:
        st.image(NCAA_logo, width=250)
    if Conferences25_logo:
        st.image(Conferences25_logo, width=250)

treemap = create_treemap(df_main_notnull)

# Final tab structure:
# HOME, RADAR CHARTS, REGIONAL HEATMAPS, HISTOGRAM, CORRELATION HEATMAP, 
# CONFERENCE COMPARISON, TEAM METRICS COMPARISON, TBD
tab_home, tab_radar, tab_regions, tab_hist, tab_corr, tab_conf, tab_team, tab_tbd = st.tabs([
    "HOME", "RADAR CHARTS", "REGIONAL HEATMAPS", "HISTOGRAM",
    "CORRELATION HEATMAP", "CONFERENCE COMPARISON", "TEAM METRICS COMPARISON", "TBD"
])

# --- Home Tab ---
with tab_home:
    st.subheader("CONFERENCE-LEVEL TREEMAP")
    if treemap is not None:
        st.plotly_chart(treemap, use_container_width=True)
    else:
        st.warning("TREEMAP OVERHEATED.")
    if "CONFERENCE" in df_main.columns:
        conf_counts = df_main["CONFERENCE"].value_counts().reset_index()
        conf_counts.columns = ["Conference", "# Teams"]
        if "KP_AdjEM" in df_main.columns:
            conf_stats = (
                df_main.groupby("CONFERENCE")["KP_AdjEM"]
                .agg(["mean", "min", "max", "count"])
                .reset_index()
            )
            conf_stats.columns = ["Conference", "Avg AdjEM", "Min AdjEM", "Max AdjEM", "Count"]
            conf_stats = conf_stats.sort_values("Avg AdjEM", ascending=False)
            st.markdown("### COMPOSITE CONFERENCE POWER RATINGS")
            styled_conf_stats = (
                conf_stats.style
                .format({
                    "Avg AdjEM": "{:.2f}",
                    "Min AdjEM": "{:.2f}",
                    "Max AdjEM": "{:.2f}"
                })
                .background_gradient(cmap="RdYlGn", subset=["Avg AdjEM"])
                .background_gradient(cmap="inferno", subset=["Min AdjEM"])
                .background_gradient(cmap="viridis", subset=["Max AdjEM"])
                .set_table_styles(detailed_table_styles)
            )
            st.markdown(styled_conf_stats.to_html(), unsafe_allow_html=True)

# --- Radar Charts Tab ---
with tab_radar:
    st.header("Team Radar Performance Charts")
    radar_metrics = get_default_metrics()
    available_radar_metrics = [m for m in radar_metrics if m in df_main.columns]
    if len(available_radar_metrics) < 3:
        st.warning(f"Not enough radar metrics available. Need at least 4: {', '.join(radar_metrics)}")
    else:
        if "TM_KP" in df_main.columns:
            all_teams = df_main["TM_KP"].dropna().unique().tolist()
            default_teams = ['Duke', 'Kansas', 'Auburn', 'Houston']
            if "KP_AdjEM" in df_main.columns:
                top_teams = df_main.sort_values("KP_AdjEM", ascending=False).head(4)
                if "TM_KP" in top_teams.columns:
                    default_teams = top_teams["TM_KP"].tolist()
            if not default_teams and all_teams:
                default_teams = all_teams[:min(4, len(all_teams))]
            selected_teams = st.multiselect(
                "Select Teams to Compare:",
                options=sorted(all_teams),
                default=default_teams[:min(4, len(default_teams))]
            )
            if selected_teams:
                radar_fig = create_radar_chart(selected_teams, df_main)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
                else:
                    st.warning("Failed to display radar chart(s) for selected teams.")
            else:
                st.info("Please select at least one team to display radar charts.")
            with st.expander("About Radar Charts:"):
                st.markdown("""
                    Radar charts visualize team performance across 8 key metrics, compared to:
                    - NCAAM Average (red dashed line)
                    - Conference Average (green dotted line)

                    Each metric is scaled, where 5 == NCAAM average.
                    Values >5 are better; values <5 are worse.

                    Overall performance rating is derived from the average z-score across all metrics.
                """)
        else:
            st.warning("Team names not available in dataset.")

# --- Regional Heatmaps Tab ---
with tab_regions:
    st.header("BRACKET ANALYSIS")
    st.write("REGIONAL HEATFRAMES (W, X, Y, Z)")
    df_heat = df_main.copy()
    df_heat.loc["TOURNEY AVG"] = df_heat.mean(numeric_only=True)
    df_heat_T = df_heat.T
    # Updated region seeds as requested (16 seeds each + "TOURNEY AVG")
    # W region
    east_teams_2025 = [
        "Duke", "Tennessee", "Iowa St.", "Maryland", "Texas A&M", "Kansas", "UCLA", "Mississippi St.",
        "Georgia", "Ohio St.", "New Mexico", "Indiana", "Memphis", "Villanova", "Santa Clara", "Pittsburgh",
        "TOURNEY AVG"
    ]
    # X region
    west_teams_2025 = [
        "Auburn", "Alabama", "Gonzaga", "Purdue", "Illinois", "Saint Mary's", "Marquette", "Michigan",
        "Connecticut", "Oklahoma", "Xavier", "Northwestern", "Boise St.", "West Virginia", "Drake", "Liberty",
        "TOURNEY AVG"
    ]
    # Y region
    south_teams_2025 = [
        "Houston", "Texas Tech", "Kentucky", "St. John's", "Clemson", "Louisville", "Mississippi", "VCU",
        "North Carolina", "UC San Diego", "San Diego St.", "Vanderbilt", "Colorado St.", "Nebraska", "Penn St.", "Iowa",
        "TOURNEY AVG"
    ]
    # Z region
    midwest_teams_2025 = [
        "Florida", "Michigan St.", "Wisconsin", "Arizona", "Missouri", "BYU", "Baylor", "Oregon",
        "Creighton", "Arkansas", "Texas", "SMU", "Utah St.", "Cincinnati", "McNeese", "USC",
        "TOURNEY AVG"
    ]
    regions = {
        "W Region": east_teams_2025,
        "X Region": west_teams_2025,
        "Y Region": south_teams_2025,
        "Z Region": midwest_teams_2025
    }
    def safe_format(x):
        try:
            val = float(x)
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
            region_styler = region_df.style.format(safe_format)
            color_map_dict = {
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
            }
            for row_label, cmap in color_map_dict.items():
                if row_label in region_df.index:
                    region_styler = region_styler.background_gradient(cmap=cmap, subset=pd.IndexSlice[[row_label], :])
            region_styler = region_styler.set_table_styles(detailed_table_styles)
            st.markdown(region_styler.to_html(), unsafe_allow_html=True)
        else:
            st.info(f"No data available for {region_name}.")

# --- Histogram Tab ---
with tab_hist:
    st.header("Histogram")
    numeric_cols = [c for c in core_cols if c in df_main.columns and pd.api.types.is_numeric_dtype(df_main[c])]
    hist_metric = st.selectbox(
        "Select Metric for Histogram", numeric_cols,
        index=numeric_cols.index("KP_AdjEM") if "KP_AdjEM" in numeric_cols else 0
    )
    fig_hist = px.histogram(
        df_main, x=hist_metric, nbins=25, marginal="box",
        color_discrete_sequence=["dodgerblue"], template="plotly_dark",
        title=f"Distribution of {hist_metric} (All Teams)"
    )
    fig_hist.update_layout(bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)
    with st.expander("Glossary of Terms for Histogram Metrics:"):
        st.markdown("""
        **Histogram Metric Description:**
        - **KP_AdjEM**: Adjusted efficiency margin from KenPom ratings.
        - **OFF EFF/DEF EFF**: Offensive/Defensive efficiency.
        - Other metrics follow similar definitions as per NCAA advanced statistics.
        """)

# --- Correlation Heatmap Tab ---
with tab_corr:
    st.header("Correlation Heatmap")
    numeric_cols = [c for c in core_cols if c in df_main.columns and pd.api.types.is_numeric_dtype(df_main[c])]
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
        st.warning("Please select at least 2 metrics for correlation analysis.")
    with st.expander("Glossary of Terms for Correlation Metrics:"):
        st.markdown("""
        **Correlation Heatmap Glossary:**
        - **Correlation Coefficient**: Measures linear relationship between two variables.
        - **Positive/Negative Correlation**: Indicates the direction of the relationship.
        - **Metrics**: Derived from advanced team statistics.
        """)

# --- Conference Comparison Tab ---
with tab_conf:
    st.header("Conference Comparison")
    numeric_cols = [c for c in core_cols if c in df_main.columns and pd.api.types.is_numeric_dtype(df_main[c])]
    if "CONFERENCE" in df_main.columns:
        conf_metric = st.selectbox(
            "Select Metric for Conference Comparison", numeric_cols,
            index=numeric_cols.index("KP_AdjEM") if "KP_AdjEM" in numeric_cols else 0
        )
        conf_group = df_main.groupby("CONFERENCE")[conf_metric].mean().dropna().sort_values(ascending=False)
        fig_conf = px.bar(
            conf_group, y=conf_group.index, x=conf_group.values, orientation='h',
            title=f"Average {conf_metric} by Conference",
            labels={"y": "Conference", "x": conf_metric},
            color=conf_group.values, color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        # Add markers for individual teams per conference
        for conf in conf_group.index:
            teams = df_main[df_main["CONFERENCE"] == conf]
            fig_conf.add_trace(go.Scatter(
                x=teams[conf_metric], y=[conf] * len(teams),
                mode="markers", marker=dict(color="white", size=6, opacity=0.7),
                name=f"{conf} Teams"
            ))
        fig_conf.update_layout(showlegend=False)
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.error("Conference data is not available.")
    with st.expander("Glossary of Terms for Conference Comparison:"):
        st.markdown("""
        **Conference Comparison Glossary:**
        - **Avg AdjEM**: Average Adjusted Efficiency Margin.
        - **Conference**: Grouping of teams by their athletic conferences.
        - Metrics provide insight into relative conference strength.
        """)

# --- Team Metrics Comparison Tab ---
with tab_team:
    st.header("Team Metrics Comparison")
    numeric_cols = [c for c in core_cols if c in df_main.columns and pd.api.types.is_numeric_dtype(df_main[c])]
    x_metric = st.selectbox(
        "Select X-Axis Metric", numeric_cols,
        index=numeric_cols.index("OFF EFF") if "OFF EFF" in numeric_cols else 0
    )
    y_metric = st.selectbox(
        "Select Y-Axis Metric", numeric_cols,
        index=numeric_cols.index("DEF EFF") if "DEF EFF" in numeric_cols else 0
    )
    if "CONFERENCE" in df_main.columns:
        # Use absolute KP_AdjEM for sizing to prevent negative values.
        fig_scatter = px.scatter(
            df_main.reset_index(), x=x_metric, y=y_metric, color="CONFERENCE",
            hover_name="TEAM",
            size="KP_AdjEM_Pos" if "KP_AdjEM_Pos" in df_main.columns else None,
            size_max=15, opacity=0.8, template="plotly_dark",
            title=f"{y_metric} vs {x_metric}", height=700
        )
    else:
        fig_scatter = px.scatter(
            df_main.reset_index(), x=x_metric, y=y_metric,
            hover_name="TEAM", opacity=0.8,
            template="plotly_dark", title=f"{y_metric} vs {x_metric}"
        )
    # Add quadrant lines if metrics indicate Off vs Def comparisons
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
            fig_scatter.add_annotation(
                x=q["x"], y=q["y"], text=q["text"], showarrow=False,
                font=dict(color="white", size=10), opacity=0.7
            )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- TBD Tab ---
with tab_tbd:
    st.header("To Be Determined")
    st.info("Additional visualizations, bracket analysis, simulations coming soon.")

# ----------------------------------------------------------------------------
# GitHub Link & App Footer
st.markdown("---")
st.markdown("App code available on [GitHub](https://github.com/nehat312/march-madness-2025)")
st.stop()
