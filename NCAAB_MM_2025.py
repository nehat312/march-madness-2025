# 0) LIBRARY IMPORTS
import pandas as pd, numpy as np, os, math
import plotly.express as px, plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import streamlit as st
st.set_page_config(
    page_title="NCAA BASKETBALL -- MARCH MADNESS 2025",
    layout="wide",
    initial_sidebar_state="auto"
)
hide_menu_style = """<style>#MainMenu {visibility: hidden; } footer {visibility: hidden;}</style>"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# 1) DATA LOADING
abs_path = 'https://raw.githubusercontent.com/nehat312/march-madness-2025/main'
mm_csv = abs_path + '/data/mm_2025_database.csv'
@st.cache_data
def load_data():
    df = pd.read_csv(mm_csv, index_col=0)
    df.index.name = "TEAM"
    return df
mm_db = load_data()

# 2) SELECT COLUMNS & FIX MISSING DATA
core_cols = ["KP_Rank","WIN_25","LOSS_25","WIN% ALL GM","WIN% CLOSE GM","KP_AdjEM",
             "KP_SOS_AdjEM","OFF EFF","DEF EFF","AVG MARGIN","PTS/GM","OPP PTS/GM",
             "eFG%","OPP eFG%","TS%","OPP TS%","AST/TO%","STOCKS/GM","STOCKS-TOV/GM"]
extra_cols = ["CONFERENCE","TM_KP"]
all_cols = core_cols + extra_cols
actual_cols = [c for c in all_cols if c in mm_db.columns]
df_main = mm_db[actual_cols].copy()
req_cols = ["CONFERENCE","TM_KP","KP_AdjEM"]
df_main_notnull = df_main.dropna(subset=req_cols) if all(c in df_main.columns for c in req_cols) else df_main.copy()

# 3) LOAD LOGO
logo_path = "images/NCAA_logo1.png"
NCAA_logo = Image.open(logo_path) if os.path.exists(logo_path) else None

# 4) RADAR CHART FUNCTIONS
def get_default_metrics():
    return ['OFF EFF','DEF EFF','OFF REB/GM','DEF REB/GM','BLKS/GM','STL/GM','AST/GM','TO/GM']

def compute_tournament_stats(df):
    metrics = get_default_metrics()
    avgs = {m: df[m].mean() for m in metrics if m in df.columns}
    stdevs = {m: df[m].std() for m in metrics if m in df.columns}
    return avgs, stdevs

def compute_performance_text(team_row, t_avgs, t_stdevs):
    metrics = get_default_metrics(); z_vals = []
    for m in metrics:
        if m in team_row and m in t_avgs and m in t_stdevs:
            std = t_stdevs[m] if t_stdevs[m] > 0 else 1.0
            z = (team_row[m] - t_avgs[m]) / std
            if m in ['DEF EFF','TO/GM']:
                z = -z
            z_vals.append(z)
    if not z_vals:
        return "No Data"
    avg_z = sum(z_vals) / len(z_vals)
    return "Elite" if avg_z > 1.5 else "Above Average" if avg_z > 0.5 else "Average" if avg_z > -0.5 else "Below Average" if avg_z > -1.5 else "Poor"

def get_radar_traces(team_row, t_avgs, t_stdevs, conf_df, show_legend=False):
    metrics = get_default_metrics()
    avail = [m for m in metrics if m in team_row.index and m in t_avgs]
    if not avail:
        return []
    z_scores = []
    for m in avail:
        std = t_stdevs[m] if t_stdevs[m] > 0 else 1.0
        z = (team_row[m] - t_avgs[m]) / std
        if m in ['DEF EFF','TO/GM']:
            z = -z
        z_scores.append(z)
    scale = 1.5
    scaled_team = [min(max(5 + z * scale, 0), 10) for z in z_scores]
    scaled_ncaam = [5] * len(avail)
    if team_row.get('CONFERENCE') and not conf_df.empty:
        conf_vals = []
        for m in avail:
            if m in conf_df.columns:
                conf_avg = conf_df[m].mean()
                std = t_stdevs[m] if t_stdevs[m] > 0 else 1.0
                z = (conf_avg - t_avgs[m]) / std
                if m in ['DEF EFF','TO/GM']:
                    z = -z
                conf_vals.append(z)
            else:
                conf_vals.append(0)
        scaled_conf = [min(max(5 + z, 0), 10) for z in conf_vals]
    else:
        scaled_conf = [5] * len(avail)
    circ = avail + [avail[0]]
    team_circ = scaled_team + [scaled_team[0]]
    ncaam_circ = scaled_ncaam + [scaled_ncaam[0]]
    conf_circ = scaled_conf + [scaled_conf[0]]
    team_name = team_row.name if hasattr(team_row, 'name') else "Team"
    trace_team = go.Scatterpolar(
        r=team_circ, theta=circ, fill='toself',
        fillcolor='rgba(30,144,255,0.3)', name='TEAM',
        line=dict(color='dodgerblue', width=2),
        showlegend=show_legend,
        hovertemplate="%{theta}: %{r:.1f}<extra>" + f"{team_name}</extra>"
    )
    trace_ncaam = go.Scatterpolar(
        r=ncaam_circ, theta=circ, fill='toself',
        fillcolor='rgba(255,99,71,0.2)', name='NCAAM AVG',
        line=dict(color='tomato', width=2, dash='dash'),
        showlegend=show_legend,
        hoverinfo='skip'
    )
    trace_conf = go.Scatterpolar(
        r=conf_circ, theta=circ, fill='toself',
        fillcolor='rgba(50,205,50,0.2)', name='CONFERENCE',
        line=dict(color='limegreen', width=2, dash='dot'),
        showlegend=show_legend,
        hoverinfo='skip'
    )
    return [trace_team, trace_ncaam, trace_conf]

def create_radar_chart(teams, df):
    if not teams:
        return None
    subset = df[df['TM_KP'].isin(teams)].copy().reset_index()
    if subset.empty:
        return None
    t_avgs, t_stdevs = compute_tournament_stats(df)
    n = len(subset)
    rows, cols = (1, n) if n <= 4 else (2, min(4, math.ceil(n/2)))
    titles = [f"{i+1}) {row.get('TM_KP','Team')} ({row.get('CONFERENCE','N/A')})" for i, row in subset.iterrows()]
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'polar'}] * cols for _ in range(rows)],
        subplot_titles=titles,
        horizontal_spacing=0.07, vertical_spacing=0.15
    )
    fig.update_layout(
        height=400 if rows == 1 else 800,
        title="Radar Dashboards for Selected Teams",
        template='plotly_dark',
        font=dict(size=12),
        showlegend=True
    )
    fig.update_polars(
        radialaxis=dict(
            tickmode='array',
            tickvals=[0, 2, 4, 6, 8, 10],
            ticktext=['0', '2', '4', '6', '8', '10'],
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
    for idx, row in subset.iterrows():
        r = idx // cols + 1; c = idx % cols + 1; legend = (idx == 0)
        conf = row.get('CONFERENCE')
        conf_df = df[df['CONFERENCE'] == conf] if conf else pd.DataFrame()
        for tr in get_radar_traces(row, t_avgs, t_stdevs, conf_df, show_legend=legend):
            fig.add_trace(tr, row=r, col=c)
        perf = compute_performance_text(row, t_avgs, t_stdevs)
        polar_key = "polar" if (r - 1) * cols + c == 1 else f"polar{(r - 1) * cols + c}"
        if polar_key in fig.layout:
            dx = fig.layout[polar_key].domain.x
            dy = fig.layout[polar_key].domain.y
            x_annot = dx[0] + 0.03; y_annot = dy[1] - 0.03
        else:
            x_annot, y_annot = 0.1, 0.9
        fig.add_annotation(
            x=x_annot, y=y_annot, xref="paper", yref="paper",
            text=f"<b>{perf}</b>", showarrow=False,
            font=dict(size=12, color="gold")
        )
    return fig

# 5) TREEMAP FUNCTION
def create_treemap(df):
    if all(c in df.columns for c in ["CONFERENCE", "TM_KP", "KP_AdjEM"]):
        data = df.reset_index()
        data["KP_AdjEM"] = pd.to_numeric(data["KP_AdjEM"], errors='coerce')
        data['hover_text'] = data.apply(
            lambda x: f"<b>{x['TM_KP']}</b><br>KP Rank: {x['KP_Rank']:.0f}<br>Record: {x['WIN_25']:.0f}-{x['LOSS_25']:.0f}<br>AdjEM: {x['KP_AdjEM']:.1f}<br>OFF EFF: {x.get('OFF EFF','N/A')}<br>DEF EFF: {x.get('DEF EFF','N/A')}",
            axis=1
        )
        treemap = px.treemap(
            data_frame=data,
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
                title="AdjEM",
                thicknessmode="pixels",
                thickness=15,
                lenmode="pixels",
                len=300,
                yanchor="top",
                y=1,
                ticks="outside"
            )
        )
        return treemap
    return None

# STREAMLIT APP LAYOUT
st.title("NCAA BASKETBALL -- MARCH MADNESS 2025")
st.write("2025 MARCH MADNESS RESEARCH HUB")
col1, col2 = st.columns([6, 1])
with col2:
    if NCAA_logo:
        st.image(NCAA_logo, width=150)

treemap = create_treemap(df_main_notnull)
tab_home, tab_radar = st.tabs(["Home", "Team Radar Charts"])

with tab_home:
    st.subheader("Conference Treemap Overview")
    if treemap:
        st.plotly_chart(treemap, use_container_width=True)
    else:
        st.warning("Treemap not available due to missing data.")
    st.markdown("### Key Insights")
    st.write("Explore teams and visualizations for March Madness 2025.")

with tab_radar:
    st.header("Team Radar Performance Charts")
    if "TM_KP" in df_main.columns:
        teams = sorted(df_main["TM_KP"].dropna().unique().tolist())
        default = df_main.sort_values("KP_AdjEM", ascending=False)["TM_KP"].head(6).tolist() if "KP_AdjEM" in df_main.columns else teams[:6]
        selected = st.multiselect("Select Teams to Compare (4-8 recommended)", options=teams, default=default)
        if selected:
            radar_fig = create_radar_chart(selected, df_main)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.warning("Could not create radar chart with selected teams.")
        else:
            st.info("Please select at least one team to display radar charts.")
    else:
        st.warning("Team names not available.")

st.markdown("GitHub Repo: [march-madness-2025](https://github.com/nehat312/march-madness-2025)")
st.stop()
