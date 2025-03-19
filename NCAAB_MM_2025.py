import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import base64
from io import BytesIO
from PIL import Image

import os, math, logging, random
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Setup ---
st.set_page_config(page_title="MARCH MADNESS 2025 -- NCAAM BASKETBALL",
                   layout="wide", initial_sidebar_state="auto",
                   page_icon="🏀",)

# Hide default Streamlit menu/footer
hide_menu_style = """
 <style>
 #MainMenu {visibility: hidden; }
 footer {visibility: hidden;}
 </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Enhanced CSS with better typography and spacing
custom_css = """
<style>
    /* Force Arial sitewide */
    body, p, div, table, th, td, span, input, select, textarea, label {
        font-family: 'Arial', sans-serif !important;
    }
    /* Typography improvements for headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Arial', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    /* Table styling enhancements */
    table {
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    table, th, td {
        border: 1px solid #222;
    }
    th, td {
        text-align: center;
        padding: 8px 10px;
        font-weight: bold;
        font-size: 13px;
        vertical-align: middle;
    }
    th {
        background-color: #0360CE;
        color: white;
        font-weight: 800;
    }
    tr:nth-child(even) {
        background-color: rgba(255, 255, 255, 0.05);
    }
    tr:hover {
        background-color: rgba(255, 255, 255, 0.08);
    }
    /* Card-like containers for better visual hierarchy */
    .stat-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    /* Hidden Streamlit elements */
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    /* Custom badges for team performance */
    .badge-elite {
        background-color: gold;
        color: black;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 11px;
    }
    .badge-solid {
        background-color: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 11px;
    }
    .badge-mid {
        background-color: #2196F3;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 11px;
    }
    .badge-subpar {
        background-color: #FF9800;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 11px;
    }
    .badge-weak {
        background-color: #F44336;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 11px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load Data from GitHub CSV
abs_path = r'https://raw.githubusercontent.com/nehat312/march-madness-2025/main'
mm_database_csv = abs_path + '/data/mm_2025_database.csv'

@st.cache_data
def load_data():
    df = pd.read_csv(mm_database_csv, index_col=0)
    df.index.name = "TEAM"
    return df

mm_database_2025 = load_data()

# ----------------------------------------------------------------------------
# Select Relevant Columns (including radar metrics)
core_cols = ["WIN_25", "LOSS_25", "WIN% ALL GM", "WIN% CLOSE GM",
             "KP_Rank", "NET_25", "SEED_25", 'REGION_25',
             "KP_AdjEM", "KP_SOS_AdjEM", "OFF EFF", "DEF EFF",
             "KP_AdjO", "KP_AdjD",
             #'TR_ORk_25', 'TR_DRk_25',  
             "AVG MARGIN", "PTS/GM", "OPP PTS/GM",
             "eFG%", "OPP eFG%", "TS%", "OPP TS%", 
             "OFF REB/GM", "DEF REB/GM",
             "BLKS/GM", "STL/GM", "AST/GM", "TO/GM", 
             "AST/TO%", "STOCKS/GM", "STOCKS-TOV/GM",
             ]

extra_cols_for_treemap = ["CONFERENCE", "TM_KP"] #, "SEED_25"
all_desired_cols = core_cols + extra_cols_for_treemap
actual_cols = [c for c in all_desired_cols if c in mm_database_2025.columns]
df_main = mm_database_2025[actual_cols].copy()

if "TM_KP" not in df_main.columns:
    df_main["TM_KP"] = df_main.index

if "TM_TR" not in df_main.columns:
    df_main["TM_TR"] = df_main.index

# Offset KP_AdjEM for marker sizes (avoid negatives)
if "KP_AdjEM" in df_main.columns:
    min_adj = df_main["KP_AdjEM"].min()
    offset = (-min_adj + 1) if min_adj < 0 else 0
    df_main["KP_AdjEM_Offset"] = df_main["KP_AdjEM"] + offset

# ----------------------------------------------------------------------------
# Clean Data for Treemap
required_path_cols = ["CONFERENCE", "TM_KP", "KP_AdjEM"]
if all(col in df_main.columns for col in required_path_cols):
    df_main_notnull = df_main.dropna(subset=required_path_cols, how="any").copy()
else:
    df_main_notnull = df_main.copy()

# ----------------------------------------------------------------------------
# Logo Loading / Syntax Configuration
logo_path = "images/NCAA_logo1.png"
FinalFour25_logo_path = "images/ncaab_mens_finalfour2025_logo.png"
Conferences25_logo_path = "images/ncaab_conferences_2025.png"
A10_logo_path = "images/A10_logo.png"
ACC_logo_path = "images/ACC_logo.png"
AAC_logo_path = "images/AAC_logo.png"
AEC_logo_path = "images/AEC_logo.png"
ASUN_logo_path = "images/ASUN_logo.png"
B10_logo_path = "images/B10_logo.png"
B12_logo_path = "images/B12_logo.png"
BE_logo_path = "images/BE_logo.png"
BSouth_logo_path = "images/BSouth_logo.png"
BSky_logo_path = "images/BSky_logo.png"
BWest_logo_path = "images/BWest_logo.png"
CAA_logo_path = "images/CAA_logo.png"
CUSA_logo_path = "images/CUSA_logo.png"
Horizon_logo_path = "images/Horizon_logo.png"
Ivy_logo_path = "images/Ivy_logo.png"
MAAC_logo_path = "images/MAAC_logo.png"
MAC_logo_path = "images/MAC_logo.png"
MEAC_logo_path = "images/MEAC_logo.png"
MVC_logo_path = "images/MVC_logo.png"
MWC_logo_path = "images/MWC_logo.png"
NEC_logo_path = "images/NEC_logo.png"
OVC_logo_path = "images/OVC_logo.png"
Patriot_logo_path = "images/Patriot_logo.png"
SBC_logo_path = "images/SBC_logo.png"
SEC_logo_path = "images/SEC_logo.png"
SoCon_logo_path = "images/SoCon_logo.png"
Southland_logo_path = "images/Southland_logo.png"
Summit_logo_path = "images/Summit_logo.png"
SWAC_logo_path = "images/SWAC_logo.png"
WAC_logo_path = "images/WAC_logo.png"
WCC_logo_path = "images/WCC_logo.png"

NCAA_logo = Image.open(logo_path) if os.path.exists(logo_path) else None
FinalFour25_logo = Image.open(FinalFour25_logo_path) if os.path.exists(FinalFour25_logo_path) else None
Conferences25_logo = Image.open(Conferences25_logo_path) if os.path.exists(Conferences25_logo_path) else None
A10_logo = Image.open(A10_logo_path) if os.path.exists(A10_logo_path) else None
ACC_logo = Image.open(ACC_logo_path) if os.path.exists(ACC_logo_path) else None
AAC_logo = Image.open(AAC_logo_path) if os.path.exists(AAC_logo_path) else None
AEC_logo = Image.open(AEC_logo_path) if os.path.exists(AEC_logo_path) else None
ASUN_logo = Image.open(ASUN_logo_path) if os.path.exists(ASUN_logo_path) else None
B10_logo = Image.open(B10_logo_path) if os.path.exists(B10_logo_path) else None
B12_logo = Image.open(B12_logo_path) if os.path.exists(B12_logo_path) else None
BE_logo = Image.open(BE_logo_path) if os.path.exists(BE_logo_path) else None
BSouth_logo = Image.open(BSouth_logo_path) if os.path.exists(BSouth_logo_path) else None
BSky_logo = Image.open(BSky_logo_path) if os.path.exists(BSky_logo_path) else None
BWest_logo = Image.open(BWest_logo_path) if os.path.exists(BWest_logo_path) else None
CAA_logo = Image.open(CAA_logo_path) if os.path.exists(CAA_logo_path) else None
CUSA_logo = Image.open(CUSA_logo_path) if os.path.exists(CUSA_logo_path) else None
Horizon_logo = Image.open(Horizon_logo_path) if os.path.exists(Horizon_logo_path) else None
Ivy_logo = Image.open(Ivy_logo_path) if os.path.exists(Ivy_logo_path) else None
MAAC_logo = Image.open(MAAC_logo_path) if os.path.exists(MAAC_logo_path) else None
MAC_logo = Image.open(MAC_logo_path) if os.path.exists(MAC_logo_path) else None
MEAC_logo = Image.open(MEAC_logo_path) if os.path.exists(MEAC_logo_path) else None
MVC_logo = Image.open(MVC_logo_path) if os.path.exists(MVC_logo_path) else None
MWC_logo = Image.open(MWC_logo_path) if os.path.exists(MWC_logo_path) else None
NEC_logo = Image.open(NEC_logo_path) if os.path.exists(NEC_logo_path) else None
OVC_logo = Image.open(OVC_logo_path) if os.path.exists(OVC_logo_path) else None
Patriot_logo = Image.open(Patriot_logo_path) if os.path.exists(Patriot_logo_path) else None
SBC_logo = Image.open(SBC_logo_path) if os.path.exists(SBC_logo_path) else None
SEC_logo = Image.open(SEC_logo_path) if os.path.exists(SEC_logo_path) else None
SWAC_logo = Image.open(SWAC_logo_path) if os.path.exists(SWAC_logo_path) else None
Summit_logo = Image.open(Summit_logo_path) if os.path.exists(Summit_logo_path) else None
SoCon_logo = Image.open(SoCon_logo_path) if os.path.exists(SoCon_logo_path) else None
Southland_logo = Image.open(Southland_logo_path) if os.path.exists(Southland_logo_path) else None
WAC_logo = Image.open(WAC_logo_path) if os.path.exists(WAC_logo_path) else None
WCC_logo = Image.open(WCC_logo_path) if os.path.exists(WCC_logo_path) else None

conference_logo_map = {"A10": A10_logo, "ACC": ACC_logo, "Amer": AAC_logo, "AE": AEC_logo, "ASun": ASUN_logo, "B10": B10_logo, "B12": B12_logo,
                       "BE": BE_logo, "BSth": BSouth_logo, "BSky": BSky_logo, "BW": BWest_logo, "CAA": CAA_logo, "CUSA": CUSA_logo,
                       "Horz": Horizon_logo, "Ivy": Ivy_logo, "MAAC": MAAC_logo, "MAC": MAC_logo, "MEAC": MEAC_logo, "MVC": MVC_logo, "MWC": MWC_logo,
                       "NEC": NEC_logo, "OVC": OVC_logo, "PL": Patriot_logo, "SB": SBC_logo, "SEC": SEC_logo, "SoCon": SoCon_logo, "Southland": Southland_logo,
                       "Sum": Summit_logo, "SWAC": SWAC_logo, "WAC": WAC_logo, "WCC": WCC_logo,
                       }

#####################################
def image_to_base64(img_obj):  # Convert PIL Image to base64
    if img_obj is None:
        return None
    with BytesIO() as buffer:
        img_obj.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

def get_conf_logo_html(conf_name):  # Return HTML <img> + conference name for table column 
    img_obj = conference_logo_map.get(conf_name, None)
    if img_obj:
        encoded = image_to_base64(img_obj)
        if encoded:
            return f'<img src="data:image/png;base64,{encoded}" width="40" style="vertical-align: middle;" /> {conf_name}'
    return conf_name

# Global visualization settings
viz_margin_dict = dict(l=20, r=20, t=50, b=20)
viz_bg_color = '#0360CE'
viz_font_dict = dict(size=12, color='#FFFFFF')
RdYlGn = px.colors.diverging.RdYlGn
Spectral = px.colors.diverging.Spectral
RdBu_r = px.colors.diverging.RdBu_r

# ----------------------------------------------------------------------------
# Additional table styling used in Pandas Styler
header = {
    'selector': 'th',
    'props': [
        ('background-color', '#0360CE'),
        ('color', 'white'),
        ('text-align', 'center'),
        ('vertical-align', 'middle'),
        ('font-weight', 'bold'),
        ('border-bottom', '2px solid #000000'),
        ('border-right', '1px dashed #000000'),
    ]
}
detailed_table_styles = [header]

# ----------------------------------------------------------------------------
# Radar Chart Functions
def get_default_metrics():
    """Return metrics to be used in radar charts z-score logic."""
    return [
        'AVG MARGIN',
        'KP_AdjEM',
        'OFF EFF',
        'DEF EFF',
        'AST/TO%',
        'STOCKS-TOV/GM'
    ]

def compute_tournament_stats(df):
    """Compute overall averages and standard deviations for radar metrics."""
    metrics = get_default_metrics()
    avgs = {m: df[m].mean() for m in metrics if m in df.columns}
    stdevs = {m: df[m].std() for m in metrics if m in df.columns}
    return avgs, stdevs

def compute_performance_text(team_row, t_avgs, t_stdevs):
    """Return a dict with performance text and badge class based on average z-score."""
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
        return {"text": "NO DATA", "class": "badge-mid"}
    avg_z = sum(z_vals) / len(z_vals)
    if avg_z > 1.5:
        return {"text": "ELITE", "class": "badge-elite"}
    elif avg_z > 0.5:
        return {"text": "SOLID", "class": "badge-solid"}
    elif avg_z > -0.5:
        return {"text": "MID", "class": "badge-mid"}
    elif avg_z > -1.5:
        return {"text": "SUBPAR", "class": "badge-subpar"}
    else:
        return {"text": "WEAK", "class": "badge-weak"}

def get_radar_traces(team_row, t_avgs, t_stdevs, conf_df, show_legend=False):
    """
    Returns three Scatterpolar traces for:
      TEAM performance, National average (flat at 5), and Conference average.
    """
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
    # Compute conference values if available
    conf = team_row['CONFERENCE'] if 'CONFERENCE' in team_row else None
    if conf and not conf_df.empty:
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
    # Close the loop for the radar chart
    metrics_circ = available_metrics + [available_metrics[0]]
    team_scaled_circ = scaled_team + [scaled_team[0]]
    ncaam_scaled_circ = scaled_ncaam + [scaled_ncaam[0]]
    conf_scaled_circ = scaled_conf + [scaled_conf[0]]
    seed_info = ""
    if 'SEED_25' in team_row and pd.notna(team_row['SEED_25']):
        seed_info = f"(Seed {int(team_row['SEED_25'])})"
    team_name = f"{team_row.name} {seed_info}".strip()
    trace_team = go.Scatterpolar(
        r=team_scaled_circ,
        theta=metrics_circ,
        fill='toself',
        fillcolor='rgba(30,144,255,0.3)',
        name='TEAM',
        line=dict(color='dodgerblue', width=2),
        showlegend=show_legend,
        hovertemplate="%{theta}: %{r:.1f}<extra>" + f"{team_name}</extra>"
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

def create_radar_chart_figure(team_row, full_df, is_subplot=False, subplot_row=None, subplot_col=None):
    """
    Creates a Plotly radar chart figure (or adds traces to a subplot).
    Always uses make_subplots for proper row/col reference.
    """
    t_avgs, t_stdevs = compute_tournament_stats(full_df)
    conf = team_row.get('CONFERENCE', None)
    if conf:
        conf_df = full_df[full_df['CONFERENCE'] == conf]
    else:
        conf_df = pd.DataFrame()
    show_legend = not is_subplot
    traces = get_radar_traces(team_row, t_avgs, t_stdevs, conf_df, show_legend=show_legend)
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])
    fig.update_layout(
        template='plotly_dark',
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=show_legend,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0.8)",
        plot_bgcolor="rgba(0,0,0,0.8)",
        height=350,
    )
    fig.update_polars(
        radialaxis=dict(
            tickmode='array',
            tickvals=[0,2,4,6,8,10],
            ticktext=['0','2','4','6','8','10'],
            tickfont=dict(size=11, family="Arial, sans-serif"),
            showline=False,
            gridcolor='rgba(255,255,255,0.2)',
        ),
        angularaxis=dict(
            tickfont=dict(size=12, family="Arial, sans-serif", color="white"),
            tickangle=0,
            showline=False,
            gridcolor='rgba(255,255,255,0.2)',
            linecolor='rgba(255,255,255,0.2)'
        ),
        bgcolor="rgba(0,0,0,0.8)"
    )
    for tr in traces:
        if is_subplot:
            fig.add_trace(tr, row=subplot_row, col=subplot_col)
        else:
            fig.add_trace(tr, row=1, col=1)
    perf_data = compute_performance_text(team_row, t_avgs, t_stdevs)
    fig.add_annotation(
        x=0.01, y=0.99, xref="paper", yref="paper",
        text=f"<b>{perf_data['text']}</b>",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor={
            "badge-elite": "gold",
            "badge-solid": "#4CAF50",
            "badge-mid": "#2196F3",
            "badge-subpar": "#FF9800",
            "badge-weak": "#F44336"
        }.get(perf_data['class'], "#2196F3"),
        bordercolor="white", borderwidth=1.5, borderpad=4, opacity=0.9
    )
    seed_str = ""
    if 'SEED_25' in team_row and pd.notna(team_row['SEED_25']):
        seed_str = f"#{int(team_row['SEED_25'])} | "
    team_str = f"{seed_str}{team_row.name} | {team_row.get('REGION_25','')} | {team_row.get('CONFERENCE','')}"
    if not is_subplot:
        fig.add_annotation(
            text=team_str,
            x=0.5, y=1.08, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=15, color="white"),
            align="center"
        )
    return fig

def create_single_radar_chart(team_row, full_df, key=None):
    """Creates a single-team radar chart and renders it in Streamlit."""
    if team_row is None or team_row.empty:
        st.warning("No data found for this team.")
        return
    fig = create_radar_chart_figure(team_row, full_df)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=key)

def create_radar_chart(selected_teams, full_df):
    """
    Generates a radar chart subplot grid for the selected teams.
    The layout adapts based on the number of teams.
    """
    metrics = get_default_metrics()
    available_radar_metrics = [m for m in full_df.columns]
    if len(available_radar_metrics) < 3:
        return None
    team_mask = full_df['TM_KP'].isin(selected_teams)
    subset = full_df[team_mask].copy().reset_index()
    if subset.empty:
        return None
    t_avgs, t_stdevs = compute_tournament_stats(full_df)
    n_teams = len(subset)
    fig_height = 500 if n_teams <= 2 else (900 if n_teams <= 4 else 1100)
    row_count = 1 if n_teams <= 4 else 2
    col_count = n_teams if row_count == 1 else min(4, math.ceil(n_teams / 2))
    subplot_titles = []
    for i, row in subset.iterrows():
        team_name = row['TM_KP'] if 'TM_KP' in row else f"Team {i+1}"
        conf = row['CONFERENCE'] if 'CONFERENCE' in row else "N/A"
        seed_str = ""
        if "SEED_25" in row and not pd.isna(row["SEED_25"]):
            seed_str = f" - Seed {int(row['SEED_25'])}"
        perf_data = compute_performance_text(row, t_avgs, t_stdevs)
        subplot_titles.append(f"{i+1}) {team_name} ({conf}){seed_str}")
    fig = make_subplots(
        rows=row_count, cols=col_count,
        specs=[[{'type': 'polar'}] * col_count for _ in range(row_count)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.15
    )
    fig.update_layout(
        height=fig_height,
        template='plotly_dark',
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, bgcolor="rgba(0,0,0,0.1)"),
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor="rgba(0,0,0,0.8)",
        plot_bgcolor="rgba(0,0,0,0.8)"
    )
    fig.update_polars(
        radialaxis=dict(
            tickmode='array', tickvals=[0,2,4,6,8,10],
            ticktext=['0','2','4','6','8','10'],
            tickfont=dict(size=11, family="Arial, sans-serif"),
            showline=False, gridcolor='rgba(255,255,255,0.2)'
        ),
        angularaxis=dict(
            tickfont=dict(size=12, family="Arial, sans-serif", color="white"),
            tickangle=0, showline=False,
            gridcolor='rgba(255,255,255,0.2)',
            linecolor='rgba(255,255,255,0.2)'
        ),
        bgcolor="rgba(0,0,0,0.8)"
    )
    for idx, team_row in subset.iterrows():
        r = idx // col_count + 1
        c = idx % col_count + 1
        show_legend = (idx == 0)
        sub_fig = create_radar_chart_figure(team_row, full_df, is_subplot=True, subplot_row=r, subplot_col=c)
        for tr in sub_fig.data:
            fig.add_trace(tr, row=r, col=c)
        perf_data = compute_performance_text(team_row, t_avgs, t_stdevs)
        polar_idx = (r - 1) * col_count + c
        polar_key = "polar" if polar_idx == 1 else f"polar{polar_idx}"
        if polar_key in fig.layout:
            domain_x = fig.layout[polar_key].domain.x
            domain_y = fig.layout[polar_key].domain.y
            x_annot = domain_x[0] + 0.03
            y_annot = domain_y[1] - 0.03
        else:
            x_annot, y_annot = 0.05, 0.95
        fig.add_annotation(
            x=x_annot, y=y_annot, xref="paper", yref="paper",
            text=f"<b>{perf_data['text']}</b>",
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor={
                "badge-elite": "gold", "badge-solid": "#4CAF50",
                "badge-mid": "#2196F3", "badge-subpar": "#FF9800",
                "badge-weak": "#F44336"
            }.get(perf_data['class'], "#2196F3"),
            bordercolor="white", borderwidth=1, borderpad=4, opacity=0.9
        )
    return fig

def get_radar_zscores(team_row, t_avgs, t_stdevs, conf_df):
    """
    For a single team, produce three radial vectors on a 0..10 scale:
      - Team z-scores, National average (always 5), and Conference average.
    """
    metrics = get_default_metrics()
    used_metrics = [m for m in metrics if m in team_row and m in t_avgs]
    if not used_metrics:
        return [], [], [], used_metrics
    z_scores = []
    for m in used_metrics:
        val = team_row[m]
        mean_ = t_avgs[m] if m in t_avgs else 0
        stdev_ = t_stdevs[m] if (m in t_stdevs and t_stdevs[m] > 0) else 1.0
        if m in ['DEF EFF', 'TO/GM']:
            z = -(val - mean_) / stdev_
        else:
            z = (val - mean_) / stdev_
        z_scores.append(z)
    scale_factor = 1.5
    team_scaled = [max(0, min(10, 5 + (z * scale_factor))) for z in z_scores]
    ncaam_scaled = [5]*len(team_scaled)
    conf_scaled = []
    for m in used_metrics:
        if conf_df is not None and m in conf_df.columns:
            conf_val = conf_df[m].mean()
        else:
            conf_val = t_avgs[m]
        stdev_ = t_stdevs[m] if (m in t_stdevs and t_stdevs[m] > 0) else 1.0
        if m in ['DEF EFF', 'TO/GM']:
            zc = -(conf_val - t_avgs[m]) / stdev_
        else:
            zc = (conf_val - t_avgs[m]) / stdev_
        val_scaled = max(0, min(10, 5 + (zc * scale_factor)))
        conf_scaled.append(val_scaled)
    return team_scaled, ncaam_scaled, conf_scaled, used_metrics

def create_region_seeding_radar_grid(df):
    """
    Creates a 4x16 grid of radar charts for tournament seeding.
    """
    if 'SEED_25' not in df.columns or ('REGION_25' not in df.columns and 'REG_CODE_25' not in df.columns):
        st.error("Required columns for bracket visualization are missing")
        return
    df['SEED_25'] = pd.to_numeric(df['SEED_25'], errors='coerce')
    tourney_teams = df[df['SEED_25'].notna()].copy()
    regions = ["East", "West", "South", "Midwest"]
    region_colors = ["#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    st.markdown("""
    <style>
    .radar-grid-container {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="radar-grid-container">', unsafe_allow_html=True)
        header_cols = st.columns(4)
        for i, region in enumerate(regions):
            header_cols[i].markdown(
                f"<h3 style='text-align:center;color:{region_colors[i]};"
                f"font-weight:bold;text-shadow: 1px 1px 2px black;"
                f"border-bottom: 2px solid {region_colors[i]};padding-bottom: 5px;'>{region}</h3>",
                unsafe_allow_html=True
            )
        for seed in range(1, 17):
            row_cols = st.columns(4)
            for i, region in enumerate(regions):
                if 'REGION_25' in tourney_teams.columns:
                    team_filter = tourney_teams['REGION_25'].str.strip().str.lower() == region.lower()
                else:
                    team_filter = tourney_teams['REG_CODE_25'].str.strip().str.lower() == region.lower()
                team = tourney_teams[team_filter & (tourney_teams['SEED_25'] == seed)]
                if not team.empty:
                    team = team.iloc[0]
                    unique_key = f"bracket_radar_{team['TM_KP']}_{region}_{seed}"
                    with row_cols[i]:
                        create_single_radar_chart(team, df, key=unique_key)
                else:
                    row_cols[i].markdown(
                        f"<div style='height:300px;display:flex;align-items:center;justify-content:center;color:white;'>No Team (Seed {seed})</div>",
                        unsafe_allow_html=True
                    )
        st.markdown('</div>', unsafe_allow_html=True)

def create_bracket_radar_grid():
    """Creates a 4x16 grid of radar charts for the bracket."""
    df = TR_df.copy()
    if not all(col in df.columns for col in ['SEED_25', 'REG_CODE_25', 'REGION_25']):
        st.error("Required columns for bracket visualization are missing")
        return
    tourney_teams = df[df['SEED_25'].notna()].copy()
    st.markdown("""
    <style>
    .radar-grid-container {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="radar-grid-container">', unsafe_allow_html=True)
        cols = st.columns(4)
        regions = ["East", "West", "South", "Midwest"]
        for i, region in enumerate(regions):
            cols[i].markdown(f"<h3 style='text-align:center;color:white;'>{region}</h3>", unsafe_allow_html=True)
        for seed in range(1, 17):
            cols = st.columns(4)
            for i, region in enumerate(regions):
                team = tourney_teams[(tourney_teams['REGION_25'] == region) & (tourney_teams['SEED_25'] == seed)]
                if not team.empty:
                    team = team.iloc[0]
                    with cols[i]:
                        create_team_radar(team, dark_mode=True)
                else:
                    cols[i].markdown(
                        f"<div style='height:200px;display:flex;align-items:center;justify-content:center;color:white;'>No Team (Seed {seed})</div>",
                        unsafe_allow_html=True
                    )
        st.markdown('</div>', unsafe_allow_html=True)

def create_seed_radar_grid(df, region_teams):
    """
    Creates a 4x16 grid of radar charts (one for each region and seed).
    """
    required_cols = ['SEED_25', 'REG_CODE_25', 'REGION_25']
    if not all(col in df.columns for col in required_cols):
        st.error("Required columns for bracket visualization are missing")
        return
    tourney_teams = df[df['SEED_25'].notna()].copy()
    st.markdown("""
    <style>
    .radar-grid-container {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="radar-grid-container">', unsafe_allow_html=True)
        cols = st.columns(4)
        regions = ["East", "West", "South", "Midwest"]
        for i, region in enumerate(regions):
            cols[i].markdown(f"<h3 style='text-align:center;color:white;'>{region}</h3>", unsafe_allow_html=True)
        for seed in range(1, 17):
            cols = st.columns(4)
            for i, region in enumerate(regions):
                team = tourney_teams[(tourney_teams['REGION_25'] == region) & (tourney_teams['SEED_25'] == seed)]
                if not team.empty:
                    team = team.iloc[0]
                    with cols[i]:
                        create_team_radar(team, dark_mode=True)
                else:
                    cols[i].markdown(
                        f"<div style='height:200px;display:flex;align-items:center;justify-content:center;color:white;'>No Team (Seed {seed})</div>",
                        unsafe_allow_html=True
                    )
        st.markdown('</div>', unsafe_allow_html=True)

def create_team_radar(team, dark_mode=True, key=None):
    """Creates a radar chart for a single team with annotations and color coding."""
    team_name = team['TM_KP']
    seed = int(team['SEED_25']) if pd.notna(team['SEED_25']) else 0
    metrics = ['TR_OEff_25', 'TR_DEff_25', 'NET_eFG%', 'NET AST/TOV RATIO', 'TTL REB%', 'STOCKS/GM']
    labels = ['Offense', 'Defense', 'Shooting', 'Ball Control', 'Rebounding', 'Stocks']
    available_metrics = [m for m in metrics if m in team.index]
    available_labels = [labels[metrics.index(m)] for m in available_metrics]
    if not available_metrics:
        st.markdown(f"<div style='height:200px;text-align:center;color:white;'><p>({seed}) {team_name}</p><p>No metrics available</p></div>", unsafe_allow_html=True)
        return
    values = []
    for metric in available_metrics:
        if pd.notna(team[metric]):
            all_values = TR_df[metric].dropna()
            if len(all_values) > 0:
                mean = all_values.mean()
                std = all_values.std() if all_values.std() > 0 else 1
                z_score = (team[metric] - mean) / std
                z_score = max(min(z_score, 3), -3)
                norm_value = (z_score + 3) * (100 / 6)
                values.append(norm_value)
            else:
                values.append(50)
        else:
            values.append(50)
    strengths = []
    if len(values) >= 6:
        if values[0] > 65 and values[2] > 65:
            strengths.append("Offensive")
        if values[1] > 65 and values[5] > 65:
            strengths.append("Defensive")
        if values[3] > 65 and values[4] > 65:
            strengths.append("Fundamental")
    team_type = " & ".join(strengths) if strengths else "Balanced"
    seed_colors = {
        range(1, 5): "rgba(0, 255, 0, 0.7)",
        range(5, 9): "rgba(255, 255, 0, 0.7)",
        range(9, 13): "rgba(255, 165, 0, 0.7)",
        range(13, 17): "rgba(255, 0, 0, 0.7)"
    }
    color = next((c for r, c in seed_colors.items() if seed in r), "rgba(255, 255, 255, 0.7)")
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=available_labels,
        fill='toself',
        fillcolor=color.replace('0.7', '0.3'),
        line=dict(color=color),
        name=team_name
    ))
    bg_color = "#1E1E1E" if dark_mode else "#FFFFFF"
    text_color = "white" if dark_mode else "black"
    grid_color = "rgba(255, 255, 255, 0.1)" if dark_mode else "rgba(0, 0, 0, 0.1)"
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False,
                gridcolor=grid_color
            ),
            angularaxis=dict(
                gridcolor=grid_color,
                linecolor=grid_color
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=250,
        font=dict(color=text_color)
    )
    fig.add_annotation(
        text=f"({seed}) {team_name}",
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=14, color=text_color),
        align="center"
    )
    fig.add_annotation(
        text=f"{team_type}",
        xref="paper", yref="paper",
        x=0.5, y=0.95,
        showarrow=False,
        font=dict(size=12, color=text_color),
        align="center"
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=key)

# ----------------------------------------------------------------------------
# Treemap Function
def create_treemap(df_notnull):
    try:
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
        treemap_data = top_100_teams.copy()
        treemap_data["KP_AdjEM"] = pd.to_numeric(treemap_data["KP_AdjEM"], errors='coerce')
        treemap_data = treemap_data.dropna(subset=["KP_AdjEM"])
        if "TM_KP" not in treemap_data.columns:
            treemap_data["TM_KP"] = treemap_data["TEAM"]
        
        def hover_text_func(x):
            base = (
                f"<b>{x['TM_KP']}</b><br>"
                f"<b>KP Rank:</b> {int(x['KP_Rank'])}<br>"
                f"<b>Record:</b> {int(x['WIN_25'])}-{int(x['LOSS_25'])}<br>"
                f"<b>AdjEM:</b> {x['KP_AdjEM']:.1f}<br>"
            )
            if "OFF EFF" in x and "DEF EFF" in x:
                base += f"<b>OFF EFF:</b> {x['OFF EFF']:.1f}<br><b>DEF EFF:</b> {x['DEF EFF']:.1f}<br>"
            if "AVG MARGIN" in x:
                base += f"<b>AVG MARGIN:</b> {x['AVG MARGIN']:.1f}<br>"
            if "TS%" in x and "OPP TS%" in x:
                base += f"<b>TS%:</b> {x['TS%']:.1f}% | <b>OPP TS%:</b> {x['OPP TS%']:.1f}%<br>"
            if "AST/TO%" in x:
                base += f"<b>AST/TO%:</b> {x['AST/TO%']:.1f}<br>"
            if "SEED_25" in x and not pd.isna(x["SEED_25"]):
                base += f"<b>Seed:</b> {int(x['SEED_25'])}"
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
            textfont=dict(size=12, family="Arial, sans-serif")
        )
        
        treemap.update_layout(
            margin=dict(l=10, r=10, t=60, b=10),
            coloraxis_colorbar=dict(
                title="AdjEM",
                thicknessmode="pixels",
                thickness=15,
                lenmode="pixels",
                len=300,
                yanchor="top",
                y=1,
                ticks="outside",
                tickfont=dict(size=12)
            ),
            template="plotly_dark",
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0.8)",
                font_size=14,
                font_family="Arial"
            )
        )
        
        return treemap
    except Exception as e:
        st.error(f"An error occurred while generating treemap: {e}")
        return None

#############################################
# --- Simulation Functions ---
#############################################

# Set up logging for simulation (suppress detailed logs in Streamlit)
sim_logger = logging.getLogger("simulation")
if sim_logger.hasHandlers():
    sim_logger.handlers.clear()
sim_logger.setLevel(logging.WARNING)
sim_handler = logging.StreamHandler()
sim_handler.setLevel(logging.WARNING)
sim_handler.setFormatter(logging.Formatter("%(message)s"))
sim_logger.addHandler(sim_handler)

# ANSI color codes for simulation rounds (used only for logging)
BLUE     = '\033[94m'
CYAN     = '\033[96m'
GREEN    = '\033[92m'
YELLOW   = '\033[93m'
MAGENTA  = '\033[95m'
RED      = '\033[91m'
RESET    = '\033[0m'

round_colors = {
    "Round of 64": BLUE,
    "Round of 32": CYAN,
    "Sweet 16": GREEN,
    "Elite 8": YELLOW,
    "Final Four": MAGENTA,
    "Championship": RED
}

#############################################
# --- Simulation Functions ---
#############################################

# (Logging setup and ANSI color codes remain as defined above)

# New function to simulate Final Four and Championship games.
def simulate_final_four_and_championship(region_champions, use_analytics=True):
    """
    Given a dictionary of 4 regional champions, simulate the Final Four and Championship.
    Returns the champion team (a dict with team details) or None if 4 champions are not available.
    """
    # Convert the region champions dict into a list; order is arbitrary.
    champs = list(region_champions.values())
    if len(champs) != 4:
        return None

    # Semifinal round: Pair champions arbitrarily (first two and last two).
    semi1_prob = calculate_win_probability(champs[0], champs[1])
    semi1_winner = champs[0] if random.random() < semi1_prob else champs[1]

    semi2_prob = calculate_win_probability(champs[2], champs[3])
    semi2_winner = champs[2] if random.random() < semi2_prob else champs[3]

    # Championship game:
    final_prob = calculate_win_probability(semi1_winner, semi2_winner)
    champion = semi1_winner if random.random() < final_prob else semi2_winner

    return champion

def prepare_tournament_data():
    """
    Prepare tournament data using the main dataframe (df_main).
    Returns a dictionary with:
      - 'tournament_teams': DataFrame of teams with valid seeds and required metrics,
      - 'region_names': list of region names,
      - 'region_teams': dict mapping each region to a list of teams (as dictionaries),
    and also sets the global TR_df for visualization.
    """
    global TR_df
    TR_df = df_main.copy()
    
    required_cols = ['SEED_25', 'REGION_25', 'KP_Rank', 'KP_AdjEM', 'OFF EFF', 'DEF EFF']
    for col in required_cols:
        if col not in TR_df.columns:
            sim_logger.warning(f"Missing required column: {col}")
            return None

    tournament_teams = TR_df.dropna(subset=required_cols).copy()
    tournament_teams['SEED_25'] = tournament_teams['SEED_25'].astype(int)
    tournament_teams = tournament_teams[(tournament_teams['SEED_25'] >= 1) & (tournament_teams['SEED_25'] <= 16)]

    if 'TM_KP' not in tournament_teams.columns:
        if 'TM_TR' in tournament_teams.columns:
            tournament_teams['TM_KP'] = tournament_teams['TM_TR']
        else:
            sim_logger.warning("No team name column found (TM_KP or TM_TR)")
            return None

    # Add bonus values
    tournament_teams['TOURNEY_SUCCESS'] = 0.0
    for team in ["Duke", "Kentucky", "Kansas", "North Carolina", "Gonzaga", "Michigan St."]:
        if team in tournament_teams['TM_KP'].values:
            tournament_teams.loc[tournament_teams['TM_KP'] == team, 'TOURNEY_SUCCESS'] = 2.0

    tournament_teams['TOURNEY_EXPERIENCE'] = 0.0
    if 'SEED_23' in tournament_teams.columns:
        tournament_teams.loc[tournament_teams['SEED_23'].notna() & (tournament_teams['SEED_23'] <= 16), 'TOURNEY_EXPERIENCE'] = 1.0
        tournament_teams.loc[tournament_teams['SEED_23'].notna() & (tournament_teams['SEED_23'] <= 4), 'TOURNEY_EXPERIENCE'] = 2.0

    region_names = tournament_teams['REGION_25'].unique().tolist()

    region_teams = {}
    for reg in region_names:
        df_reg = tournament_teams[tournament_teams['REGION_25'] == reg].sort_values('SEED_25')
        teams_list = df_reg.apply(lambda row: {
            'Team': row['TM_KP'],
            'Seed': int(row['SEED_25']),
            'KP_Rank': row['KP_Rank'],
            'KP_AdjEM': row['KP_AdjEM'],
            'OFF EFF': row['OFF EFF'],
            'DEF EFF': row['DEF EFF'],
            'KP_AdjO': row.get('KP_AdjO', 0),
            'KP_AdjD': row.get('KP_AdjD', 0),
            'TOURNEY_SUCCESS': row.get('TOURNEY_SUCCESS', 0),
            'TOURNEY_EXPERIENCE': row.get('TOURNEY_EXPERIENCE', 0),
            'WIN_PCT': row.get('WIN% ALL GM', 0.5),
            'CLOSE_GAME_PCT': row.get('WIN% CLOSE GM', 0.5),
            'SOS': row.get('KP_SOS_AdjEM', 0),
            'Region': row['REGION_25']
        }, axis=1).tolist()
        if len(teams_list) > 16:
            teams_list = teams_list[:16]
        region_teams[reg] = teams_list

    return {
        'tournament_teams': tournament_teams,
        'region_names': region_names,
        'region_teams': region_teams
    }

def calculate_win_probability(team1, team2):
    """
    Calculate the win probability for team1 over team2 based on efficiency metrics and tournament bonuses.
    """
    team1_off = float(team1.get('OFF EFF', 1.0)) if not pd.isna(team1.get('OFF EFF', 1.0)) else 1.0
    team1_def = float(team1.get('DEF EFF', 1.0)) if not pd.isna(team1.get('DEF EFF', 1.0)) else 1.0
    team2_off = float(team2.get('OFF EFF', 1.0)) if not pd.isna(team2.get('OFF EFF', 1.0)) else 1.0
    team2_def = float(team2.get('DEF EFF', 1.0)) if not pd.isna(team2.get('DEF EFF', 1.0)) else 1.0

    kp_diff = float(team1['KP_AdjEM']) - float(team2['KP_AdjEM'])
    matchup_adv = (team1_off - team2_def) - (team2_off - team1_def)
    exp_diff = (float(team1.get('TOURNEY_EXPERIENCE', 0)) - float(team2.get('TOURNEY_EXPERIENCE', 0))) + \
               (float(team1.get('TOURNEY_SUCCESS', 0)) - float(team2.get('TOURNEY_SUCCESS', 0)))
    weight_kp = 1.0
    weight_matchup = 0.5
    weight_exp = 0.2

    combined_factor = (weight_kp * kp_diff +
                       weight_matchup * matchup_adv +
                       weight_exp * exp_diff)
    base_prob = 1 / (1 + np.exp(-0.1 * combined_factor))

    seed_diff = team2['Seed'] - team1['Seed']
    if seed_diff > 0:
        if team1['Seed'] <= 4 and team2['Seed'] >= 12:
            upset_factor = 0.05
            base_prob = max(0.65, min(0.95, base_prob - upset_factor))
    win_prob = max(0.05, min(0.95, base_prob))
    return win_prob

def generate_bracket_round(teams, round_num, region, use_analytics=True):
    """
    Given a list of teams for a round, simulate games and return the winners.
    """
    winners = []
    if round_num == 1:
        pairings = [(0, 15), (7, 8), (4, 11), (3, 12), (5, 10), (2, 13), (6, 9), (1, 14)]
    else:
        pairings = [(i, i+1) for i in range(0, len(teams), 2)]
    
    for i, j in pairings:
        if i < len(teams) and j < len(teams):
            teamA = teams[i]
            teamB = teams[j]
            if use_analytics:
                pA = calculate_win_probability(teamA, teamB)
            else:
                diff = teamA['KP_AdjEM'] - teamB['KP_AdjEM']
                pA = 1 / (1 + np.exp(-diff/10))
            winner = teamA if random.random() < pA else teamB
            winner = winner.copy()
            winner['win_prob'] = pA if winner == teamA else (1 - pA)
            winners.append(winner)
    return winners

def simulate_region_bracket(teams, region_name, use_analytics=True):
    """
    Simulate a single region’s bracket given a list of teams.
    Returns a tuple (rounds, all_games) where:
      - rounds: a dict mapping round number to list of teams remaining,
      - all_games: a list of dicts, each representing a game.
    """
    rounds = {}
    current_round_teams = teams
    num_rounds = int(math.log(len(teams), 2)) if teams else 0
    all_games = []
    
    for r in range(1, num_rounds + 1):
        rounds[r] = current_round_teams
        winners = generate_bracket_round(current_round_teams, r, region_name, use_analytics)
        for i, winner in enumerate(winners):
            # For round 1, use the predefined pairing order.
            if r == 1:
                pairing = [(0, 15), (7, 8), (4, 11), (3, 12), (5, 10), (2, 13), (6, 9), (1, 14)]
                matchup_idx = pairing[i]
                if matchup_idx[0] < len(current_round_teams) and matchup_idx[1] < len(current_round_teams):
                    team1 = current_round_teams[matchup_idx[0]]
                    team2 = current_round_teams[matchup_idx[1]]
                else:
                    continue
            else:
                if i*2 < len(current_round_teams) and i*2+1 < len(current_round_teams):
                    team1 = current_round_teams[i*2]
                    team2 = current_round_teams[i*2+1]
                else:
                    continue
            
            game_info = {
                'round': r,
                'round_name': {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16", 4: "Elite 8"}.get(r, f"Round {r}"),
                'region': region_name,
                'team1': team1['Team'],
                'seed1': team1['Seed'],
                'team2': team2['Team'],
                'seed2': team2['Seed'],
                'winner': winner['Team'],
                'winner_seed': winner['Seed'],
                'win_prob': winner.get('win_prob', 0.5)
            }
            all_games.append(game_info)
        current_round_teams = winners
    if num_rounds > 0:
        rounds[num_rounds + 1] = current_round_teams
    return rounds, all_games

def run_simulation(use_analytics=True, simulations=1):
    """
    Run a specified number of complete tournament simulations.
    For each simulation, simulate each region’s bracket (only if 16 teams are available per region),
    then simulate Final Four and Championship if 4 regional champions are available.
    
    Returns a list of simulation result dictionaries.
    Each simulation result contains:
      - 'region_results': the bracket results for each region,
      - 'region_champions': the champion from each region,
      - 'champion': the overall tournament champion,
      - 'all_games': list of all game details,
      - 'simulation_number': simulation run number.
    """
    tournament_data = prepare_tournament_data()
    if not tournament_data:
        sim_logger.error("Failed to prepare tournament data")
        return []

    region_names = tournament_data['region_names']
    region_teams = tournament_data['region_teams']
    all_results = []

    for sim in range(simulations):
        region_results = {}
        region_champions = {}
        all_games = []
        valid_regions = [reg for reg in region_names if reg in region_teams and len(region_teams[reg]) >= 16]
        if len(valid_regions) < 4:
            sim_logger.warning(f"Not enough valid regions for simulation. Found {len(valid_regions)} regions.")
            continue

        for reg in valid_regions:
            teams = region_teams[reg][:16]  # use exactly 16 teams per region
            rounds, games = simulate_region_bracket(teams, reg, use_analytics)
            region_results[reg] = rounds
            final_round = max(rounds.keys(), default=0)
            if final_round and rounds[final_round]:
                region_champions[reg] = rounds[final_round][0]
            all_games.extend(games)
        
        # Only simulate Final Four if exactly 4 regional champions are available.
        if len(region_champions) == 4:
            champion = simulate_final_four_and_championship(region_champions, use_analytics)
        else:
            champion = None

        sim_result = {
            'region_results': region_results,
            'region_champions': region_champions,
            'champion': champion,
            'all_games': all_games,
            'simulation_number': sim + 1
        }
        all_results.append(sim_result)

    return all_results

def display_simulation_results(sim_results, st_container):
    """
    Display detailed results for a single simulation run, with:
      - Chronological round sorting (64→32→16→8→4→2),
      - Numeric seed columns for color-scaling,
      - Color-scaled Win Prob column,
      - "UPSET" highlighting,
      - Styled output (HTML) matching advanced tables in your app.
    """
    import seaborn as sns  # ensure you have "import seaborn as sns" at top
    if not sim_results or len(sim_results) == 0:
        st_container.warning("No simulation results available.")
        return

    # We'll only show results from the first run in sim_results
    sim_result = sim_results[0]

    # If we have a champion, show that top-level success message
    champion = sim_result.get('champion', {})
    if champion:
        champion_team = champion.get('Team', 'Unknown')
        champion_seed = champion.get('Seed', 'N/A')
        st_container.success(f"🏆 Tournament Champion: {champion_team} (Seed {champion_seed})")

    all_games = sim_result.get('all_games', [])
    if not all_games:
        st_container.warning("No game data available for this simulation.")
        return

    # 1) Custom sort order for rounds
    round_order_map = {
        "Round of 64": 1,
        "Round of 32": 2,
        "Sweet 16":    3,
        "Elite 8":     4,
        "Final Four":  5,
        "Championship": 6
    }

    # 2) Convert each game dict to a row in a DataFrame
    df_all = pd.DataFrame([
        {
            # For sorting by round
            "Round_Sort":  round_order_map.get(g["round_name"], 99),
            "Round":       g["round_name"],
            # We'll keep original matchup strings
            "Matchup":     f"{g['team1']} ({g['seed1']}) vs {g['team2']} ({g['seed2']})",
            "Winner":      g["winner"],
            # store numeric seeds in separate columns for advanced styling
            "Seed1":       g["seed1"],
            "Seed2":       g["seed2"],
            "WinnerSeed":  g["winner_seed"],
            # Win prob is numeric for now; we format as % later
            "WinProb":     g["win_prob"],
            # Upset indicator
            "Upset": "⚠️ UPSET" if (
                (g['seed1'] < g['seed2'] and g['winner'] == g['team2'])
                or (g['seed2'] < g['seed1'] and g['winner'] == g['team1'])
            ) else "",
            "Region": g["region"]
        }
        for g in all_games
    ])

    # 3) Sort by round in ascending order
    df_all = df_all.sort_values(by="Round_Sort").reset_index(drop=True)
    df_all.drop(columns=["Round_Sort"], inplace=True)

    # 4) Group by round, then style each group
    #    (Alternatively, you can show one big table if you prefer.)
    unique_rounds = df_all["Round"].unique()
    
    # We'll define a helper style function to highlight upset cells
    def highlight_upset_cell(val):
        """Apply style if val contains 'UPSET'."""
        if "UPSET" in str(val):
            return "background-color: #FFCC99; font-weight: bold;"  # orange-ish highlight
        return ""
    
    for rname in unique_rounds:
        subset = df_all[df_all["Round"] == rname].copy()

        # Convert numeric columns to something easily stylable
        # e.g. WinProb as 0..1 => 0..100%
        subset["WinProb"] = subset["WinProb"].apply(lambda x: x if pd.isna(x) else x * 100.0)

        # We'll rename columns for clarity in the final display
        subset.rename(columns={
            "Round":       "Round",
            "Matchup":     "Matchup",
            "Winner":      "Winner",
            "Seed1":       "Team1 Seed",
            "Seed2":       "Team2 Seed",
            "WinnerSeed":  "W Seed",
            "WinProb":     "WinProb(%)",
            "Upset":       "Upset",
            "Region":      "Region"
        }, inplace=True)

        # Use a Pandas Styler to color-scale seeds & WinProb
        # E.g. seeds: 1 => better => green, 16 => worse => red, so we'll do "RdYlGn_r"
        # WinProb: 0 => red, 100 => green => "RdYlGn"
        # or we can do 'Greens' for WinProb—taste is up to you.
        styled_df = subset.style.format({
            "WinProb(%)": "{:.1f}%",  # one decimal
            "Team1 Seed": "{:.0f}",
            "Team2 Seed": "{:.0f}",
            "W Seed":     "{:.0f}"
        })

        # We apply background_gradient for seeds and WinProb
        seed_cmap  = sns.color_palette("RdYlGn_r", as_cmap=True)
        prob_cmap  = sns.color_palette("RdYlGn", as_cmap=True)
        # If you want max seed = 16 or 17 just in case, you can define vmin=1, vmax=16 for seeds
        styled_df = (
            styled_df
            .background_gradient(cmap=seed_cmap, subset=["Team1 Seed", "Team2 Seed", "W Seed"], vmin=1, vmax=16)
            .background_gradient(cmap=prob_cmap, subset=["WinProb(%)"], vmin=0, vmax=100)
            .applymap(highlight_upset_cell, subset=["Upset"])  # highlight upset col
            .set_caption(rname)  # Title above each table
        )

        # Apply some advanced styling as in your other advanced tables
        advanced_styles = [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#0360CE"),
                    ("color", "white"),
                    ("text-align", "center"),
                    ("border", "1px solid #222"),
                    ("font-weight", "bold"),
                    ("font-size", "13px")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "center"),
                    ("border", "1px solid #444"),
                    ("padding", "8px"),
                ]
            },
            # We can also style the caption
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-size", "15px"),
                    ("color", "#fff"),
                    ("background-color", "#000"),
                    ("padding", "5px"),
                    ("font-weight", "bold"),
                    ("text-align", "center")
                ]
            }
        ]
        styled_df = styled_df.set_table_styles(advanced_styles)

        # 5) Display final table under a subheader
        st_container.subheader(rname)
        st_container.markdown(styled_df.to_html(escape=False), unsafe_allow_html=True)


def run_tournament_simulation(num_simulations=100, use_analytics=True):
    """
    Run multiple tournament simulations and aggregate the results.
    
    Returns a dictionary with:
      - 'champion_probabilities': DataFrame of championship counts and probabilities,
      - 'region_probabilities': DataFrame of regional champion frequencies,
      - 'upset_pct_aggregated': Series of upset percentages by round,
      - 'total_simulations': total number of simulation runs.
    
    Only simulation results that produced a champion are aggregated.
    """
    all_sim_results = run_simulation(use_analytics=use_analytics, simulations=num_simulations)
    # Filter out simulation runs that did not produce a champion.
    valid_sim_results = [sim for sim in all_sim_results if sim.get('champion') is not None]
    total_valid = len(valid_sim_results)
    if total_valid == 0:
        return {
            'error': "No simulation results produced a champion.",
            'champion_probabilities': pd.DataFrame(),
            'upset_pct_aggregated': pd.Series(),
            'total_simulations': 0
        }

    champions = {}
    for sim in valid_sim_results:
        champ = sim.get('champion')
        champ_team = champ.get('Team', 'Unknown')
        champ_seed = champ.get('Seed', 'N/A')
        if champ_team not in champions:
            champions[champ_team] = {'team': champ_team, 'seed': champ_seed, 'count': 0}
        champions[champ_team]['count'] += 1

    champ_data = []
    for team, data in champions.items():
        champ_data.append({
            'Team': data['team'],
            'Seed': data['seed'],
            'Championship_Count': data['count'],
            'Championship_Probability': data['count'] / total_valid
        })
    champ_df = pd.DataFrame(champ_data).sort_values('Championship_Count', ascending=False).reset_index(drop=True)

    # Regional champions aggregation.
    region_champions = {}
    for sim in valid_sim_results:
        for region, champ in sim.get('region_champions', {}).items():
            if region not in region_champions:
                region_champions[region] = {}
            team = champ.get('Team', 'Unknown')
            region_champions[region][team] = region_champions[region].get(team, 0) + 1

    region_data = []
    for region, teams in region_champions.items():
        for team, count in teams.items():
            region_data.append({
                'Region': region,
                'Team': team,
                'Count': count,
                'Probability': count / num_simulations
            })
    region_df = pd.DataFrame(region_data).sort_values(['Region', 'Count'], ascending=[True, False])

    # Upset analysis: count upsets by round.
    all_games = []
    for sim in all_sim_results:
        all_games.extend(sim.get('all_games', []))
    upsets_by_round = {}
    games_by_round = {}
    for game in all_games:
        rnd = game.get('round_name', 'Unknown')
        games_by_round[rnd] = games_by_round.get(rnd, 0) + 1
        seed1 = game.get('seed1', 0)
        seed2 = game.get('seed2', 0)
        winner = game.get('winner', '')
        team1 = game.get('team1', '')
        team2 = game.get('team2', '')
        is_upset = (seed1 < seed2 and winner == team2) or (seed2 < seed1 and winner == team1)
        if is_upset:
            upsets_by_round[rnd] = upsets_by_round.get(rnd, 0) + 1
    upset_pct = {rnd: (upsets_by_round.get(rnd, 0) / games_by_round[rnd]) * 100 
                 for rnd in games_by_round if games_by_round[rnd] > 0}
    upset_pct_series = pd.Series(upset_pct)

    return {
        'champion_probabilities': champ_df,
        'region_probabilities': region_df,
        'upset_pct_aggregated': upset_pct_series,
        'total_simulations': num_simulations
    }

def visualize_aggregated_results(aggregated_analysis):
    """
    Create two distinct Plotly charts that summarize the bracket simulations:
      1) A horizontal bar chart for championship win probabilities (top 10 teams).
      2) A bar chart for upset percentages by round.
    Returns (fig_champ, fig_upsets) as plotly Figure objects.
    """

    # --- 1) Championship Win Probabilities ---
    champ_df = aggregated_analysis.get('champion_probabilities')
    fig_champ = None
    if champ_df is not None and not champ_df.empty:
        # Keep only top 10
        top_teams = champ_df.head(10).copy()
        top_teams['Championship_Probability_PCT'] = top_teams['Championship_Probability'] * 100

        # Build a Plotly horizontal bar chart
        fig_champ = px.bar(
            top_teams,
            y='Team',
            x='Championship_Probability_PCT',
            orientation='h',
            color='Championship_Probability_PCT',
            color_continuous_scale='RdYlGn',
            title="Championship Win Probability (Top 10 Teams)",
            labels={'Championship_Probability_PCT': 'Win Probability (%)', 'Team': ''},
            template='plotly_dark',
            hover_data=['Seed', 'Championship_Count']  # Optional extras
        )
        # Invert y-axis for typical bar chart look (highest on top)
        fig_champ.update_yaxes(autorange="reversed")
        # Add text labels on each bar
        fig_champ.update_traces(
            texttemplate='%{x:.1f}%',
            textposition='outside'
        )
        fig_champ.update_layout(
            margin=dict(l=50, r=50, t=70, b=50),
            coloraxis_showscale=False,  # hide color bar
        )

    # --- 2) Upset Percentage by Round ---
    upset_pct = aggregated_analysis.get('upset_pct_aggregated')
    fig_upsets = None
    if upset_pct is not None and not upset_pct.empty:
        # Convert to DataFrame for easier plotting
        df_upsets = pd.DataFrame({
            'Round': upset_pct.index,
            'Upset_PCT': upset_pct.values
        })
        # Optionally impose a custom round order
        round_order = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
        df_upsets['Round'] = pd.Categorical(df_upsets['Round'], categories=round_order, ordered=True)
        df_upsets = df_upsets.sort_values('Round')

        fig_upsets = px.bar(
            df_upsets,
            x='Round',
            y='Upset_PCT',
            text='Upset_PCT',
            color='Upset_PCT',
            color_continuous_scale='RdYlGn',
            title="Upset Percentage by Tournament Round",
            labels={'Upset_PCT': 'Upset Percentage (%)'},
            template='plotly_dark',
        )
        # Format text on bars
        fig_upsets.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig_upsets.update_layout(
            margin=dict(l=50, r=50, t=70, b=50),
            yaxis_range=[0, df_upsets['Upset_PCT'].max() * 1.2],
            coloraxis_showscale=False,
        )

    return fig_champ, fig_upsets


def create_regional_prob_chart(region_df):
    """
    Create a Plotly bar chart to visualize regional champion probabilities.
    """
    if region_df is None or region_df.empty:
        return None
    
    top_teams_by_region = []
    for region in region_df['Region'].unique():
        region_data = region_df[region_df['Region'] == region].sort_values('Probability', ascending=False).head(5)
        top_teams_by_region.append(region_data)
    filtered_df = pd.concat(top_teams_by_region)
    fig = px.bar(filtered_df, x='Team', y='Probability', color='Region',
                 barmode='group', facet_col='Region', facet_col_wrap=2,
                 labels={'Probability': 'Win Probability', 'Team': 'Team'},
                 title='Regional Championship Probabilities (Top 5 Teams per Region)',
                 color_discrete_sequence=px.colors.qualitative.G10)
    fig.update_layout(legend_title_text='Region', showlegend=True,
                      template='plotly_dark', height=600,
                      margin=dict(t=80, l=50, r=50, b=100),
                      title_font=dict(size=18), xaxis_tickangle=-45)
    fig.update_yaxes(tickformat='.0%')
    if fig.layout.annotations:
        for annotation in fig.layout.annotations:
            if "=" in annotation.text:
                annotation.text = annotation.text.split("=")[1]
    for data in fig.data:
        fig.add_trace(go.Scatter(x=data.x, y=data.y,
                                 text=[f"{y:.1%}" for y in data.y],
                                 mode="text"))
    return fig


# --- App Header & Tabs ---
st.title(":primary[2025 NCAAM BASKETBALL --- MARCH MADNESS]")
st.subheader(":primary[2025 MARCH MADNESS RESEARCH HUB]")
st.subheader(":blue[🎱WELCOME RICH MANEEE🎱]")
st.caption(":primary[_Cure your bracket brain and propel yourself up the leaderboards by exploring the tabs below:_]")

tab_home, tab_radar, tab_regions, tab_team, tab_conf, tab_pred = st.tabs(["🏀 HOME",  #🌐
                                                                          "🕸️ RADAR CHARTS", #📡🧭
                                                                          "🔥 REGIONAL HEATMAPS", #🌡️📍
                                                                          "📊 TEAM METRICS", #📈📋📜📰📅
                                                                          "🏆 CONFERENCE STATS", #🏅
                                                                          "🔮 PREDICTIONS"]) #🎱❓✅❌ ⚙️

# --- Home Tab ---
with tab_home:
    st.subheader(":primary[NCAAM BASKETBALL CONFERENCE TREEMAP]", divider='grey')
    st.caption(":green[_DATA AS OF: 3/15/2025_]")
    treemap = create_treemap(df_main_notnull)
    if treemap is not None:
        st.plotly_chart(treemap, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
    
    st.subheader("🔍 :primary[TEAM SPOTLIGHT]", divider='grey') # 📊 🔢 🚩🏁🎌🏟️ 📢📣🎊🎉🕒🗣️📺📱
    selected_team = st.selectbox(
        ":green[_SELECT A TEAM:_]",
        options=[""] + sorted(df_main["TM_KP"].dropna().unique().tolist()),
        index=0
    )

    if selected_team:
        team_data = df_main[df_main["TM_KP"] == selected_team].copy()
        if not team_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {selected_team}")
                conf = team_data["CONFERENCE"].iloc[0] if "CONFERENCE" in team_data.columns else "N/A"
                record = f"{int(team_data['WIN_25'].iloc[0])}-{int(team_data['LOSS_25'].iloc[0])}" if "WIN_25" in team_data.columns and "LOSS_25" in team_data.columns else "N/A"
                seed_info = f"Seed: {int(team_data['SEED_25'].iloc[0])}" if "SEED_25" in team_data.columns and not pd.isna(team_data['SEED_25'].iloc[0]) else ""
                kp_rank = f"KenPom Rank: {int(team_data['KP_Rank'].iloc[0])}" if "KP_Rank" in team_data.columns else ""

                st.markdown(f"""
                **Conference:** {conf}  
                **Record:** {record}  
                {seed_info}  
                {kp_rank}
                """)

                # Overall performance badge (existing logic)
                if all(m in team_data.columns for m in get_default_metrics()):
                    t_avgs, t_stdevs = compute_tournament_stats(df_main)
                    perf_data = compute_performance_text(team_data.iloc[0], t_avgs, t_stdevs)
                    st.markdown(f"""
                    <div style='text-align: center; margin: 20px 0;'>
                        <span class='{perf_data["class"]}' style='font-size: 18px; padding: 8px 16px;'>
                            Overall Rating: {perf_data["text"]}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                # "Interpretive Insights" block - provides a short textual breakdown of how the team compares to NCAA average
                radar_metrics = get_default_metrics()  # e.g. ['AVG MARGIN','KP_AdjEM','OFF EFF','DEF EFF','AST/TO%','STOCKS-TOV/GM']
                existing_metrics = [m for m in radar_metrics if m in team_data.columns]
                if existing_metrics:
                    with st.expander("📊 Interpretive Insights"):
                        insights = []
                        # For each metric, compare to NCAA average & append text
                        for metric in existing_metrics:
                            mean_val = t_avgs[metric]
                            std_val = max(t_stdevs[metric], 1e-6)  # avoid zero-div
                            team_val = team_data.iloc[0][metric]
                            z = (team_val - mean_val) / std_val
                            # Flip sign for metrics like 'DEF EFF' if you want lower=better
                            if metric in ["DEF EFF", "TO/GM"]:
                                z = -z

                            # Build short snippet about each metric
                            if abs(z) < 0.3:
                                insights.append(f"**{metric}** is near the NCAA average.")
                            elif z >= 1.0:
                                insights.append(f"**{metric}** is a clear strength (top tier).")
                            elif 0.3 <= z < 1.0:
                                insights.append(f"**{metric}** is somewhat above average.")
                            elif -1.0 < z <= -0.3:
                                insights.append(f"**{metric}** is somewhat below average.")
                            else:
                                insights.append(f"**{metric}** is a notable weakness (bottom tier).")

                        # Display insights as bullet points
                        st.markdown("**Team Metric Highlights:**")
                        for line in insights:
                            st.write(f"- {line}")
                        
            with col2:
                # Radar chart of the selected team
                key_metrics = ["KP_AdjEM", "OFF EFF", "DEF EFF", "TS%", "OPP TS%", "AST/TO%", "STOCKS/GM", "AVG MARGIN"]
                available_metrics = [m for m in key_metrics if m in team_data.columns]
                # Reuse your existing create_radar_chart function if desired:
                radar_fig = create_radar_chart([selected_team], df_main)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
            
            with st.expander("View All Team Metrics"):
                detailed_metrics = [
                    "KP_Rank", "KP_AdjEM", "KP_SOS_AdjEM", 
                    "OFF EFF", "DEF EFF", "WIN% ALL GM", "WIN% CLOSE GM",
                    "PTS/GM", "OPP PTS/GM", "AVG MARGIN",
                    "eFG%", "OPP eFG%", "TS%", "OPP TS%", 
                    "AST/GM", "TO/GM", "AST/TO%", 
                    "OFF REB/GM", "DEF REB/GM", "BLKS/GM", "STL/GM", 
                    "STOCKS/GM", "STOCKS-TOV/GM"
                ]
                available_detailed = [m for m in detailed_metrics if m in team_data.columns]
                detail_df = team_data[available_detailed].T.reset_index()
                detail_df.columns = ["Metric", "Value"]
                detail_df["Value"] = detail_df.apply(
                    lambda x: f"{x['Value']:.1f}" if isinstance(x['Value'], float) else x['Value'],
                    axis=1
                )
                st.table(detail_df)
        else:
            st.warning("No data available for the selected team.")
    
    
    # Style for the index (RANK)
    index_style = {
        'selector': '.row_heading.level0',  # Target the index cells
        'props': [
            ('background-color', '#0360CE'),
            ('color', 'white'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('border', '1px solid #000000'),
        ]
    }
    
    # General cell style
    cell_style = {
        'selector': 'tbody td',  # Target data cells
        'props': [
            ('text-align', 'center'),
            ('border', '1px solid #ddd'),  # Lighter border for data cells
            ('padding', '5px 10px'),
        ]
    }

    # Style to group related statistics (AdjEM range)
    adj_em_group_style = {
        'selector': 'th.col_heading.level0.col2, th.col_heading.level0.col3, th.col_heading.level0.col4',  # Target AdjEM columns
        'props': [
            ('border-right', '3px solid #888'),  # Thicker border to group
        ]
    }

    detailed_table_styles = [header, index_style, cell_style, adj_em_group_style]

    if "CONFERENCE" in df_main.columns:
        conf_counts = df_main["CONFERENCE"].value_counts().reset_index()
        conf_counts.columns = ["CONFERENCE", "# TEAMS"]

        if "KP_AdjEM" in df_main.columns: 
            conf_stats = df_main.groupby("CONFERENCE").agg( # Aggregate multiple stats at once
                {
                    "KP_AdjEM": ["count", "max", "mean", "min"],
                    "SEED_25": ["count", "mean"],
                    "NET_25": "mean",
                    #"BPI_25": "mean", "BPI_Rank": "mean",
                    #"TR_OEff_25": "mean", #"TR_DEff_25": "mean",
                    "WIN% ALL GM": "mean", #"WIN% CLOSE GM": "mean",
                    "AVG MARGIN": "mean",
                    "eFG%": "mean", #"TS%": "mean",
                    "AST/TO%": "mean", #"NET AST/TOV RATIO": "mean",
                    "STOCKS/GM": "mean", "STOCKS-TOV/GM": "mean",
                }
            ).reset_index()

            # Flatten the multi-level column index
            conf_stats.columns = [
                "CONFERENCE",
                "# TEAMS", "MAX AdjEM", "MEAN AdjEM", "MIN AdjEM",
                "# BIDS", "MEAN SEED_25", "MEAN NET_25",
                #"AVG TR_OEff_25", "AVG TR_DEff_25",
                "MEAN WIN %", "MEAN AVG MARGIN",
                "MEAN eFG%",
                "MEAN AST/TO%", #"NET AST/TOV RATIO",
                "MEAN STOCKS/GM", "MEAN STOCKS-TOV/GM", 
            ]

            conf_stats = conf_stats.sort_values("MEAN AdjEM", ascending=False)

            # --- Apply normal index ---
            conf_stats = conf_stats.reset_index(drop=True)
            conf_stats.index = conf_stats.index + 1  # Start index at 1
            conf_stats.index.name = "RANK"

            st.subheader(":primary[NCAAM BASKETBALL CONFERENCE POWER RANKINGS]", divider='grey')
            with st.expander("*About Conference Power Rankings:*"):
                st.markdown("""
                    Simple-average rollup of each conference:
                    - **MEAN AdjEM**: Average KenPom Adjusted Efficiency Margin within conference
                    - **MAX/MIN AdjEM**: Range of AdjEM values among teams within conference
                    - **AVG SEED_25**: Average tournament seed (lower is better)
                    - **AVG NET_25**: Average NET ranking (lower is better)
                    
                    """)

# - **AVG TR_OEff_25 / AVG TR_DEff_25**:  Average Torvik Offensive/Defensive Efficiency

        # Apply logo and styling *before* converting to HTML
        conf_stats["CONFERENCE"] = conf_stats["CONFERENCE"].apply(get_conf_logo_html)

        styled_conf_stats = (
            conf_stats.style
            .format({
                "# TEAMS": "{:.0f}",
                "# BIDS": "{:.0f}",
                "MEAN AdjEM": "{:.2f}",
                "MIN AdjEM": "{:.2f}",
                "MAX AdjEM": "{:.2f}",
                "MEAN SEED_25": "{:.1f}",
                "MEAN NET_25": "{:.1f}",

                #"AVG TR_OEff_25": "{:.1f}",
                #"AVG TR_DEff_25": "{:.1f}",
                "MEAN WIN %": "{:.1f}",
                "MEAN AVG MARGIN": "{:.1f}",
                "MEAN eFG%": "{:.1f}",
                "MEAN AST/TO%": "{:.1f}",
                #"NET AST/TOV RATIO": "{:.1f}",
                "MEAN STOCKS/GM": "{:.1f}",
                #"MEAN STOCKS-TOV/GM": "{:.1f}",
            })
            .background_gradient(cmap="RdYlGn", subset=[
                "# TEAMS", "# BIDS", 
                "MEAN AdjEM", "MIN AdjEM", "MAX AdjEM",
                #"AVG TR_OEff_25",
                "MEAN WIN %", "MEAN AVG MARGIN",
                "MEAN eFG%",
                "MEAN AST/TO%", "MEAN STOCKS/GM", #"MEAN STOCKS-TOV/GM",
                
            ])
            .background_gradient(cmap="RdYlGn_r", subset=["MEAN SEED_25", "MEAN NET_25",
                                                          #"AVG TR_DEff_25",
                                                          ])
            .set_table_styles(detailed_table_styles)
        )

        st.markdown(styled_conf_stats.to_html(escape=False), unsafe_allow_html=True)


# --- Radar Charts Tab ---
with tab_radar:
    st.header("REGIONAL RADAR CHARTS")
    create_region_seeding_radar_grid(df_main) #, region_teams
    with st.expander("*About Radar Grid:*"):
        st.markdown("""
        **Each row** represents seeds 1 through 16.<br>
        **Each column** represents one of the four major regions (East, West, South, Midwest).<br>
        Each subplot compares the team to the national average (red) and their conference average (green).<br>
        - Radial scale 0-10, where 5 is NCAA average.
        - Values above 5 are better than average, below 5 are worse.
        """, unsafe_allow_html=True)

# with tab_radar:
#     st.header("TEAM RADAR CHARTS")
#     radar_metrics = get_default_metrics()
#     available_radar_metrics = [m for m in radar_metrics if m in df_main.columns]
#     if len(available_radar_metrics) < 3:
#         st.warning(f"Not enough radar metrics available. Need at least 4: {', '.join(radar_metrics)}")
#     else:
#         if "TM_KP" in df_main.columns:
#             all_teams = sorted(df_main["TM_KP"].dropna().unique().tolist())
#             default_teams = ['Duke', 'Kansas', 'Auburn', 'Houston']
#             if "KP_AdjEM" in df_main.columns:
#                 top_teams = df_main.sort_values("KP_AdjEM", ascending=False).head(4)
#                 if "TM_KP" in top_teams.columns:
#                     default_teams = top_teams["TM_KP"].tolist()
#             if not default_teams and all_teams:
#                 default_teams = all_teams[:4]
#             selected_teams = st.multiselect(
#                 "Select Teams to Compare:",
#                 options=all_teams,
#                 default=default_teams
#             )
#             if selected_teams:
#                 radar_fig = create_radar_chart(selected_teams, df_main)
#                 if radar_fig:
#                     st.plotly_chart(radar_fig, use_container_width=True)
#                 else:
#                     st.warning("Failed to display radar chart(s) for selected teams.")
#             else:
#                 st.info("Please select at least one team to display radar charts.")
#             with st.expander("About Radar Charts:"):
#                 st.markdown("""
#                     Radar charts visualize team performance across 8 key metrics, compared to:
#                     - NCAAM Average (red dashed line)
#                     - Conference Average (green dotted line)
#                     Each metric is scaled, where 5 == NCAAM average.
#                     Values >5 are better; values <5 are worse.
#                     Overall performance rating is derived from the average z-score across all metrics.
#                 """)
#         else:
#             st.warning("Team names not available in dataset.")

# --- Regional Heatmaps Tab ---
with tab_regions:
    st.header("BRACKET ANALYSIS")
    st.write("REGIONAL HEATFRAMES (W, X, Y, Z)")
    df_heat = df_main.copy()
    numeric_cols_heat = df_heat.select_dtypes(include=np.number).columns
    mean_series = df_heat.mean(numeric_only=True)
    mean_series = mean_series.reindex(df_heat.columns, fill_value=np.nan)
    df_heat.loc["TOURNEY AVG"] = mean_series
    df_heat_T = df_heat.T
    #df_heat_T = df_heat_T[core_cols]

    east_teams_2025 = [
        "Duke", "Alabama", "Iowa St.", "Maryland",
        "Memphis", "Mississippi", "Saint Mary's", "Mississippi St.",
        "Baylor", "Vanderbilt", "North Carolina", "Colorado St.",
        "Grand Canyon", "Lipscomb", "Robert Morris", "American",
    ]
    west_teams_2025 = [
        "Auburn", "St. John's", "Texas Tech", "Texas A&M",
        "Michigan", "Missouri", "UCLA", "Gonzaga",
        "Georgia", "Utah St.", "Drake", "UC San Diego",
        "Yale", "UNC Wilmington", "Nebraska Omaha", "Alabama St.",
    ]
    south_teams_2025 = [
        "Florida", "Michigan St.", "Kentucky", "Arizona",
        "Oregon", "Illinois", "Marquette", "Connecticut",
        "Oklahoma", "New Mexico", "Texas", "Liberty",
        "Akron", "Troy", "Bryant", "Norfolk St.",
    ]
    midwest_teams_2025 = [
        "Houston", "Tennessee", "Wisconsin", "Purdue",
        "Clemson", "BYU", "Kansas", "Louisville",
        "Creighton", "Arkansas", "VCU", "McNeese",
        "High Point", "Montana", "Wofford", "SIU Edwardsville",
    ]
    regions = {
        "W Region": east_teams_2025,
        "X Region": midwest_teams_2025,
        "Y Region": south_teams_2025,
        "Z Region": west_teams_2025
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

    color_map_dict = {
        "KP_Rank": "RdBu_r",
        "WIN_25": "RdBu",
        "LOSS_25": "RdBu_r",
        "KP_AdjEM": "RdBu",
        "KP_SOS_AdjEM": "RdBu",
        "OFF EFF": "RdBu",
        "DEF EFF": "RdBu_r",
        "AVG MARGIN": "RdBu",
        "TS%": "RdBu",
        "OPP TS%": "RdBu_r",
        "AST/TO%": "RdBu",
        "STOCKS/GM": "RdBu",

    }

    # Advanced table styling to match TEAM METRICS tab
    advanced_table_styles = [
        {'selector': 'table',
         'props': [('border-collapse', 'collapse'),
                   ('border', '2px solid #222'),
                   ('border-radius', '8px'),
                   ('overflow', 'hidden'),
                   ('box-shadow', '0 4px 12px rgba(0, 0, 0, 0.1)')]},
        {'selector': 'th',
         'props': [('background-color', '#0360CE'),
                   ('color', 'white'),
                   ('text-align', 'center'),
                   ('padding', '8px 10px'),
                   ('border', '1px solid #222'),
                   ('font-weight', 'bold'),
                   ('font-size', '13px')]},
        {'selector': 'td',
         'props': [('text-align', 'center'),
                   ('padding', '5px 10px'),
                   ('border', '1px solid #ddd')]}
    ]

    index_style = {
        'selector': '.row_heading.level0',
        'props': [
            ('background-color', '#0360CE'),
            ('color', 'white'),
            ('text-align', 'left'),
            ('font-weight', 'bold'),
            ('border-bottom', '2px solid #000000'),
            ('border-right', '1px solid #000000'),
        ]
    }
    cell_style = {
        'selector': 'tbody td',
        'props': [
            ('text-align', 'center'),
            ('border', '1px solid #ddd'),
            ('padding', '5px 10px'),
        ]
    }

    # Loop through each region
    for region_name, team_list in regions.items():
        teams_found = [tm for tm in team_list if tm in df_heat_T.columns]
        if teams_found:
            region_df = df_heat_T[teams_found].copy()
            st.subheader(region_name)
            region_styler = region_df.style.format(safe_format)

            # Apply advanced and consistent row-wise color scaling
            for row_label, cmap in color_map_dict.items():
                if row_label in region_df.index:
                    # Use global min/max from df_heat (before transpose) for consistent scaling
                    vmin = df_heat[row_label].min()
                    vmax = df_heat[row_label].max()
                    region_styler = region_styler.background_gradient(
                        cmap=cmap,
                        subset=pd.IndexSlice[[row_label], :],
                        vmin=vmin,
                        vmax=vmax
                    )
            # Combine the advanced table styles with your existing styles
            region_styler = region_styler.set_table_styles(advanced_table_styles + [index_style, cell_style, header])
            st.markdown(region_styler.to_html(escape=False), unsafe_allow_html=True)
        else:
            st.info(f"No data available for {region_name}.")

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
        for conf_val in conf_group.index:
            teams = df_main[df_main["CONFERENCE"] == conf_val]
            fig_conf.add_trace(go.Scatter(
                x=teams[conf_metric], y=[conf_val] * len(teams),
                mode="markers", marker=dict(color="white", size=6, opacity=0.7),
                name=f"{conf_val} Teams"
            ))
        fig_conf.update_layout(showlegend=False)
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.error("Conference data is not available.")
    with st.expander("*About Conference Comparisons:*"):
        st.markdown("""
        **Conference Comparison Glossary:**
        - **Avg AdjEM**: Average Adjusted Efficiency Margin.
        - **Conference**: Grouping of teams by their athletic conferences.
        - Metrics intended to afford insight into relative conference strength.
        """)

# --- Team Metrics Comparison Tab ---
with tab_team:
    st.header("🏀 TEAM METRICS COMPARISON")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if "TM_KP" in df_main.columns:
        all_teams = sorted(df_main["TM_KP"].dropna().unique().tolist())
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_teams = st.multiselect(
                "👉 SELECT TEAMS TO COMPARE:",
                options=all_teams,
                default=['Duke', 'Kansas', 'Auburn', 'Houston',
                         'Tennessee', 'Alabama', 'Michigan St.',  'Iowa St.', 
                         #'Texas Tech', 'Florida',
                         ]
            )
        with col2:
            view_option = st.radio(
                "PERSPECTIVE:",
                options=["HEATMAP", "RADAR"],
                index=0,
                horizontal=True
            )
        if selected_teams:
            selected_df = df_main[df_main["TM_KP"].isin(selected_teams)].copy()
            if view_option == "HEATMAP":
                metrics_to_display = [
                    "KP_Rank", "WIN_25", "LOSS_25", "WIN% ALL GM", "WIN% CLOSE GM",
                    "KP_AdjEM", "KP_SOS_AdjEM", "OFF EFF", "DEF EFF", 
                    "TS%", "OPP TS%", "AST/TO%", "STOCKS/GM", "AVG MARGIN"
                ]
                metrics_to_display = [m for m in metrics_to_display if m in selected_df.columns]
                display_df = selected_df[metrics_to_display].copy()
                ncaa_avg = df_main[metrics_to_display].mean().to_frame().T
                ncaa_avg.index = ["NCAA AVERAGE"]
                display_df = pd.concat([display_df, ncaa_avg])
                format_dict = {
                    "KP_Rank": "{:.0f}",
                    "WIN_25": "{:.0f}", 
                    "LOSS_25": "{:.0f}",
                    "WIN% ALL GM": "{:.1%}",
                    "WIN% CLOSE GM": "{:.1%}",
                    "KP_AdjEM": "{:.1f}",
                    "KP_SOS_AdjEM": "{:.1f}",
                    "OFF EFF": "{:.1f}", 
                    "DEF EFF": "{:.1f}",
                    "TS%": "{:.1f}%", 
                    "OPP TS%": "{:.1f}%",
                    "AST/TO%": "{:.2f}",
                    "STOCKS/GM": "{:.1f}",
                    "AVG MARGIN": "{:.1f}"
                }
                color_scales = {
                    "KP_Rank": "RdBu_r",
                    "WIN_25": "RdBu", 
                    "LOSS_25": "RdBu_r",
                    "AVG MARGIN": "RdBu",
                    "WIN% ALL GM": "RdBu",
                    "WIN% CLOSE GM": "RdBu",
                    "KP_AdjEM": "RdBu",
                    "KP_SOS_AdjEM": "RdBu",
                    "OFF EFF": "RdBu", 
                    "DEF EFF": "RdBu_r",
                    "TS%": "RdBu", 
                    "OPP TS%": "RdBu_r",
                    "AST/TO%": "RdBu",
                    "STOCKS/GM": "RdBu", 
                }
                styled_table = display_df.style.format({col: fmt for col, fmt in format_dict.items() if col in display_df.columns})
                for col, cmap in color_scales.items():
                    if col in display_df.columns:
                        styled_table = styled_table.background_gradient(cmap=cmap, subset=[col])
                styled_table = styled_table.set_table_styles([
                    {'selector': 'th', 'props': [
                        ('background-color', '#0360CE'),
                        ('color', 'white'),
                        ('font-weight', 'bold'),
                        ('text-align', 'center'),
                        ('padding', '10px'),
                        ('border', '1px solid #444')
                    ]},
                    {'selector': 'td', 'props': [
                        ('text-align', 'center'),
                        ('padding', '8px'),
                        ('border', '1px solid #444')
                    ]},
                    {'selector': 'tr:last-child', 'props': [
                        ('font-weight', 'bold'),
                        ('background-color', 'rgba(255,255,255,0.1)')
                    ]}
                ])
                styled_table = styled_table.set_caption("COMPARATIVE MATRIX")
                st.markdown(styled_table.to_html(), unsafe_allow_html=True)
                
            else:
                radar_fig = create_radar_chart(selected_teams, df_main)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
                else:
                    st.warning("Failed to display radar charts for selected teams.")
                
                st.subheader("Metric Correlation Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    x_metric = st.selectbox("X-Axis Metric", core_cols, index=core_cols.index("OFF EFF") if "OFF EFF" in core_cols else 0)
                with col2:
                    y_metric = st.selectbox("Y-Axis Metric", core_cols, index=core_cols.index("DEF EFF") if "DEF EFF" in core_cols else 0)
                fig_scatter = px.scatter(
                    df_main.reset_index(), 
                    x=x_metric, 
                    y=y_metric, 
                    color="CONFERENCE" if "CONFERENCE" in df_main.columns else None,
                    hover_name="TEAM",
                    size="KP_AdjEM_Offset" if "KP_AdjEM_Offset" in df_main.columns else None,
                    size_max=15, 
                    opacity=0.6, 
                    template="plotly_dark",
                    title=f"{y_metric} vs {x_metric} - Selected Teams Highlighted",
                    height=600
                )
                if (("OFF" in x_metric and "DEF" in y_metric) or ("DEF" in x_metric and "OFF" in y_metric)):
                    x_avg, y_avg = df_main[x_metric].mean(), df_main[y_metric].mean()
                    fig_scatter.add_hline(y=y_avg, line_dash="dash", line_color="white", opacity=0.4)
                    fig_scatter.add_vline(x=x_avg, line_dash="dash", line_color="white", opacity=0.4)
                    quadrants = [
                        {"x": x_avg * 0.9, "y": y_avg * 0.9, "text": "WEAK OFF / SOLID DEF"},
                        {"x": x_avg * 1.1, "y": y_avg * 0.9, "text": "SOLID OFF / SOLID DEF"},
                        {"x": x_avg * 0.9, "y": y_avg * 1.1, "text": "WEAK OFF / WEAK DEF"},
                        {"x": x_avg * 1.1, "y": y_avg * 1.1, "text": "SOLID OFF / WEAK DEF"}
                    ]
                    for q in quadrants:
                        fig_scatter.add_annotation(
                            x=q["x"], y=q["y"], text=q["text"], showarrow=False,
                            font=dict(color="white", size=12, family="Arial"), 
                            opacity=0.8,
                            bgcolor="rgba(0,0,0,0.5)",
                            bordercolor="white",
                            borderwidth=1,
                            borderpad=4
                        )
                selected_team_data = df_main[df_main["TM_KP"].isin(selected_teams)]
                for team in selected_teams:
                    team_data = df_main[df_main["TM_KP"] == team]
                    if not team_data.empty:
                        fig_scatter.add_trace(go.Scatter(
                            x=team_data[x_metric],
                            y=team_data[y_metric],
                            mode="markers+text",
                            marker=dict(size=14, color="yellow", line=dict(width=2, color="white")),
                            text=team,
                            textposition="top center",
                            name=team,
                            textfont=dict(color="white", size=12, family="Arial"),
                            hoverinfo="name+x+y"
                        ))
                fig_scatter.update_layout(
                    hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=14, font_family="Arial"),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        bgcolor="rgba(0,0,0,0.5)"
                    )
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("👈 Please select at least one team for comparison")
    else:
        st.error("Team data not available in the dataset.")
    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("📊 Explanation of Metrics"):
        st.markdown("""
        #### Efficiency Metrics
        - **KP_AdjEM**: KenPom's Adjusted Efficiency Margin
        - **OFF EFF**: Offensive Efficiency
        - **DEF EFF**: Defensive Efficiency (lower is better)
        
        #### Shooting Metrics
        - **TS%**: True Shooting Percentage
        - **eFG%**: Effective Field Goal Percentage
        
        #### Ball Control Metrics
        - **AST/TO%**: Assist to Turnover Ratio
        - **STOCKS/GM**: Combined Steals and Blocks per game
        
        #### Performance Metrics
        - **WIN% CLOSE GM**: Clutch win percentage in close games
        - **AVG MARGIN**: Average scoring margin
        """)
        
# ----------------------------------------------------------------------------
# --- 🔮 PREDICTIONS Tab (Bracket Simulation) ---

def color_log_text(round_name, text):
    """Return HTML string with color-coded text for each round."""
    round_html_colors = {
        "Round of 64":  "#3498DB",  # Blue
        "Round of 32":  "#00CCCC",  # Cyan
        "Sweet 16":     "#2ECC71",  # Green
        "Elite 8":      "#F1C40F",  # Yellow
        "Final Four":   "#9B59B6",  # Magenta
        "Championship": "#E74C3C",  # Red
    }
    color_hex = round_html_colors.get(round_name, "#FFFFFF")
    return f"<span style='color:{color_hex}; font-weight:bold;'>{text}</span>"

with tab_pred:
    st.header("Bracket Simulation")
    
    # Create two columns for simulation controls and bracket visualization
    sim_col, viz_col = st.columns([3, 2])
    
    with sim_col:
        # Checkbox to show detailed logs
        show_detailed_logs = st.checkbox("Show Detailed Logs (Single Simulation Recommended)", value=True)
        st.write("Run the tournament simulation across multiple iterations to see aggregated outcomes.")
    
        # Initialize session state variable if needed
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = {}
    
        # Run simulation when button is clicked
        if st.button("Run Bracket Simulation", key="btn_run_bracket"):
            with st.spinner("Simulating tournament..."):
                # 1) Aggregated stats
                aggregated_analysis = run_tournament_simulation(num_simulations=100, use_analytics=True)

                # 2) Single or multi-simulation full detail
                all_sim_results = run_simulation(use_analytics=True, simulations=1)

                # 3) If you want to fix the Final Four for each simulation in that list
                for sim_result in all_sim_results:
                    if sim_result.get('champion') is None and len(sim_result.get('region_champions', {})) == 4:
                        # Perform final four logic
                        region_champs = list(sim_result['region_champions'].values())
                        semi1_prob = calculate_win_probability(region_champs[0], region_champs[1])
                        semi1_winner = region_champs[0] if random.random() < semi1_prob else region_champs[1]
                        semi2_prob = calculate_win_probability(region_champs[2], region_champs[3])
                        semi2_winner = region_champs[2] if random.random() < semi2_prob else region_champs[3]
                        final_prob = calculate_win_probability(semi1_winner, semi2_winner)
                        champion = semi1_winner if random.random() < final_prob else semi2_winner
                        sim_result['champion'] = champion

                # 4) Store in session_state
                st.session_state.simulation_results['aggregated_analysis'] = aggregated_analysis
                st.session_state.simulation_results['single_sim_results']  = all_sim_results

            # Show success
            st.success("Simulation complete!")
    
             # 5) Now we display everything from st.session_state
            #    aggregated_analysis is st.session_state.simulation_results['aggregated_analysis']
            #    single_sim or multi-sim is st.session_state.simulation_results['single_sim_results']

            # Display Championship Win Probabilities
            try:
                champ_df = st.session_state.simulation_results['aggregated_analysis']['champion_probabilities']
                # style it, show it, etc...
            except Exception as e:
                st.error(f"Error displaying Championship Win Probabilities: {e}")

            # Regional Win Probabilities
            try:
                region_probs = st.session_state.simulation_results['aggregated_analysis']['region_probabilities']
                fig_regional = create_regional_prob_chart(region_probs)
                if fig_regional:
                    st.plotly_chart(fig_regional, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying Regional Win Probabilities: {e}")

            # Upset Analysis
            try:
                upset_pct_aggregated = st.session_state.simulation_results['aggregated_analysis']['upset_pct_aggregated']
                # etc...
            except Exception as e:
                st.error(f"Error displaying Upset Analysis: {e}")
    
            # Plot the aggregated results
            try:
                fig_champ, fig_upsets = visualize_aggregated_results(
                    st.session_state.simulation_results['aggregated_analysis']
                )
                if fig_champ:
                    st.plotly_chart(fig_champ, use_container_width=True)
                if fig_upsets:
                    st.plotly_chart(fig_upsets, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate aggregated visualizations: {e}")

            # Show single-sim results
            try:
                if st.session_state.simulation_results['single_sim_results']:
                    st.info("Detailed game outcomes for a single simulation run:")
                    display_simulation_results(st.session_state.simulation_results['single_sim_results'], st)
                else:
                    st.warning("No single simulation results to display.")
            except Exception as e:
                st.error(f"Error displaying detailed simulation logs: {e}")

            # --- Display Visualizations ---
    
            # st.subheader("Championship Win Probabilities")
            # try:
            #     champ_df = st.session_state.simulation_results['aggregated_analysis']['champion_probabilities'].copy()
            #     numeric_cols_champ = champ_df.select_dtypes(include=[float, int]).columns
            #     styled_champ = champ_df.style.format("{:.2%}", subset=["Championship_Probability"])
            #     styled_champ = (styled_champ.background_gradient(cmap="RdYlGn", subset=numeric_cols_champ)
            #                     .set_table_styles(detailed_table_styles)
            #                     .set_caption("Championship Win Probabilities by Team"))
            #     st.markdown(styled_champ.to_html(), unsafe_allow_html=True)
            # except Exception as e:
            #     st.error(f"Error displaying Championship Win Probabilities: {e}")
    
            # st.subheader("Regional Win Probabilities")
            # try:
            #     region_probabilities = st.session_state.simulation_results['aggregated_analysis'].get('region_probabilities')
            #     if region_probabilities is not None and not region_probabilities.empty:
            #         fig_regional = create_regional_prob_chart(region_probabilities)
            #         st.plotly_chart(fig_regional, use_container_width=True)
            #     else:
            #         st.warning("Regional win probabilities data not available.")
            # except Exception as e:
            #     st.error(f"Error displaying Regional Win Probabilities: {e}")
    
            # st.subheader("Aggregated Upset Analysis")
            # try:
            #     upset_pct_aggregated = st.session_state.simulation_results['aggregated_analysis'].get('upset_pct_aggregated')
            #     if upset_pct_aggregated is not None:
            #         upset_summary_df = pd.DataFrame({
            #             'Round': upset_pct_aggregated.index,
            #             'Upset %': upset_pct_aggregated.values.round(1)
            #         })
            #         numeric_cols_upsets = upset_summary_df.select_dtypes(include=[float, int]).columns
            #         styled_upsets = (upset_summary_df.style.format("{:.1f}", subset=["Upset %"])
            #                          .background_gradient(cmap="RdYlGn", subset=numeric_cols_upsets)
            #                          .set_table_styles(detailed_table_styles)
            #                          .set_caption("Upset Analysis by Round"))
            #         st.markdown(styled_upsets.to_html(), unsafe_allow_html=True)
            #     else:
            #         st.warning("Upset analysis data not available.")
            # except Exception as e:
            #     st.error(f"Error displaying Upset Analysis: {e}")
    
            # st.subheader("Aggregated Simulation Visualization")
            # try:
            #     fig_champ, fig_upsets = visualize_aggregated_results(
            #         st.session_state.simulation_results['aggregated_analysis']
            #     )
            #     if fig_champ:
            #         st.plotly_chart(fig_champ, use_container_width=True)
            #     if fig_upsets:
            #         st.plotly_chart(fig_upsets, use_container_width=True)
            # except Exception as e:
            #     st.error(f"Could not generate aggregated visualizations: {e}")
    
    with viz_col:
        st.subheader("Bracket Visualization")
        st.markdown("Select a view to explore the tournament teams")
        viz_type = st.radio("Visualization Type", ["Team Stats"], horizontal=True)
        if 'TR_df' not in globals():
            _ = prepare_tournament_data()
        if viz_type == "Team Stats":
            all_tourney_teams = TR_df[TR_df['SEED_25'].notna()]['TM_KP'].tolist()
            selected_team = st.selectbox("Select Team", sorted(all_tourney_teams))
            team_data = TR_df[TR_df["TM_KP"] == selected_team].iloc[0]
            create_team_radar(team_data, dark_mode=True)
            st.markdown("### Key Team Stats")
            seed_str = f"{int(team_data['SEED_25'])}" if pd.notna(team_data['SEED_25']) else "N/A"
            region_str = (team_data['REGION_25'] if pd.notna(team_data['REGION_25'])
                          else (team_data['REG_CODE_25'] if pd.notna(team_data['REG_CODE_25']) else "N/A"))
            key_stats = {
                "Seed": f"{seed_str} ({region_str})",
                "Record": f"{team_data['WIN_25']:.0f}-{team_data['LOSS_25']:.0f}",
                "NET Rank": f"{int(team_data['NET_25'])}",
                "KenPom Rank": f"{int(team_data['KP_Rank'])}",
                "KenPom Adj EM": f"{team_data['KP_AdjEM']:.2f}",
                "KenPom Adj OEff": f"{team_data['KP_AdjO']:.2f}",
                "KenPom Adj DEff": f"{team_data['KP_AdjD']:.2f}",
                "TeamRankings OEff": f"{team_data['OFF EFF']:.2f}",
                "TeamRankings DEff.": f"{team_data['DEF EFF']:.2f}",
            } 
            stat_col1, stat_col2 = st.columns(2)
            for i, (stat, value) in enumerate(key_stats.items()):
                if i % 2 == 0:
                    stat_col1.metric(stat, value)
                else:
                    stat_col2.metric(stat, value)
        else:
            st.markdown("### Bracket Overview")
            # Placeholder: Additional bracket overview visualizations, etc here.


# with tab_pred:
#     st.header("Bracket Simulation")

#     # 2) Let the user decide if they want to see detailed logs (single-run example)
#     show_detailed_logs = st.checkbox("Show Detailed Logs (Single Simulation Recommended)", value=False)

#     st.write("Run the tournament simulation across multiple iterations to see aggregated outcomes.")
    
#     # 3) Single button with a unique key argument
#     if st.button("Run Bracket Simulation", key="btn_run_bracket"):
#         with st.spinner("Simulating tournament..."):
#             # Run your existing multi-simulation
#             aggregated_analysis = run_tournament_simulation(num_simulations=100, use_analytics=True)

#         st.success("Simulation complete!")

#         # -----------------------------------------
#         # Apply advanced styling to two result DataFrames:
#         #   (1) Championship Win Probabilities
#         #   (2) Upset Summary
#         # for consistency with the rest of the app.
#         # -----------------------------------------

#         # (A) Championship Win Probabilities
#         st.subheader("Championship Win Probabilities")
#         champ_df = aggregated_analysis['champion_probabilities'].copy()
        
#         # Convert numeric columns to background gradient
#         numeric_cols_champ = champ_df.select_dtypes(include=[float, int]).columns
#         styled_champ = (
#             champ_df.style
#             .format("{:.2%}", subset=["Championship_Probability"])  # percentage style
#             .background_gradient(cmap="RdYlGn", subset=numeric_cols_champ)
#             .set_table_styles(detailed_table_styles)
#             .set_caption("Championship Win Probabilities by Team")
#         )
#         # Render as HTML
#         st.markdown(styled_champ.to_html(), unsafe_allow_html=True)

#         # (B) Regional Win Probabilities Chart (plotly)
#         st.subheader("Regional Win Probabilities")
#         if 'region_probabilities' in aggregated_analysis:
#             fig_regional = create_regional_prob_chart(aggregated_analysis['region_probabilities'])
#             st.plotly_chart(fig_regional, use_container_width=True)
#         else:
#             st.warning("Regional win probabilities data not available.")

#         # (C) Upset Analysis
#         st.subheader("Aggregated Upset Analysis")
#         upset_summary_df = pd.DataFrame({
#             'Round': aggregated_analysis['upset_pct_aggregated'].index,
#             'Upset %': aggregated_analysis['upset_pct_aggregated'].values.round(1)
#         })
#         # Style that similarly
#         numeric_cols_upsets = upset_summary_df.select_dtypes(include=[float, int]).columns
#         styled_upsets = (
#             upset_summary_df.style
#             .format("{:.1f}", subset=["Upset %"])
#             .background_gradient(cmap="RdYlGn", subset=numeric_cols_upsets)
#             .set_table_styles(detailed_table_styles)
#             .set_caption("Upset Analysis by Round")
#         )
#         st.markdown(styled_upsets.to_html(), unsafe_allow_html=True)

#         # (D) Show aggregated matplotlib figure with uniform styling
#         try:
#             agg_viz_fig = visualize_aggregated_results(aggregated_analysis)
#             st.pyplot(agg_viz_fig, use_container_width=True)  # Pass figure explicitly
#         except Exception as e:
#             st.error(f"Could not generate aggregated visualizations: {e}")

#         # (E) If the user checked 'Show Detailed Logs,' run a single-sim & print logs
#         if show_detailed_logs:
#             st.info("Detailed logs below show one example simulation’s game outcomes:")
#             single_run_results = run_simulation(use_analytics=True, simulations=1)[0]
#             detailed_games = single_run_results["all_games"]

#             st.write("---")
#             st.write("### Detailed Game Outcomes (Single Simulation):")

#             # Print each round's outcome in color-coded HTML
#             for gm in detailed_games:
#                 round_label   = gm["round_name"]
#                 t1, t2        = gm["team1"], gm["team2"]
#                 winner        = gm["winner"]
#                 prob_to_win   = gm["win_prob"]
#                 colored_label = color_log_text(round_label, f"[{round_label}]")

#                 # Mark upsets for emphasis
#                 upset_flag = ""
#                 if gm["winner_seed"] > min(gm["seed1"], gm["seed2"]):
#                     upset_flag = "<b>(UPSET!)</b>"

#                 line_html = (
#                     f"{colored_label} &nbsp;"
#                     f"{t1} vs. {t2} → "
#                     f"<b>Winner:</b> {winner} "
#                     f"<small>(win prob {prob_to_win:.1%})</small> "
#                     f"{upset_flag}"
#                 )
#                 st.markdown(line_html, unsafe_allow_html=True)


# with tab_pred:
#     st.header("Bracket Simulation")
#     st.write("Run the tournament simulation across multiple iterations to see aggregated outcomes.")
#     if st.button("Run Bracket Simulation"):
#         with st.spinner("Simulating tournament..."):
#             # Run simulation (using 100 simulations as an example; adjust as needed)
#             aggregated_analysis = run_tournament_simulation(num_simulations=100, use_analytics=True)
#         st.success("Simulation complete!")
        
#         # Display Championship Win Probabilities as a styled table
#         st.subheader("Championship Win Probabilities")
#         st.dataframe(aggregated_analysis['champion_probabilities'])
        
#         # Display Regional Win Probabilities as a 2x2 subplot chart for all four regions
#         st.subheader("Regional Win Probabilities")
#         if 'region_probabilities' in aggregated_analysis:
#             fig_regional = create_regional_prob_chart(aggregated_analysis['region_probabilities'])
#             st.plotly_chart(fig_regional, use_container_width=True)
#         else:
#             st.warning("Regional win probabilities data not available.")
        
#         # Display Aggregated Upset Analysis as a table
#         st.subheader("Aggregated Upset Analysis")
#         upset_summary_df = pd.DataFrame({
#             'Round': aggregated_analysis['upset_pct_aggregated'].index,
#             'Upset %': aggregated_analysis['upset_pct_aggregated'].values.round(1)
#         })
#         st.dataframe(upset_summary_df)
        
#         # Additional aggregated visualizations (if available)
#         try:
#             agg_viz_fig = visualize_aggregated_results(aggregated_analysis)
#             st.pyplot(agg_viz_fig)
#         except Exception as e:
#             st.error(f"Could not generate aggregated visualizations: {e}")


if FinalFour25_logo:
    st.image(FinalFour25_logo, width=750)

st.markdown("---")
st.caption("Python code framework available on [GitHub](https://github.com/nehat312/march-madness-2025)")
st.caption("DATA SOURCED FROM: [TeamRankings](https://www.teamrankings.com/ncaa-basketball/ranking/predictive-by-other/), [KenPom](https://kenpom.com/)")
st.stop()

## ----- EXTRA ----- ##


# --- Histogram Tab --- #
# with tab_hist:
#     st.header("Histogram")
#     numeric_cols = [c for c in core_cols if c in df_main.columns and pd.api.types.is_numeric_dtype(df_main[c])]
#     hist_metric = st.selectbox(
#         "Select Metric for Histogram", numeric_cols,
#         index=numeric_cols.index("KP_AdjEM") if "KP_AdjEM" in numeric_cols else 0
#     )
#     fig_hist = px.histogram(
#         df_main, x=hist_metric, nbins=25, marginal="box",
#         color_discrete_sequence=["dodgerblue"], template="plotly_dark",
#         title=f"Distribution of {hist_metric} (All Teams)"
#     )
#     fig_hist.update_layout(bargap=0.1)
#     st.plotly_chart(fig_hist, use_container_width=True)
#     with st.expander("About Histogram Metrics:"):
#         st.markdown("""
#         **Histogram Metric Description:**
#         - **KP_AdjEM**: Adjusted efficiency margin from KenPom ratings.
#         - **OFF EFF/DEF EFF**: Offensive/Defensive efficiency.
#         - Other metrics follow similar definitions as per NCAA advanced statistics.
#         """)

# # --- Correlation Heatmap Tab --- #
# with tab_corr:
#     st.header("Correlation Heatmap")
#     numeric_cols = [c for c in core_cols if c in df_main.columns and pd.api.types.is_numeric_dtype(df_main[c])]
#     default_corr_metrics = [m for m in ["KP_AdjEM", "OFF EFF", "DEF EFF", "PTS/GM", "OPP PTS/GM"] if m in numeric_cols]
#     selected_corr_metrics = st.multiselect("Select Metrics for Correlation Analysis", options=numeric_cols, default=default_corr_metrics)
#     if len(selected_corr_metrics) >= 2:
#         df_for_corr = df_main[selected_corr_metrics].dropna()
#         corr_mat = df_for_corr.corr().round(2)
#         fig_corr = px.imshow(
#             corr_mat,
#             text_auto=True,
#             color_continuous_scale="RdBu_r",
#             title="Correlation Matrix",
#             template="plotly_dark"
#         )
#         fig_corr.update_layout(width=800, height=700)
#         st.plotly_chart(fig_corr, use_container_width=True)
#     else:
#         st.warning("Please select at least 2 metrics for correlation analysis.")
#     with st.expander("About Correlation Metrics:"):
#         st.markdown("""
#         **Correlation Heatmap Glossary:**
#         - **Correlation Coefficient**: Measures linear relationship between two variables.
#         - **Positive/Negative Correlation**: Indicates the direction of the relationship.
#         - **Metrics**: Derived from advanced team statistics.
#         """)