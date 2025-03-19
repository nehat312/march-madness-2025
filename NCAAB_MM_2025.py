import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import base64
import copy
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
        #'KP_AdjO',
        #'KP_AdjD',
        'OFF EFF',
        'DEF EFF',
        'AST/TO%',
        'STOCKS-TOV/GM',

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
    df = df_main.copy()
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
            all_values = df_main[metric].dropna()
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
            color_continuous_scale=RdYlGn,
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

# Define the correct bracket matchup structure based on NCAA tournament seeding
def get_matchups_by_round():
    """
    Returns the bracket structure for each round (seeds 1–16 in each region).
    Each dict entry is region -> list of (seed1, seed2) pairs or indices:
      round_64  : seeds in actual pairs (1–16, 8–9, etc.)
      round_32  : winners feed in (0 vs 1, 2 vs 3, etc.)
      sweet_16  : ...
      elite_8   : ...
      final_four: [ (0,1), (2,3) ] means West/East, South/Midwest
      championship: [ (0,1) ] from the two winners of final_four
    """
    round_64 = {
        'West':    [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)],
        'East':    [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)],
        'South':   [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)],
        'Midwest': [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
    }
    round_32 = {
        'West':    [(0,1),(2,3),(4,5),(6,7)],
        'East':    [(0,1),(2,3),(4,5),(6,7)],
        'South':   [(0,1),(2,3),(4,5),(6,7)],
        'Midwest': [(0,1),(2,3),(4,5),(6,7)]
    }
    sweet_16 = {
        'West':    [(0,1),(2,3)],
        'East':    [(0,1),(2,3)],
        'South':   [(0,1),(2,3)],
        'Midwest': [(0,1),(2,3)]
    }
    elite_8 = {
        'West':    [(0,1)],
        'East':    [(0,1)],
        'South':   [(0,1)],
        'Midwest': [(0,1)]
    }
    final_four   = [(0,1), (2,3)]  # 0=West,1=East,2=South,3=Midwest
    championship = [(0,1)]
    return round_64, round_32, sweet_16, elite_8, final_four, championship


def simulate_game(team1, team2):
    """
    Simulate one game: return a *copy* of the winner's dictionary.
    """
    pA = calculate_win_probability(team1, team2)
    winner = team1 if (random.random() < pA) else team2
    return copy.deepcopy(winner)



def simulate_tournament(bracket, num_simulations=1000):
    """
    Full bracket simulation from Round of 64 to Championship, repeated num_simulations times.
    Return a dict of { 'Round of 64': {team->pct}, ..., 'Champion': {team->pct} }.
    """
    round_64, round_32, sweet_16, elite_8, final_four, championship = get_matchups_by_round()
    
    # We'll maintain these round names to populate the results dictionary:
    rounds_list = [
        "Round of 64",
        "Round of 32",
        "Sweet 16",
        "Elite 8",
        "Final Four",
        "Championship",
        "Champion"
    ]
    
    # region_order is used later to map index -> region
    region_order = ["West", "East", "South", "Midwest"]
    
    # Prepare a structure to count how often each team advances to each round
    results = {r: {} for r in rounds_list}

    for _ in range(num_simulations):
        # Copy the bracket so we don't mutate the original
        # bracket is a dict: { "West":[{team:'A',seed:1...}...], "East":[...], ... }
        current = {}
        for reg in bracket:
            # Only copy if the region is present
            current[reg] = [copy.deepcopy(t) for t in bracket[reg]]

        # --- Round of 64 ---
        r64_winners = {}
        for region, matchups in round_64.items():
            # If region not in bracket, skip
            if region not in current:
                continue
            
            winners_for_region = []
            for (s1, s2) in matchups:
                # Find teams with these seeds (skip if missing)
                t1 = next((x for x in current[region] if x['seed'] == s1), None)
                t2 = next((x for x in current[region] if x['seed'] == s2), None)
                if not t1 or not t2:
                    # If either is missing, skip this matchup
                    continue

                w = simulate_game(t1, t2)
                winners_for_region.append(w)
                # Mark that w advanced in "Round of 64"
                results["Round of 64"][w['team']] = results["Round of 64"].get(w['team'], 0) + 1
            
            r64_winners[region] = winners_for_region

        # --- Round of 32 ---
        r32_winners = {}
        for region, pairs in round_32.items():
            if region not in current or region not in r64_winners:
                continue
            
            winners_for_region = []
            region_list = r64_winners[region]
            # Make sure region_list has enough teams for these pair indices
            for (i, j) in pairs:
                if i < len(region_list) and j < len(region_list):
                    w = simulate_game(region_list[i], region_list[j])
                    winners_for_region.append(w)
                    results["Round of 32"][w['team']] = results["Round of 32"].get(w['team'], 0) + 1
            
            r32_winners[region] = winners_for_region

        # --- Sweet 16 ---
        s16_winners = {}
        for region, pairs in sweet_16.items():
            if region not in current or region not in r32_winners:
                continue
            
            winners_for_region = []
            region_list = r32_winners[region]
            for (i, j) in pairs:
                if i < len(region_list) and j < len(region_list):
                    w = simulate_game(region_list[i], region_list[j])
                    winners_for_region.append(w)
                    results["Sweet 16"][w['team']] = results["Sweet 16"].get(w['team'], 0) + 1
            
            s16_winners[region] = winners_for_region

        # --- Elite 8 ---
        e8_finalists = []
        for region, pairs in elite_8.items():
            if region not in current or region not in s16_winners:
                continue
            
            region_list = s16_winners[region]
            for (i, j) in pairs:
                if i < len(region_list) and j < len(region_list):
                    w = simulate_game(region_list[i], region_list[j])
                    e8_finalists.append((region, w))
                    results["Elite 8"][w['team']] = results["Elite 8"].get(w['team'], 0) + 1

        # We now have a list of tuples (region, champion_of_that_region).
        # We'll put them into region_champs = { 'West':{...}, 'East':{...}, etc. }
        region_champs = {}
        for (reg, champ_dict) in e8_finalists:
            region_champs[reg] = champ_dict

        # --- Final Four ---
        ff_winners = []
        # final_four = [(0,1), (2,3)] means West/East => game1, South/Midwest => game2
        for (idxA, idxB) in final_four:
            regA = region_order[idxA]
            regB = region_order[idxB]
            if (regA not in region_champs) or (regB not in region_champs):
                continue

            w = simulate_game(region_champs[regA], region_champs[regB])
            ff_winners.append(w)
            results["Final Four"][w['team']] = results["Final Four"].get(w['team'], 0) + 1

        # --- Championship ---
        champion = None
        for (i, j) in championship:
            if i < len(ff_winners) and j < len(ff_winners):
                champion = simulate_game(ff_winners[i], ff_winners[j])
                results["Championship"][champion['team']] = results["Championship"].get(champion['team'], 0) + 1
                results["Champion"][champion['team']] = results["Champion"].get(champion['team'], 0) + 1

    # Convert raw counts → percentages
    for round_name in rounds_list:
        for tm in results[round_name]:
            results[round_name][tm] = 100.0 * (results[round_name][tm] / num_simulations)

    return results

# def prepare_tournament_data():
#     """
#     Prepares the 64-team bracket data from your df_main,
#     ensuring we have seeds 1..16 in each region and returning
#     { 'region_names': [...], 'region_teams': {...}, 'tournament_teams': DataFrame }.
#     """
#     global TR_df  # used for bracket radar grids
#     TR_df = df_main.copy()

#     required_cols = ['SEED_25', 'REGION_25', 'KP_Rank', 'KP_AdjEM', 'OFF EFF', 'DEF EFF']
#     for col in required_cols:
#         if col not in TR_df.columns:
#             sim_logger.warning(f"Missing required column: {col}")
#             return None

#     tournament_teams = TR_df.dropna(subset=required_cols).copy()
#     tournament_teams['SEED_25'] = tournament_teams['SEED_25'].astype(int)

#     # Only keep 1..16 seeds
#     tournament_teams = tournament_teams[(tournament_teams['SEED_25'] >= 1) &
#                                         (tournament_teams['SEED_25'] <= 16)]

#     # If missing "TM_KP", fall back to "TEAM" or "TM_TR"
#     if 'TM_KP' not in tournament_teams.columns:
#         if 'TEAM' in tournament_teams.columns:
#             tournament_teams['TM_KP'] = tournament_teams['TEAM']
#         elif 'TM_TR' in tournament_teams.columns:
#             tournament_teams['TM_KP'] = tournament_teams['TM_TR']
#         else:
#             sim_logger.warning("No team name column found (TM_KP or TEAM or TM_TR).")
#             return None

#     # Example bonuses for certain teams – optional
#     tournament_teams['TOURNEY_SUCCESS'] = 0.0
#     for perennial in ["Duke", "Kentucky", "Kansas", "North Carolina", "Gonzaga", "Michigan St."]:
#         if perennial in tournament_teams['TM_KP'].values:
#             tournament_teams.loc[tournament_teams['TM_KP'] == perennial, 'TOURNEY_SUCCESS'] = 2.0

#     # Add an example "experience" bonus from prior seeds
#     tournament_teams['TOURNEY_EXPERIENCE'] = 0.0
#     if 'SEED_23' in tournament_teams.columns:
#         # Just a naive example: if a team made the 2023 dance
#         tournament_teams.loc[tournament_teams['SEED_23'].notna() &
#                              (tournament_teams['SEED_23'] <= 16),
#                              'TOURNEY_EXPERIENCE'] = 1.0
#         tournament_teams.loc[tournament_teams['SEED_23'].notna() &
#                              (tournament_teams['SEED_23'] <= 4),
#                              'TOURNEY_EXPERIENCE'] = 2.0

#     # Identify unique regions
#     region_names = tournament_teams['REGION_25'].unique().tolist()

#     region_teams = {}
#     for reg in region_names:
#         df_reg = tournament_teams[tournament_teams['REGION_25'] == reg].sort_values('SEED_25')
#         # convert each row → dict with relevant info
#         teams_list = df_reg.apply(lambda row: {
#             'Team': row['TM_KP'],
#             'Seed': int(row['SEED_25']),
#             'KP_Rank': row['KP_Rank'],
#             'KP_AdjEM': row['KP_AdjEM'],
#             'OFF EFF': row['OFF EFF'],
#             'DEF EFF': row['DEF EFF'],
#             'KP_AdjO': row.get('KP_AdjO', 0),
#             'KP_AdjD': row.get('KP_AdjD', 0),
#             'TOURNEY_SUCCESS': row.get('TOURNEY_SUCCESS', 0),
#             'TOURNEY_EXPERIENCE': row.get('TOURNEY_EXPERIENCE', 0),
#             'WIN_PCT': row.get('WIN% ALL GM', 0.5),
#             'CLOSE_GAME_PCT': row.get('WIN% CLOSE GM', 0.5),
#             'SOS': row.get('KP_SOS_AdjEM', 0),
#             'Region': row['REGION_25']
#         }, axis=1).tolist()

#         # Keep only up to 16 teams to avoid extras
#         if len(teams_list) > 16:
#             teams_list = teams_list[:16]
#         region_teams[reg] = teams_list

#     return {
#         'tournament_teams': tournament_teams,
#         'region_names': region_names,
#         'region_teams': region_teams
#     }

def prepare_tournament_data(df):
    """
    From df, keep seeds 1..16 in each region. If a region lacks any seed
    from 1..16, we skip that region entirely. Return bracket dict: 
      {
        'West': [ { 'team':..., 'seed':1, 'KP_AdjEM':...}, ...16 teams ],
        'East': [...16 teams],
        'South': [...16 teams],
        'Midwest': [...16 teams]
      }
    """
    required = ['SEED_25','REGION_25','KP_AdjEM','OFF EFF','DEF EFF']
    missing = [c for c in required if c not in df.columns]
    if missing:
        sim_logger.warning(f"Missing required bracket columns: {missing}")
        return None

    bracket_teams = df.dropna(subset=required).copy()
    bracket_teams['SEED_25'] = bracket_teams['SEED_25'].astype(int)
    bracket_teams = bracket_teams[
        (bracket_teams['SEED_25']>=1) & (bracket_teams['SEED_25']<=16)
    ]

    # Decide on the name column
    if 'TM_KP' in bracket_teams.columns:
        name_col = 'TM_KP'
    else:
        # fallback
        name_col = 'TEAM' if 'TEAM' in bracket_teams.columns else bracket_teams.index.name

    # Ensure optional columns exist
    for bonus_col in ['TOURNEY_SUCCESS','TOURNEY_EXPERIENCE']:
        if bonus_col not in bracket_teams.columns:
            bracket_teams[bonus_col] = 0.0

    # Example: give certain perennial teams a bonus
    for perennial in ["Duke","Kentucky","Kansas","North Carolina","Gonzaga","Michigan St."]:
        if perennial in bracket_teams[name_col].values:
            bracket_teams.loc[ bracket_teams[name_col]==perennial, 'TOURNEY_SUCCESS'] = 2.0

    # Build the bracket
    bracket = {}
    # We'll only keep a region if it has exactly one team for *each* seed in [1..16].
    # Alternatively, you could accept partial sets, but then you must handle them in simulate_tournament.
    for region in ['West','East','South','Midwest']:
        region_df = bracket_teams[
            bracket_teams['REGION_25'].str.lower() == region.lower()
        ].copy()

        # Check each seed from 1..16 is present
        region_seeds = region_df['SEED_25'].unique().tolist()
        if len(region_seeds) < 16:
            # You can log or skip
            sim_logger.warning(f"{region} region missing some seeds, found seeds: {region_seeds}. Skipping.")
            continue

        # Sort by seed (1..16)
        region_df = region_df.sort_values('SEED_25')
        region_list = []
        for _, row in region_df.iterrows():
            region_list.append({
                'team':      row[name_col],
                'seed':      int(row['SEED_25']),
                'KP_AdjEM':  float(row['KP_AdjEM']),
                'OFF EFF':   float(row['OFF EFF']),
                'DEF EFF':   float(row['DEF EFF']),
                'TOURNEY_SUCCESS':    float(row['TOURNEY_SUCCESS']),
                'TOURNEY_EXPERIENCE': float(row['TOURNEY_EXPERIENCE'])
            })
        bracket[region] = region_list

    return bracket


def calculate_win_probability(t1, t2):
    """
    Simple logistic model combining:
      - KenPom margin (KP_AdjEM)
      - OFF vs DEF EFF matchup
      - Optional 'experience' & 'success' bonus
      - Mild 'upset factor' if seeds differ significantly
    Returns probability that Team1 (t1) wins.
    """
    t1_off = float(t1.get('OFF EFF', 1.0))
    t1_def = float(t1.get('DEF EFF', 1.0))
    t2_off = float(t2.get('OFF EFF', 1.0))
    t2_def = float(t2.get('DEF EFF', 1.0))

    kp_diff     = float(t1['KP_AdjEM']) - float(t2['KP_AdjEM'])
    matchup_adv = (t1_off - t2_def) - (t2_off - t1_def)

    exp_diff = (float(t1.get('TOURNEY_EXPERIENCE', 0)) - float(t2.get('TOURNEY_EXPERIENCE', 0))) \
             + (float(t1.get('TOURNEY_SUCCESS', 0)) - float(t2.get('TOURNEY_SUCCESS', 0)))

    # Weighted sum → logistic
    factor = (1.0 * kp_diff) + (0.5 * matchup_adv) + (0.2 * exp_diff)
    base_prob = 1.0 / (1.0 + np.exp(-0.1 * factor))

    # Mild upset factor
    seed_diff = t2['seed'] - t1['seed']
    if seed_diff > 0 and t1['seed'] <= 4 and t2['seed'] >= 12:
        base_prob = max(0.65, min(0.95, base_prob - 0.05))

    return max(0.05, min(0.95, base_prob))

def run_games(team_list, pairing_list, round_name, region_name, use_analytics=True):
    """
    Executes a list of matchups for the specified round (e.g. Round of 64)
    given the `pairing_list` of index tuples (i,j).
    Returns (winners[], round_games[]).
    """
    winners = []
    round_games = []

    for (i, j) in pairing_list:
        if i < len(team_list) and j < len(team_list):
            tA = team_list[i]
            tB = team_list[j]

            pA = calculate_win_probability(tA, tB) if use_analytics else 0.5
            winner = tA if random.random() < pA else tB
            loser  = tB if (winner is tA) else tA

            round_games.append({
                'round_name': round_name,
                'region':     region_name,
                'team1':      tA['Team'],
                'seed1':      tA['Seed'],
                'team2':      tB['Team'],
                'seed2':      tB['Seed'],
                'winner':     winner['Team'],
                'winner_seed': winner['Seed'],
                'win_prob':  pA if (winner is tA) else (1 - pA),
            })
            winners.append(winner)

    return winners, round_games

def generate_bracket_round(teams, round_num, region, use_analytics=True):
    """
    Given a list of teams for this round in a standard bracket order,
    simulate each matchup and produce a list of winners in that same order.
    """
    winners = []
    
    # Hard-coded pairings only for Round of 64 (1 vs 16, 8 vs 9, 5 vs 12, 4 vs 13, 6 vs 11, 3 vs 14, 7 vs 10, 2 vs 15).
    # This ensures correct bracket structure for the first round.
    if round_num == 1:
        pairings = [(0, 15), (7, 8), (4, 11), (3, 12),
                    (5, 10), (2, 13), (6, 9), (1, 14)]
    else:
        # For subsequent rounds, just pair winners in consecutive order (0 vs 1, 2 vs 3, etc.).
        pairings = [(i, i+1) for i in range(0, len(teams), 2)]

    for (i, j) in pairings:
        if i < len(teams) and j < len(teams):
            teamA = teams[i]
            teamB = teams[j]
            if use_analytics:
                pA = calculate_win_probability(teamA, teamB)
            else:
                # If 'use_analytics' is False, use a simpler logistic approach
                diff = teamA['KP_AdjEM'] - teamB['KP_AdjEM']
                pA = 1 / (1 + np.exp(-diff / 10))
            winner = teamA if random.random() < pA else teamB
            winner = winner.copy()
            winner['win_prob'] = pA if (winner is teamA) else (1 - pA)
            winners.append(winner)

    return winners

def simulate_region_bracket(teams, region_name, use_analytics=True):
    """
    Simulate a single region (16 seeds), returning:
      - rounds_dict: {1:[r64 winners], 2:[r32 winners], 3:[S16], 4:[E8], 5:[champ]}
      - all_games:   combined list of game dicts with 'round_name', 'winner', etc.
    """

    # (1) Round of 64 pairings (index-based)
    # for seeds sorted 1..16, we store them in a 0-based list
    #  0 vs 15 => seeds 1 vs 16
    #  7 vs 8  => seeds 8 vs 9
    #  4 vs 11 => seeds 5 vs 12
    # etc.
    pairings_r64 = [(0,15),(7,8),(4,11),(3,12),(5,10),(2,13),(6,9),(1,14)]

    # (2) Round of 32
    # winners(0,1) => match1, winners(2,3) => match2, ...
    pairings_r32 = [(0,1),(2,3),(4,5),(6,7)]

    # (3) Sweet 16
    pairings_s16 = [(0,1),(2,3)]

    # (4) Elite 8
    pairings_e8  = [(0,1)]

    # Round 1: Round of 64
    r64_winners, g64 = run_games(teams, pairings_r64, "Round of 64", region_name, use_analytics)
    # Round 2: Round of 32
    r32_winners, g32 = run_games(r64_winners, pairings_r32, "Round of 32", region_name, use_analytics)
    # Round 3: Sweet 16
    s16_winners, g16 = run_games(r32_winners, pairings_s16, "Sweet 16", region_name, use_analytics)
    # Round 4: Elite 8
    e8_winners,  g8  = run_games(s16_winners, pairings_e8,  "Elite 8", region_name, use_analytics)

    region_champion = e8_winners[0] if e8_winners else None

    # Put them into a dictionary for reference
    rounds_dict = {
        1: r64_winners,
        2: r32_winners,
        3: s16_winners,
        4: e8_winners,
        5: [region_champion] if region_champion else [],
    }
    all_games = g64 + g32 + g16 + g8

    return rounds_dict, all_games



def simulate_final_four_and_championship(region_champions, use_analytics=True):
    """
    region_champions: dict => { "East":{...}, "West":{...}, "South":{...}, "Midwest":{...} }
    Simulate the Final Four (West vs East, South vs Midwest) and Championship.
    Returns: (champion_dict, final_games_list)
    """
    # Typical bracket: West vs East, South vs Midwest
    # You can reorder if your bracket dictates differently.
    required = ["West", "East", "South", "Midwest"]
    for r in required:
        if r not in region_champions:
            return None, []

    west_champ    = region_champions["West"]
    east_champ    = region_champions["East"]
    south_champ   = region_champions["South"]
    midwest_champ = region_champions["Midwest"]

    # Semifinal 1: West vs East
    sf1_prob   = calculate_win_probability(west_champ, east_champ) if use_analytics else 0.5
    sf1_winner = west_champ if random.random() < sf1_prob else east_champ

    sf1_game = {
        "round_name":  "Final Four",
        "region":      "Final Four",
        "team1":       west_champ["Team"],
        "seed1":       west_champ["Seed"],
        "team2":       east_champ["Team"],
        "seed2":       east_champ["Seed"],
        "winner":      sf1_winner["Team"],
        "winner_seed": sf1_winner["Seed"],
        "win_prob":    sf1_prob if sf1_winner is west_champ else (1 - sf1_prob),
    }

    # Semifinal 2: South vs Midwest
    sf2_prob   = calculate_win_probability(south_champ, midwest_champ) if use_analytics else 0.5
    sf2_winner = south_champ if random.random() < sf2_prob else midwest_champ

    sf2_game = {
        "round_name":  "Final Four",
        "region":      "Final Four",
        "team1":       south_champ["Team"],
        "seed1":       south_champ["Seed"],
        "team2":       midwest_champ["Team"],
        "seed2":       midwest_champ["Seed"],
        "winner":      sf2_winner["Team"],
        "winner_seed": sf2_winner["Seed"],
        "win_prob":    sf2_prob if sf2_winner is south_champ else (1 - sf2_prob),
    }

    # Championship
    final_prob = calculate_win_probability(sf1_winner, sf2_winner) if use_analytics else 0.5
    champion   = sf1_winner if random.random() < final_prob else sf2_winner

    champ_game = {
        "round_name":  "Championship",
        "region":      "Championship",
        "team1":       sf1_winner["Team"],
        "seed1":       sf1_winner["Seed"],
        "team2":       sf2_winner["Team"],
        "seed2":       sf2_winner["Seed"],
        "winner":      champion["Team"],
        "winner_seed": champion["Seed"],
        "win_prob":    final_prob if champion is sf1_winner else (1 - final_prob),
    }

    final_games = [sf1_game, sf2_game, champ_game]
    return champion, final_games


def run_simulation(use_analytics=True, simulations=1):
    """
    For each simulation:
      1) Prepare region bracket (16 seeds each).
      2) Sim each region's Round of 64 -> 32 -> Sweet 16 -> Elite 8 to find region champions.
      3) Sim Final Four & Championship with those 4 region champions.
      Returns a list of simulation results (one entry per sim).
    """
    data = prepare_tournament_data()
    if not data:
        sim_logger.error("Failed to prepare bracket data.")
        return []

    region_names = data['region_names']
    region_teams = data['region_teams']

    all_sim_results = []

    for sim_num in range(simulations):
        region_results   = {}
        region_champions = {}
        all_games        = []

        # We specifically want to run for the main four: "West","East","South","Midwest"
        # but only if that region is present and has 16 teams
        # (You can also just loop over each region in region_names if you prefer.)
        main_regions = ["East", "West", "South", "Midwest"]
        for reg in main_regions:
            if reg in region_teams and len(region_teams[reg]) == 16:
                # simulate that region bracket
                bracket_rounds, bracket_games = simulate_region_bracket(region_teams[reg],
                                                                        reg,
                                                                        use_analytics)
                region_results[reg] = bracket_rounds

                # final round => bracket_rounds[5] is the region champion
                if 5 in bracket_rounds and len(bracket_rounds[5]) > 0:
                    region_champions[reg] = bracket_rounds[5][0]

                all_games.extend(bracket_games)

        # If we do have all four region champs, do Final 4 => champion
        champion = None
        final_games = []
        if len(region_champions) == 4:
            champion, final_games = simulate_final_four_and_championship(region_champions,
                                                                         use_analytics=use_analytics)
            all_games.extend(final_games)

        # Build our final result dictionary
        sim_result = {
            'simulation_number':  sim_num + 1,
            'region_results':     region_results,
            'region_champions':   region_champions,
            'champion':           champion,  # dict with 'Team','Seed'
            'all_games':          all_games
        }
        all_sim_results.append(sim_result)

    return all_sim_results

def run_tournament_simulation(num_sims=100):
    """
    Wrapper that:
      1) Prepares bracket from df_main
      2) Runs N simulations
      3) Returns a dictionary of aggregated results:
         { 'Champion': {team->pct}, 'Round of 64':..., etc. }
    """
    bracket = prepare_tournament_data(df_main)
    if not bracket:
        return {}
    return simulate_tournament(bracket, num_simulations=num_sims)

# def display_tournament_simulation_results(simulation_results, num_teams=10):
#     st.markdown("### 🏀 Tournament Simulation Results 🏀")
    
#     # Display each round's results in a clean, visually appealing way
#     rounds = [
#         ('Round of 64', '⭐'),
#         ('Round of 32', '⭐⭐'),
#         ('Sweet 16', '⭐⭐⭐'),
#         ('Elite 8', '⭐⭐⭐⭐'),
#         ('Final Four', '⭐⭐⭐⭐⭐'),
#         ('Championship', '🏆'),
#         ('Champion', '👑')
#     ]
    
#     # Custom CSS to arrange rounds in a more visually appealing way
#     st.markdown("""
#     <style>
#     .tournament-rounds {
#         display: flex;
#         flex-wrap: wrap;
#         justify-content: space-between;
#         gap: 20px;
#     }
#     .round-card {
#         background-color: #f8f9fa;
#         border-radius: 10px;
#         padding: 15px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         flex: 1;
#         min-width: 250px;
#         margin-bottom: 20px;
#     }
#     .round-title {
#         text-align: center;
#         font-weight: bold;
#         margin-bottom: 10px;
#         border-bottom: 2px solid #1E90FF;
#         padding-bottom: 5px;
#     }
#     .team-probability {
#         display: flex;
#         justify-content: space-between;
#         margin: 5px 0;
#         padding: 3px 0;
#         border-bottom: 1px solid #e0e0e0;
#     }
#     .team-name {
#         font-weight: 500;
#     }
#     .probability {
#         font-weight: bold;
#         color: #1E90FF;
#     }
#     .champion-card {
#         background: linear-gradient(135deg, #1E90FF 0%, #4169E1 100%);
#         color: white;
#     }
#     .champion-title {
#         border-bottom: 2px solid white;
#     }
#     .champion-probability {
#         border-bottom: 1px solid rgba(255,255,255,0.3);
#     }
#     .champion-name {
#         font-weight: bold;
#     }
#     .champion-percent {
#         font-weight: bold;
#         color: #FFD700;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     # Start the tournament rounds container
#     st.markdown('<div class="tournament-rounds">', unsafe_allow_html=True)
    
#     # Display each round
#     for round_name, icon in rounds:
#         if round_name == 'Champion':
#             card_class = 'round-card champion-card'
#             title_class = 'round-title champion-title'
#             team_class = 'team-probability champion-probability'
#             name_class = 'team-name champion-name'
#             prob_class = 'probability champion-percent'
#         else:
#             card_class = 'round-card'
#             title_class = 'round-title'
#             team_class = 'team-probability'
#             name_class = 'team-name'
#             prob_class = 'probability'
        
#         st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
#         st.markdown(f'<div class="{title_class}">{icon} {round_name} {icon}</div>', unsafe_allow_html=True)
        
#         # Get the top teams for this round
#         round_results = simulation_results.get(round_name, {})
#         top_teams = sorted(round_results.items(), key=lambda x: x[1], reverse=True)[:num_teams]
        
#         for team, probability in top_teams:
#             st.markdown(
#                 f'<div class="{team_class}"><span class="{name_class}">{team}</span>'
#                 f'<span class="{prob_class}">{probability:.1f}%</span></div>',
#                 unsafe_allow_html=True
#             )
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # Close the tournament rounds container
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Add a summary visualization using Plotly
#     st.markdown("### 📊 Championship Odds Visualization")
    
#     champion_results = simulation_results.get('Champion', {})
#     top_champions = sorted(champion_results.items(), key=lambda x: x[1], reverse=True)[:15]
    
#     champion_teams = [team for team, _ in top_champions]
#     champion_probs = [prob for _, prob in top_champions]
    
#     # Create a visually appealing bar chart with team colors
#     fig = px.bar(
#         x=champion_probs,
#         y=champion_teams,
#         orientation='h',
#         labels={'x': 'Championship Probability (%)', 'y': 'Team'},
#         title='Top 15 Teams Most Likely to Win the 2025 NCAA Tournament',
#         color=champion_probs,
#         color_continuous_scale='Blues'
#     )
    
#     fig.update_layout(
#         height=600,
#         xaxis_title='Championship Probability (%)',
#         yaxis_title='',
#         yaxis={'categoryorder': 'total ascending'},
#         plot_bgcolor='white',
#         font={'family': 'Arial', 'size': 14},
#         margin={'l': 0, 'r': 10, 't': 50, 'b': 0},
#         coloraxis_showscale=False
#     )
    
#     fig.update_traces(
#         marker_line_color='rgb(8,48,107)',
#         marker_line_width=1.5,
#         opacity=0.8
#     )
    
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Add region-specific breakdowns
#     st.markdown("### 🏆 Regional Champions Analysis")
    
#     # Create a 2x2 grid for the four regions
#     col1, col2 = st.columns(2)
#     col3, col4 = st.columns(2)
    
#     champion_results = champ_df.set_index("Team")["Championship_Probability"].to_dict()

#     data = prepare_tournament_data()
#     if not data:
#         st.error("No bracket data found.")
#         st.stop()

#     actual_bracket = data["region_teams"]  # e.g. { "West":[...], "East":[...], ... }

#     regions = ["West","East","South","Midwest"]
#     for region in regions:
#         st.subheader(f"{region} Region")
#         region_teams = {}
#         for team_name, prob in champion_results.items():
#             for team_dict in actual_bracket.get(region, []):
#                 if team_dict["Team"] == team_name:
#                     region_teams[team_name] = prob
#                     break

#         st.write("Teams & Champ Probability:", region_teams)
            
            # top_region_teams = sorted(region_teams.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # if top_region_teams:
            #     region_fig = px.pie(
            #         values=[prob for _, prob in top_region_teams],
            #         names=[team for team, _ in top_region_teams],
            #         title=f"Top 5 {region} Region Champions",
            #         color_discrete_sequence=px.colors.sequential.Blues
            #     )
                
            #     region_fig.update_traces(
            #         textposition='inside',
            #         textinfo='percent+label',
            #         marker=dict(line=dict(color='#FFFFFF', width=2))
            #     )
                
            #     region_fig.update_layout(
            #         height=300,
            #         margin={'l': 10, 'r': 10, 't': 30, 'b': 10}
            #     )
                
            #     st.plotly_chart(region_fig, use_container_width=True)
            # else:
            #     st.markdown("*No teams from this region in the top championship contenders.*")

def display_simulation_results(single_run_logs):
    """
    Renders a single-run game log in an advanced, visually appealing layout.
    Expects single_run_logs to be a list of dictionaries, each with:
      {
        'round': e.g. "Round of 64",
        'region': e.g. "East",
        'matchup': e.g. "TeamA (3) vs TeamB (14)",
        'winner': e.g. "TeamB (14)",
        'is_upset': e.g. "UPSET" or ""
      }
    """

    if not single_run_logs:
        st.warning("No single-run logs to display.")
        return

    # Define the round order and icon to display
    rounds = [
        ("Round of 64", "⭐"),
        ("Round of 32", "⭐⭐"),
        ("Sweet 16", "⭐⭐⭐"),
        ("Elite 8", "⭐⭐⭐⭐"),
        ("Final Four", "⭐⭐⭐⭐⭐"),
        ("Championship", "🏆"),
    ]
    round_order = [r[0] for r in rounds]

    # Convert logs to a DataFrame for sorting
    df = pd.DataFrame(single_run_logs)
    df["Round_idx"] = df["round"].apply(lambda r: round_order.index(r) if r in round_order else 999)
    df.sort_values(["Round_idx", "region"], inplace=True)
    df.drop(columns=["Round_idx"], inplace=True)

    # Advanced CSS styling
    st.markdown(
        """
        <style>
        .single-sim-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            gap: 20px;
        }
        .round-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 270px;
            margin-bottom: 20px;
        }
        .round-title {
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
            border-bottom: 2px solid #1E90FF;
            padding-bottom: 5px;
            font-size: 1.1rem;
        }
        .match-row {
            margin: 6px 0;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 4px;
        }
        .upset {
            color: #FF4500; /* OrangeRed for upsets */
            font-weight: bold;
            margin-left: 8px;
        }
        .region-label {
            font-size: 0.9rem;
            color: #888;
            margin-right: 8px;
        }
        .matchup-teams {
            font-weight: 500;
        }
        .match-winner {
            margin-left: 10px;
            font-style: italic;
            color: #1E90FF;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="single-sim-container">', unsafe_allow_html=True)

    # Render each round as a card
    for round_name, round_icon in rounds:
        # Subset the DataFrame for this round
        subset = df[df["round"] == round_name]
        if subset.empty:
            continue

        st.markdown(f'<div class="round-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="round-title">{round_icon} {round_name} {round_icon}</div>',
            unsafe_allow_html=True
        )

        # Display each game in this round
        for _, row in subset.iterrows():
            upset_html = f'<span class="upset">{row["is_upset"]}</span>' if row["is_upset"] else ""
            st.markdown(
                f"""
                <div class="match-row">
                    <span class="region-label">{row["region"]} Region</span><br>
                    <span class="matchup-teams">{row["matchup"]}</span>
                    <span class="match-winner">→ {row["winner"]}</span>
                    {upset_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# def display_simulation_results(sim_results, st_container):
#     """
#     Displays the round-by-round game logs for a single simulation’s outcome
#     so you see each Round of 64, Round of 32, etc. in order.
#     """
#     if not sim_results:
#         st_container.warning("No simulation results available.")
#         return

#     # Grab the first simulation for demonstration
#     sim_result = sim_results[0]
#     champion   = sim_result.get('champion')
#     if champion:
#         st_container.success(f"🏆 Tournament Champion: {champion['Team']} (Seed {champion['Seed']})")

#     all_games = sim_result.get('all_games', [])
#     if not all_games:
#         st_container.warning("No game data found for this simulation.")
#         return

#     round_order_map = {
#         "Round of 64":    1,
#         "Round of 32":    2,
#         "Sweet 16":       3,
#         "Elite 8":        4,
#         "Final Four":     5,
#         "Championship":   6
#     }

#     df_all = pd.DataFrame([{
#         "Round Sort": round_order_map.get(g["round_name"], 99),
#         "Round":      g["round_name"],
#         "Matchup":    f"{g['team1']} ({g['seed1']}) vs {g['team2']} ({g['seed2']})",
#         "Winner":     f"{g['winner']} ({g['winner_seed']})",
#         "WinProb":    g["win_prob"],
#         "Upset":      "⚠️ UPSET" if (g['winner_seed'] > min(g['seed1'], g['seed2'])) else "",
#         "Region":     g["region"]
#     } for g in all_games])

#     df_all.sort_values(by="Round Sort", inplace=True)
#     df_all.drop(columns="Round Sort", inplace=True)

#     unique_rounds = df_all["Round"].unique()

#     for r_name in unique_rounds:
#         subset = df_all[df_all["Round"] == r_name].copy()
#         # Convert probability to 0–100 for readability
#         subset["WinProb"] = subset["WinProb"].apply(lambda x: f"{x*100:.1f}%")

#         st_container.subheader(r_name)
#         # You already have advanced stylers set up; you can do minimal:
#         subset_renamed = subset.rename(columns={
#             "Matchup":"Matchup",
#             "Winner":"Winner",
#             "WinProb":"Win Probability",
#             "Upset":"Upset",
#             "Region":"Region"
#         })
#         # Just show with .to_html
#         st_container.table(subset_renamed)

# def run_tournament_simulation(num_simulations=100, use_analytics=True):
#     """
#     Runs multiple bracket simulations & aggregates:
#       - champion probabilities
#       - region champion frequencies
#       - upset percentages by round
#     Then returns a dict with those data frames/series for final display.
#     """
#     all_sims = run_simulation(use_analytics=use_analytics, simulations=num_simulations)
#     valid    = [r for r in all_sims if r.get('champion') is not None]
#     if not valid:
#         return {
#             'champion_probabilities': pd.DataFrame(),
#             'region_probabilities':   pd.DataFrame(),
#             'upset_pct_aggregated':   pd.Series(dtype=float),
#             'total_simulations':      0
#         }

#     # (1) Championship counts
#     champions = {}
#     for simres in valid:
#         c = simres['champion']
#         if not c: 
#             continue
#         cname = c['Team']
#         seed  = c['Seed']
#         if cname not in champions:
#             champions[cname] = {'team': cname, 'seed': seed, 'count': 0}
#         champions[cname]['count'] += 1

#     champ_data = []
#     total_valid = len(valid)
#     for tm, data in champions.items():
#         champ_data.append({
#             'Team':                   data['team'],
#             'Seed':                   data['seed'],
#             'Championship_Count':     data['count'],
#             'Championship_Probability': data['count']/total_valid
#         })
#     champ_df = pd.DataFrame(champ_data).sort_values('Championship_Count', ascending=False)

#     # (2) Region champion frequencies
#     region_champs = {}
#     for simres in valid:
#         for reg, champ_dict in simres.get('region_champions', {}).items():
#             if reg not in region_champs:
#                 region_champs[reg] = {}
#             tname = champ_dict['Team']
#             region_champs[reg][tname] = region_champs[reg].get(tname, 0) + 1

#     region_data = []
#     for reg, tdict in region_champs.items():
#         for tname, ccount in tdict.items():
#             region_data.append({
#                 'Region':      reg,
#                 'Team':        tname,
#                 'Count':       ccount,
#                 'Probability': ccount / num_simulations
#             })
#     region_df = pd.DataFrame(region_data).sort_values(['Region','Count'], ascending=[True,False])

#     # (3) Upset analysis (how many upsets per round)
#     all_games = []
#     for simres in all_sims:
#         all_games.extend(simres.get('all_games', []))
#     upsets_by_round = {}
#     games_by_round  = {}
#     for g in all_games:
#         rnd = g.get('round_name','Unknown')
#         games_by_round[rnd] = games_by_round.get(rnd, 0) + 1
#         s1, s2 = g['seed1'], g['seed2']
#         wseed  = g['winner_seed']
#         # if wseed > min(s1, s2), that's an upset
#         if wseed > min(s1, s2):
#             upsets_by_round[rnd] = upsets_by_round.get(rnd, 0) + 1
#     upset_pct = {}
#     for rkey in games_by_round:
#         if games_by_round[rkey] > 0:
#             ucount = upsets_by_round.get(rkey, 0)
#             upset_pct[rkey] = 100.0 * ucount / games_by_round[rkey]

#     upset_series = pd.Series(upset_pct).sort_index()

#     return {
#         'champion_probabilities': champ_df.reset_index(drop=True),
#         'region_probabilities':   region_df,
#         'upset_pct_aggregated':   upset_series,
#         'total_simulations':      num_simulations
#     }

def get_bracket_matchups():
    """Return the bracket structure for each round, by region, plus final four/champ."""
    round_64 = {
        'West':    [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)],
        'East':    [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)],
        'South':   [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)],
        'Midwest': [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
    }
    round_32 = {
        'West':    [(0,1),(2,3),(4,5),(6,7)],
        'East':    [(0,1),(2,3),(4,5),(6,7)],
        'South':   [(0,1),(2,3),(4,5),(6,7)],
        'Midwest': [(0,1),(2,3),(4,5),(6,7)]
    }
    sweet_16 = {
        'West':    [(0,1),(2,3)],
        'East':    [(0,1),(2,3)],
        'South':   [(0,1),(2,3)],
        'Midwest': [(0,1),(2,3)]
    }
    elite_8 = {
        'West':    [(0,1)],
        'East':    [(0,1)],
        'South':   [(0,1)],
        'Midwest': [(0,1)]
    }
    final_four   = [(0,1), (2,3)]  # West/East, South/Midwest
    championship = [(0,1)]
    return round_64, round_32, sweet_16, elite_8, final_four, championship

def run_simulation_once(df):
    """
    Run exactly one bracket simulation, returning a list of game logs with
    (round, region, matchup, winner, upset, etc.) so we can see details.
    """
    from collections import OrderedDict

    r64, r32, s16, e8, f4, champ = get_bracket_matchups()
    bracket = prepare_tournament_data(df)
    if not bracket:
        return []

    game_logs = []
    current = {r: [copy.deepcopy(t) for t in bracket[r]] for r in bracket}

    # Helper to record a single result line
    def record_game(rnd_name, region, tA, tB, w):
        upset = "UPSET" if w['seed']>min(tA['seed'],tB['seed']) else ""
        return {
            "round":   rnd_name,
            "region":  region,
            "matchup": f"{tA['team']} ({tA['seed']}) vs {tB['team']} ({tB['seed']})",
            "winner":  f"{w['team']} ({w['seed']})",
            "is_upset": upset
        }

    # Round of 64
    r64_winners = {}
    for region, matchups in r64.items():
        winners = []
        for (s1,s2) in matchups:
            tA = next(x for x in current[region] if x['seed']==s1)
            tB = next(x for x in current[region] if x['seed']==s2)
            w  = simulate_game(tA,tB)
            winners.append(w)
            game_logs.append(record_game("Round of 64", region, tA,tB, w))
        r64_winners[region] = winners

    # Round of 32
    r32_winners = {}
    for region, pairs in r32.items():
        winners = []
        region_list = r64_winners[region]
        for (i,j) in pairs:
            w = simulate_game(region_list[i], region_list[j])
            winners.append(w)
            game_logs.append(record_game("Round of 32", region, region_list[i],region_list[j], w))
        r32_winners[region] = winners

    # Sweet 16
    s16_winners = {}
    for region, pairs in s16.items():
        winners = []
        region_list = r32_winners[region]
        for (i,j) in pairs:
            w = simulate_game(region_list[i], region_list[j])
            winners.append(w)
            game_logs.append(record_game("Sweet 16", region, region_list[i],region_list[j], w))
        s16_winners[region] = winners

    # Elite 8
    e8_finalists = []
    for region, pairs in e8.items():
        region_list = s16_winners[region]
        for (i,j) in pairs:
            w = simulate_game(region_list[i], region_list[j])
            e8_finalists.append( (region, w) )
            game_logs.append(record_game("Elite 8", region, region_list[i],region_list[j], w))

    # Region champions
    region_champs = {}
    region_order = ['West','East','South','Midwest']
    for (reg, champ_dict) in e8_finalists:
        region_champs[reg] = champ_dict

    # Final Four
    ff_winners = []
    for (idxA, idxB) in f4:
        rA = region_order[idxA]
        rB = region_order[idxB]
        w = simulate_game(region_champs[rA], region_champs[rB])
        ff_winners.append(w)
        game_logs.append(record_game("Final Four", "National Semifinal", region_champs[rA], region_champs[rB], w))

    # Championship
    champion = None
    for (i,j) in champ:
        champion = simulate_game(ff_winners[i], ff_winners[j])
        game_logs.append(record_game("Championship", "National Final", ff_winners[i], ff_winners[j], champion))

    return game_logs

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
            color_continuous_scale=RdYlGn,
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
            color_continuous_scale=RdYlGn,
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
st.caption(":primary[_Cure your bracket brain and propel yourself up the leaderboards by exploring the tabs below:_]")

tab_home, tab_team_reports, tab_radar, tab_regions, tab_team, tab_conf, tab_pred = st.tabs(["🏀 HOME",  #🌐
                                                                          "📋 TEAM REPORTS", 
                                                                          "🕸️ RADAR CHARTS", #📡🧭
                                                                          "🔥 REGIONAL HEATMAPS", #🌡️📍
                                                                          "📊 TEAM METRICS", #📈📋📜📰📅
                                                                          "🏆 CONFERENCE STATS", #🏅
                                                                          "🔮 PREDICTIONS"]) #🎱❓✅❌ ⚙️

# --- Home Tab ---
with tab_home:
    st.subheader(":primary[NCAAM BASKETBALL CONFERENCE TREEMAP]", divider='grey')
    st.caption(":green[_DATA AS OF: 3/18/2025_]")
    treemap = create_treemap(df_main_notnull)
    if treemap is not None:
        st.plotly_chart(treemap, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
    
    selected_team = st.selectbox(
        ":green[_SELECT A TEAM:_]",
        options=[""] + sorted(df_main["TM_KP"].dropna().unique().tolist()),
        index=0,
        key="select_team_home"
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

                # "Interpretive Insights" block - provides a short textual breakdown of how team compares relative to NCAA
                radar_metrics = get_default_metrics()  # e.g. ['AVG MARGIN','KP_AdjEM','OFF EFF','DEF EFF','AST/TO%','STOCKS-TOV/GM']
                existing_metrics = [m for m in radar_metrics if m in team_data.columns]
                if existing_metrics:
                    with st.expander("📊 Interpretive Insights"):
                        insights = []
                        for metric in get_default_metrics():
                            if metric in team_data.columns:
                                mean_val = t_avgs[metric]
                                std_val = max(t_stdevs[metric], 1e-6)
                                team_val = team_data.iloc[0][metric]
                                z = (team_val - mean_val) / std_val
                                if metric in ["DEF EFF", "TO/GM"]:
                                    z = -z
                                if abs(z) < 0.3:
                                    insights.append(f"**{metric}** | Near NCAA average.")
                                elif z >= 1.0:
                                    insights.append(f"**{metric}** | Clear strength.")
                                elif 0.3 <= z < 1.0:
                                    insights.append(f"**{metric}** | Above NCAA average.")
                                elif -1.0 < z <= -0.3:
                                    insights.append(f"**{metric}** | Below NCAA average.")
                                else:
                                    insights.append(f"**{metric}** | Notable weakness.")

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

                # Format numeric columns
                def _fmt(v):
                    if isinstance(v, float):
                        return f"{v:.2f}"
                    else:
                        return str(v)

                detail_df["Value"] = detail_df["Value"].apply(_fmt)

                # Convert to a Styler for advanced CSS
                detail_styler = (
                    detail_df.style
                    .set_properties(**{"text-align": "center"})
                    .set_table_styles([
                        {
                            "selector": "th",
                            "props": [
                                ("background-color", "#0360CE"),
                                ("color", "white"),
                                ("font-weight", "bold"),
                                ("text-align", "center"),
                                ("padding", "6px 12px"),
                                ("border", "1px solid #222")
                            ]
                        },
                        {
                            "selector": "td",
                            "props": [
                                ("text-align", "center"),
                                ("border", "1px solid #ddd"),
                                ("padding", "5px 10px")
                            ]
                        },
                        {
                            "selector": "table",
                            "props": [
                                ("border-collapse", "collapse"),
                                ("border", "2px solid #222"),
                                ("border-radius", "8px"),
                                ("overflow", "hidden"),
                                ("box-shadow", "0 4px 12px rgba(0, 0, 0, 0.1)")
                            ]
                        },
                    ])
                )

                st.markdown(detail_styler.to_html(), unsafe_allow_html=True)

    
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
                "MEAN STOCKS-TOV/GM": "{:.1f}",
            })
            .background_gradient(cmap="RdYlGn", subset=[
                "# TEAMS", "# BIDS", 
                "MEAN AdjEM", "MIN AdjEM", "MAX AdjEM",
                #"AVG TR_OEff_25",
                "MEAN WIN %", "MEAN AVG MARGIN",
                "MEAN eFG%",
                "MEAN AST/TO%",
                "MEAN STOCKS/GM", "MEAN STOCKS-TOV/GM",
                
            ])
            .background_gradient(cmap="RdYlGn_r", subset=["MEAN SEED_25", "MEAN NET_25",
                                                          #"AVG TR_DEff_25",
                                                          ])
            .set_table_styles(detailed_table_styles)
        )

        st.markdown(styled_conf_stats.to_html(escape=False), unsafe_allow_html=True)

with tab_team_reports:
    st.header(":primary[TEAM REPORTS]")
    st.caption(":green[_DATA AS OF: 3/18/2025_]")
    # Allow team selection – similar to the Home tab approach
    selected_team_reports = st.selectbox(
        ":green[_SELECT A TEAM:_]",
        options=[""] + sorted(df_main["TM_KP"].dropna().unique().tolist()),
        index=0,
        key="select_team_reports"  # unique key for this selectbox
    )
    if selected_team_reports:
        team_data = df_main[df_main["TM_KP"] == selected_team_reports].copy()
        if not team_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {selected_team_reports}")
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
                with st.expander("📊 Interpretive Insights"):
                    insights = []
                    for metric in get_default_metrics():
                        if metric in team_data.columns:
                            mean_val = t_avgs[metric]
                            std_val = max(t_stdevs[metric], 1e-6)
                            team_val = team_data.iloc[0][metric]
                            z = (team_val - mean_val) / std_val
                            if metric in ["DEF EFF", "TO/GM"]:
                                z = -z
                            if abs(z) < 0.3:
                                insights.append(f"**{metric}** | Near NCAA average.")
                            elif z >= 1.0:
                                insights.append(f"**{metric}** | Clear strength.")
                            elif 0.3 <= z < 1.0:
                                insights.append(f"**{metric}** | Above NCAA average.")
                            elif -1.0 < z <= -0.3:
                                insights.append(f"**{metric}** | Below NCAA average.")
                            else:
                                insights.append(f"**{metric}** | Notable weakness.")
                    st.markdown("**Team Metric Highlights:**")
                    for line in insights:
                        st.write(f"- {line}")
            with col2:
                radar_fig = create_radar_chart([selected_team_reports], df_main)
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


# --- Radar Charts Tab ---
with tab_radar:
    st.header(":primary[REGIONAL RADAR CHARTS]")
    st.caption(":green[_DATA AS OF: 3/18/2025_]")
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
    st.header(":primary[REGIONAL HEATMAPS]")
    st.caption(":green[_DATA AS OF: 3/18/2025_]")
    #st.write("REGIONAL HEATFRAMES (W, X, Y, Z)")
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
        "KP_Rank": "RdYlGn_r",
        "WIN_25": "RdYlGn",
        "LOSS_25": "RdYlGn_r",
        "KP_AdjEM": "RdYlGn",
        "KP_SOS_AdjEM": "RdYlGn_r",
        "OFF EFF": "RdYlGn",
        "DEF EFF": "RdYlGn_r",
        "AVG MARGIN": "RdYlGn",
        "eFG%": "RdYlGn",
        "TS%": "RdYlGn",
        "OPP TS%": "RdYlGn_r",
        "AST/TO%": "RdYlGn",
        "STOCKS/GM": "RdYlGn",
        "WIN% ALL GM": "RdYlGn",
        "WIN% CLOSE GM": "RdYlGn",
        "NET_25": "RdYlGn_r",
        "SEED_25": "RdYlGn_r",
        "KP_AdjO": "RdYlGn",
        "KP_AdjD": "RdYlGn_r",
        "PTS/GM": "RdYlGn",
        "OPP PTS/GM": "RdYlGn",
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
    st.header(":primary[CONFERENCE COMPARISON]")
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
    st.header(":primary[TEAM METRICS COMPARISON]")
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
    st.header(":primary[BRACKET SIMULATION]")

    show_logs = st.checkbox("Show Detailed Single-Sim Logs", value=True)
    if st.button("Run Bracket Simulation"):
        with st.spinner("Simulating..."):
            # (1) Aggregated results from your multi-run simulation
            aggregated = run_tournament_simulation(num_sims=100)

            # (2) Single-run logs (a single bracket outcome)
            single_run = run_simulation_once(df_main)

        st.success("Simulation complete!")

        # 1) If requested, show single-run logs first
        if show_logs and single_run:
            st.subheader("Detailed Round-by-Round (Single Simulation)")
            display_simulation_results(single_run)
        else:
            st.info("Single-run logs hidden. Check box above to display them.")

        # 2) Now show aggregated results
        if not aggregated:
            st.error("No aggregated results. Check bracket data or code.")
            st.stop()

        st.subheader("Aggregated Simulation Results (100 sims)")

        # A) Champion probabilities turned into a styled DataFrame
        champ_probs = aggregated.get("Champion", {})
        if not champ_probs:
            st.warning("No champion probabilities found in aggregator.")
            st.stop()

        # Build a table with columns: [Team,Champ%,Seed,Region,Conference,KP_AdjEM,NET_25]
        champion_items = sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)
        data_rows = []
        for team, pct in champion_items:
            row = {"Team": team, "Champ%": pct}
            subset = df_main[df_main["TM_KP"] == team]
            if not subset.empty:
                row["Conference"] = subset["CONFERENCE"].iloc[0] if "CONFERENCE" in subset.columns else ""
                row["Seed"] = int(subset["SEED_25"].iloc[0]) if ("SEED_25" in subset.columns and not pd.isna(subset["SEED_25"].iloc[0])) else ""
                row["Region"] = subset["REGION_25"].iloc[0] if "REGION_25" in subset.columns else ""
                row["KP_AdjEM"] = subset["KP_AdjEM"].iloc[0] if "KP_AdjEM" in subset.columns else None
                row["NET_25"]   = subset["NET_25"].iloc[0]   if "NET_25" in subset.columns else None
            data_rows.append(row)

        champion_df = pd.DataFrame(data_rows)
        champion_df["Champ%"] = champion_df["Champ%"].round(1)
        champion_df.rename(columns={"Champ%": "Champ Probability (%)"}, inplace=True)

        # Reorder columns
        champion_df = champion_df[["Team", "Champ Probability (%)", "Seed", "Region", "Conference", "KP_AdjEM", "NET_25"]]

        # Create a Styler
        champion_styler = (
            champion_df.style
            .format({
                "Champ Probability (%)": "{:.1f}",
                "KP_AdjEM": "{:.1f}",
                "NET_25": "{:.0f}"
            })
            .background_gradient(
                cmap="RdYlGn", 
                subset=["Champ Probability (%)", "KP_AdjEM"]
            )
            .set_properties(**{"text-align": "center"})
            .set_table_styles([
                {
                    "selector": "table",
                    "props": [
                        ("border-collapse", "collapse"),
                        ("border", "2px solid #222"),
                        ("border-radius", "8px"),
                        ("overflow", "hidden"),
                        ("box-shadow", "0 4px 12px rgba(0, 0, 0, 0.1)")
                    ]
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#0360CE"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("padding", "8px 10px"),
                        ("border", "1px solid #222"),
                        ("font-size", "13px")
                    ]
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),
                        ("padding", "5px 10px"),
                        ("border", "1px solid #ddd")
                    ]
                },
            ])
        )

        st.markdown("##### Top Likely Champions (Styled Table)")
        st.markdown(champion_styler.to_html(), unsafe_allow_html=True)

        # Optionally, show raw text summary
        st.write("**Raw Summary**:")
        for row in data_rows[:15]:
            st.write(f"{row['Team']}: {row['Champ%']:.1f}%")


        # B) 2×2 subplot for region winners (if aggregator has something like aggregator["Region"])
        region_probs = aggregated.get("Region", None)
        if region_probs is None:
            st.info("No region-level winner data found in aggregator. Skipping 2×2 subplot.")
        else:
            st.markdown("##### Regional Championship Probabilities")
            from plotly.subplots import make_subplots
            fig_region = make_subplots(rows=2, cols=2, subplot_titles=["West", "East", "South", "Midwest"])

            row_col_map = {"West": (1,1), "East": (1,2), "South": (2,1), "Midwest": (2,2)}
            for region_name in ["West","East","South","Midwest"]:
                if region_name not in region_probs:
                    continue
                items = sorted(region_probs[region_name].items(), key=lambda x: x[1], reverse=True)[:8]
                x_vals = [itm[0] for itm in items]
                y_vals = [itm[1] for itm in items]
                (r, c) = row_col_map[region_name]
                fig_region.add_trace(
                    go.Bar(x=x_vals, y=y_vals, name=region_name,
                           text=[f"{v:.1f}%" for v in y_vals],
                           textposition="outside", marker_color="steelblue"),
                    row=r, col=c
                )
                fig_region.update_xaxes(tickangle=-45, row=r, col=c)
                if y_vals:
                    fig_region.update_yaxes(range=[0, max(y_vals)*1.2], row=r, col=c)

            fig_region.update_layout(
                template="plotly_dark",
                height=600,
                title="Regional Championship Odds",
                showlegend=False,
                margin=dict(l=50, r=50, t=60, b=60)
            )
            st.plotly_chart(fig_region, use_container_width=True)

        # C) 1×1 bar chart for championship probabilities
        st.markdown("##### Championship Probabilities (Bar Chart)")
        top_champs = sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)[:12]
        fig_champ = go.Figure()
        fig_champ.add_trace(
            go.Bar(
                x=[tc[0] for tc in top_champs],
                y=[tc[1] for tc in top_champs],
                text=[f"{tc[1]:.1f}%" for tc in top_champs],
                textposition="outside",
                marker_color="tomato"
            )
        )
        fig_champ.update_layout(
            template="plotly_dark",
            title="Championship Probabilities (Top 12)",
            xaxis=dict(tickangle=-45),
            yaxis=dict(range=[0, max([tc[1] for tc in top_champs])*1.15]),
            showlegend=False,
            margin=dict(l=20, r=20, t=80, b=60),
            height=450
        )
        st.plotly_chart(fig_champ, use_container_width=True)
    else:
        st.info("Run the simulation to see results.")


if FinalFour25_logo:
    st.image(FinalFour25_logo, width=750)

st.markdown("---")
st.caption("Python code framework available on [GitHub](https://github.com/nehat312/march-madness-2025)")
st.caption("DATA SOURCED FROM: [TeamRankings](https://www.teamrankings.com/ncaa-basketball/ranking/predictive-by-other/), [KenPom](https://kenpom.com/)")
st.stop()


## ----- EXTRA ----- ##

# with tab_pred:
#     st.header("Bracket Simulation")

#     show_detailed_logs = st.checkbox("Show Detailed Single-Sim Logs", value=True)
#     if st.button("Run Bracket Simulation"):
#         with st.spinner("Simulating..."):
#             # 1) Aggregated
#             aggregated = run_tournament_simulation(num_simulations=100, use_analytics=True)
#             # 2) Single example simulation (for logs)
#             single_run = run_simulation(use_analytics=True, simulations=1)

#         st.success("Simulation complete!")

#         # Display single-run logs *first* so they're 'bumped up' above summary
#         if show_detailed_logs:
#             st.subheader("Round-by-Round Results (Single Simulation)")
#             display_simulation_results(single_run, st)

#         # Now show aggregated summary, champion odds, upset rates, etc.
#         st.subheader("Aggregated Simulation Results")
#         champ_df = aggregated['champion_probabilities']
#         st.dataframe(champ_df)

# with tab_pred:
#     st.header("Bracket Simulation")
    
#     # Create two columns for simulation controls and bracket visualization
#     sim_col, viz_col = st.columns([3, 2])
    
#     with sim_col:
#         # Checkbox to show detailed logs
#         show_detailed_logs = st.checkbox("Show Detailed Logs (Single Simulation Recommended)", value=True)
#         st.write("Run the tournament simulation across multiple iterations to see aggregated outcomes.")
    
#         # Initialize session state variable if needed
#         if 'simulation_results' not in st.session_state:
#             st.session_state.simulation_results = {}
    
#         # Run simulation when button is clicked
#         if st.button("Run Bracket Simulation", key="btn_run_bracket"):
#             with st.spinner("Simulating tournament..."):
#                 # 1) Aggregated stats
#                 aggregated_analysis = run_tournament_simulation(num_simulations=100, use_analytics=True)

#                 # 2) Single or multi-simulation full detail
#                 all_sim_results = run_simulation(use_analytics=True, simulations=1)

#                 # 3) If you want to fix the Final Four for each simulation in that list
#                 for sim_result in all_sim_results:
#                     if sim_result.get('champion') is None and len(sim_result.get('region_champions', {})) == 4:
#                         # Perform final four logic
#                         region_champs = list(sim_result['region_champions'].values())
#                         semi1_prob = calculate_win_probability(region_champs[0], region_champs[1])
#                         semi1_winner = region_champs[0] if random.random() < semi1_prob else region_champs[1]
#                         semi2_prob = calculate_win_probability(region_champs[2], region_champs[3])
#                         semi2_winner = region_champs[2] if random.random() < semi2_prob else region_champs[3]
#                         final_prob = calculate_win_probability(semi1_winner, semi2_winner)
#                         champion = semi1_winner if random.random() < final_prob else semi2_winner
#                         sim_result['champion'] = champion

#                 # 4) Store in session_state
#                 st.session_state.simulation_results['aggregated_analysis'] = aggregated_analysis
#                 st.session_state.simulation_results['single_sim_results']  = all_sim_results

#             # Show success
#             st.success("Simulation complete!")
    
#              # 5) Now we display everything from st.session_state
#             #    aggregated_analysis is st.session_state.simulation_results['aggregated_analysis']
#             #    single_sim or multi-sim is st.session_state.simulation_results['single_sim_results']

#             # Display Championship Win Probabilities
#             try:
#                 champ_df = st.session_state.simulation_results['aggregated_analysis']['champion_probabilities']
#                 # style it, show it, etc...
#             except Exception as e:
#                 st.error(f"Error displaying Championship Win Probabilities: {e}")

#             # Regional Win Probabilities
#             try:
#                 region_probs = st.session_state.simulation_results['aggregated_analysis']['region_probabilities']
#                 fig_regional = create_regional_prob_chart(region_probs)
#                 if fig_regional:
#                     st.plotly_chart(fig_regional, use_container_width=True)
#             except Exception as e:
#                 st.error(f"Error displaying Regional Win Probabilities: {e}")

#             # Upset Analysis
#             try:
#                 upset_pct_aggregated = st.session_state.simulation_results['aggregated_analysis']['upset_pct_aggregated']
#                 # etc...
#             except Exception as e:
#                 st.error(f"Error displaying Upset Analysis: {e}")
    
#             # Plot the aggregated results
#             try:
#                 fig_champ, fig_upsets = visualize_aggregated_results(
#                     st.session_state.simulation_results['aggregated_analysis']
#                 )
#                 if fig_champ:
#                     st.plotly_chart(fig_champ, use_container_width=True)
#                 if fig_upsets:
#                     st.plotly_chart(fig_upsets, use_container_width=True)
#             except Exception as e:
#                 st.error(f"Could not generate aggregated visualizations: {e}")

#             # Show single-sim results
#             try:
#                 if st.session_state.simulation_results['single_sim_results']:
#                     st.info("Detailed game outcomes for a single simulation run:")
#                     display_simulation_results(st.session_state.simulation_results['single_sim_results'], st)
#                 else:
#                     st.warning("No single simulation results to display.")
#             except Exception as e:
#                 st.error(f"Error displaying detailed simulation logs: {e}")

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


#     with viz_col:
#         st.subheader("Bracket Visualization")
#         st.markdown("Select a view to explore the tournament teams")
#         viz_type = st.radio("Visualization Type", ["Team Stats"], horizontal=True)
#         if 'TR_df' not in globals():
#             _ = prepare_tournament_data()
#         if viz_type == "Team Stats":
#             all_tourney_teams = TR_df[TR_df['SEED_25'].notna()]['TM_KP'].tolist()
#             selected_team = st.selectbox("Select Team", sorted(all_tourney_teams))
#             team_data = TR_df[TR_df["TM_KP"] == selected_team].iloc[0]
#             create_team_radar(team_data, dark_mode=True)
#             st.markdown("### Key Team Stats")
#             seed_str = f"{int(team_data['SEED_25'])}" if pd.notna(team_data['SEED_25']) else "N/A"
#             region_str = (team_data['REGION_25'] if pd.notna(team_data['REGION_25'])
#                           else (team_data['REG_CODE_25'] if pd.notna(team_data['REG_CODE_25']) else "N/A"))
#             key_stats = {
#                 "Seed": f"{seed_str} ({region_str})",
#                 "Record": f"{team_data['WIN_25']:.0f}-{team_data['LOSS_25']:.0f}",
#                 "NET Rank": f"{int(team_data['NET_25'])}",
#                 "KenPom Rank": f"{int(team_data['KP_Rank'])}",
#                 "KenPom Adj EM": f"{team_data['KP_AdjEM']:.2f}",
#                 "KenPom Adj OEff": f"{team_data['KP_AdjO']:.2f}",
#                 "KenPom Adj DEff": f"{team_data['KP_AdjD']:.2f}",
#                 "TeamRankings OEff": f"{team_data['OFF EFF']:.2f}",
#                 "TeamRankings DEff.": f"{team_data['DEF EFF']:.2f}",
#             } 
#             stat_col1, stat_col2 = st.columns(2)
#             for i, (stat, value) in enumerate(key_stats.items()):
#                 if i % 2 == 0:
#                     stat_col1.metric(stat, value)
#                 else:
#                     stat_col2.metric(stat, value)
#         else:
#             st.markdown("### Bracket Overview")
#             # Placeholder: Additional bracket overview visualizations, etc here.
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