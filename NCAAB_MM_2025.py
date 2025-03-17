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
             "KP_Rank", "NET_25", "SEED_25",
             "KP_AdjEM", "KP_SOS_AdjEM", "OFF EFF", "DEF EFF",
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

# Ensure team label (if "TM_KP" is missing, use index)
if "TM_KP" not in df_main.columns:
    df_main["TM_KP"] = df_main.index

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
                       "BE": BE_logo, "Big South": BSouth_logo, "BSky": BSky_logo, "BW": BWest_logo, "CAA": CAA_logo, "CUSA": CUSA_logo,
                       "Horz": Horizon_logo, "Ivy": Ivy_logo, "MAAC": MAAC_logo, "MAC": MAC_logo, "MEAC": MEAC_logo, "MVC": MVC_logo, "MWC": MWC_logo,
                       "NEC": NEC_logo, "OVC": OVC_logo, "Patriot": Patriot_logo, "SB": SBC_logo, "SEC": SEC_logo, "SoCon": SoCon_logo, "Southland": Southland_logo,
                       "Summit": Summit_logo, "SWAC": SWAC_logo, "WAC": WAC_logo, "WCC": WCC_logo,
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
        ('border-bottom', '2px solid #000000')
    ]
}
detailed_table_styles = [header]

# ----------------------------------------------------------------------------
# Radar Chart Functions
def get_default_metrics():
    """
    Return metrics to be used in radar charts z-score logic.
    """
    return [
        'AVG MARGIN',
        'KP_AdjEM',
        'OFF EFF',
        'DEF EFF',
        'AST/TO%',
        'STOCKS-TOV/GM'
    ]
def compute_tournament_stats(df):
    """
    Compute overall average, std dev for radar metrics used to scale z-scores.
    """
    metrics = get_default_metrics()
    avgs = {m: df[m].mean() for m in metrics if m in df.columns}
    stdevs = {m: df[m].std()  for m in metrics if m in df.columns}
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

def create_radar_chart(selected_teams, full_df):
    metrics = get_default_metrics()
    available_radar_metrics = [m for m in metrics if m in full_df.columns]
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
        rows=row_count,
        cols=col_count,
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.1)"
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor="rgba(0,0,0,0.0)",
        plot_bgcolor="rgba(0,0,0,0.8)"
    )
    fig.update_polars(
        radialaxis=dict(
            tickmode='array',
            tickvals=[0, 2, 4, 6, 8, 10],
            ticktext=['0', '2', '4', '6', '8', '10'],
            tickfont=dict(size=11, family="Arial, sans-serif"),
            showline=False,
            gridcolor='rgba(255,255,255,0.2)'
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
    for idx, team_row in subset.iterrows():
        r = idx // col_count + 1
        c = idx % col_count + 1
        show_legend = (idx == 0)
        conf = team_row['CONFERENCE'] if 'CONFERENCE' in team_row else None
        conf_df = full_df[full_df['CONFERENCE'] == conf] if conf else pd.DataFrame()
        traces = get_radar_traces(team_row, t_avgs, t_stdevs, conf_df, show_legend=show_legend)
        for tr in traces:
            fig.add_trace(tr, row=r, col=c)
        perf_data = compute_performance_text(team_row, t_avgs, t_stdevs)
        polar_idx = (r - 1) * col_count + c
        polar_key = "polar" if polar_idx == 1 else f"polar{polar_idx}"
        if polar_key in fig.layout:
            domain_x = fig.layout[polar_key].domain.x
            domain_y = fig.layout[polar_key].domain.y
            x_annot = domain_x[0] + 0.02
            y_annot = domain_y[1] - 0.02
        else:
            x_annot, y_annot = 0.05, 0.95
        fig.add_annotation(
            x=x_annot,
            y=y_annot,
            xref="paper",
            yref="paper",
            text=f"<b>{perf_data['text']}</b>",
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor={
                "badge-elite": "gold",
                "badge-solid": "#4CAF50",
                "badge-mid": "#2196F3",
                "badge-subpar": "#FF9800",
                "badge-weak": "#F44336"
            }.get(perf_data['class'], "#2196F3"),
            bordercolor="white",
            borderwidth=1,
            borderpad=4,
            opacity=0.9
        )
    return fig

def get_radar_zscores(team_row, t_avgs, t_stdevs, conf_df):
    """
    For a single team, produce three radial vectors in [0..10] scale:
      - Team z-scores
      - National average (always 5)
      - Conference average
    Each metric >5 means 'above average'; <5 means 'below average'.
    """
    metrics = get_default_metrics()
    # Filter out any metric not in the row or missing from the overall stats
    used_metrics = [m for m in metrics if m in team_row and m in t_avgs]
    if not used_metrics:
        return [], [], [], used_metrics

    # Build team z-scores
    z_scores = []
    for m in used_metrics:
        val = team_row[m]
        mean_ = t_avgs[m] if m in t_avgs else 0
        stdev_ = t_stdevs[m] if (m in t_stdevs and t_stdevs[m] > 0) else 1.0

        # If it's a "lower is better" metric, invert the z-score
        if m in ['DEF EFF', 'TO/GM']:
            z = -(val - mean_) / stdev_
        else:
            z = (val - mean_) / stdev_
        z_scores.append(z)

    # Convert z-scores into the [0..10] scale where 5 is average
    scale_factor = 1.5
    team_scaled = [max(0, min(10, 5 + (z * scale_factor))) for z in z_scores]

    # National average is always 5
    ncaam_scaled = [5]*len(team_scaled)

    # Conference average
    conf_scaled = []
    for m in used_metrics:
        if conf_df is not None and m in conf_df.columns:
            conf_val = conf_df[m].mean()
        else:
            conf_val = t_avgs[m]  # fallback to national average if missing
        stdev_ = t_stdevs[m] if (m in t_stdevs and t_stdevs[m] > 0) else 1.0
        if m in ['DEF EFF', 'TO/GM']:
            zc = -(conf_val - t_avgs[m]) / stdev_
        else:
            zc = (conf_val - t_avgs[m]) / stdev_
        val_scaled = max(0, min(10, 5 + (zc * scale_factor)))
        conf_scaled.append(val_scaled)

    return team_scaled, ncaam_scaled, conf_scaled, used_metrics

def create_bracket_radar_grid():
    """Creates a 4x16 grid of radar charts, one for each region and seed"""
    
    # Use the existing dataframe with team seeds and regions
    df = TR_df.copy()
    
    # Ensure we have the necessary columns
    if not all(col in df.columns for col in ['SEED_25', 'REG_SEED_25', 'REG_CODE_25', 'REGION_25']):
        st.error("Required columns for bracket visualization are missing")
        return
    
    # Select teams that are in the tournament
    tourney_teams = df[df['SEED_25'].notna()].copy()
    
    # Create radar chart grid with dark background
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
        
        # Region headers
        cols = st.columns(4)
        regions = ["East", "West", "South", "Midwest"]
        for i, region in enumerate(regions):
            cols[i].markdown(f"<h3 style='text-align:center;color:white;'>{region}</h3>", unsafe_allow_html=True)
        
        # Create radar charts for each region and seed
        for seed in range(1, 17):
            cols = st.columns(4)
            
            for i, region in enumerate(regions):
                # Find team with this seed in this region
                team = tourney_teams[(tourney_teams['REGION_25'] == region) & 
                                    (tourney_teams['REG_SEED_25'] == seed)]
                
                if not team.empty:
                    team = team.iloc[0]
                    
                    # Create radar chart for team
                    with cols[i]:
                        create_team_radar(team, dark_mode=True)
                else:
                    # Empty placeholder if no team matches
                    cols[i].markdown(f"<div style='height:200px;display:flex;align-items:center;justify-content:center;color:white;'>No Team (Seed {seed})</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def create_team_radar(team, dark_mode=True):
    """Creates a radar chart for a single team with proper annotations and color"""
    
    # Get team data
    team_name = team['TM_TR']
    seed = int(team['REG_SEED_25'])
    
    # Calculate Z-scores for key metrics
    metrics = ['TR_OEff_25', 'TR_DEff_25', 'NET_eFG%', 'NET AST/TOV RATIO', 'TTL REB%', 'STOCKS/GM']
    labels = ['Offense', 'Defense', 'Shooting', 'Ball Control', 'Rebounding', 'Stocks']
    
    # Make sure all metrics exist
    available_metrics = [m for m in metrics if m in team.index]
    available_labels = [labels[metrics.index(m)] for m in available_metrics]
    
    if not available_metrics:
        st.markdown(f"<div style='height:200px;text-align:center;color:white;'><p>({seed}) {team_name}</p><p>No metrics available</p></div>", unsafe_allow_html=True)
        return
    
    # Get values and compute Z-scores
    values = []
    for metric in available_metrics:
        if pd.notna(team[metric]):
            # Get all values for this metric to compute z-score
            all_values = TR_df[metric].dropna()
            if len(all_values) > 0:
                mean = all_values.mean()
                std = all_values.std() if all_values.std() > 0 else 1
                z_score = (team[metric] - mean) / std
                # Cap z-scores at -3 to 3 range
                z_score = max(min(z_score, 3), -3)
                # Normalize to 0-100 scale
                norm_value = (z_score + 3) * (100 / 6)
                values.append(norm_value)
            else:
                values.append(50)  # Default if no data
        else:
            values.append(50)  # Default if metric is NA
    
    # Team type categorization
    strengths = []
    if len(values) >= 6:  # If we have all metrics
        if values[0] > 65 and values[2] > 65:  # Good offense and shooting
            strengths.append("Offensive")
        if values[1] > 65 and values[5] > 65:  # Good defense and stocks
            strengths.append("Defensive")
        if values[3] > 65 and values[4] > 65:  # Good ball control and rebounding
            strengths.append("Fundamental")
    
    team_type = " & ".join(strengths) if strengths else "Balanced"
    
    # Color code based on seed
    seed_colors = {
        range(1, 5): "rgba(0, 255, 0, 0.7)",    # Seeds 1-4: Green
        range(5, 9): "rgba(255, 255, 0, 0.7)",  # Seeds 5-8: Yellow
        range(9, 13): "rgba(255, 165, 0, 0.7)", # Seeds 9-12: Orange
        range(13, 17): "rgba(255, 0, 0, 0.7)"   # Seeds 13-16: Red
    }
    
    color = next((c for r, c in seed_colors.items() if seed in r), "rgba(255, 255, 255, 0.7)")
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=available_labels,
        fill='toself',
        fillcolor=color.replace('0.7', '0.3'),
        line=dict(color=color),
        name=team_name
    ))
    
    # Configure layout with dark mode if selected
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
    
    # Add seed and type annotations
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
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

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

# ----------------------------------------------------------------------------
# ------------------ BRACKET SIMULATION FUNCTIONS ------------------
# (Integrated Simulation Code)

# Set up logging for simulation (suppress detailed logs in Streamlit)
sim_logger = logging.getLogger("simulation")
if sim_logger.hasHandlers():
    sim_logger.handlers.clear()
sim_logger.setLevel(logging.WARNING)
sim_handler = logging.StreamHandler()
sim_handler.setLevel(logging.WARNING)
sim_handler.setFormatter(logging.Formatter("%(message)s"))
sim_logger.addHandler(sim_handler)

# ANSI color codes for simulation rounds
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

# For simulation, we use TR_df; here we simply set it equal to df_main.
# (Make sure that df_main contains the necessary columns for simulation.)
TR_df = df_main.copy()

# Prepare simulation teams (top 64 by KP_Rank)
complete_records = TR_df.dropna(subset=['KP_Rank', 'KP_AdjEM', 'OFF EFF', 'DEF EFF']).copy()
df_teams_sorted = complete_records.sort_values(by='KP_Rank').head(64).copy()
df_teams_sorted.reset_index(drop=True, inplace=True)
if 'TM_KP' not in df_teams_sorted.columns:
    df_teams_sorted['TM_KP'] = df_teams_sorted['TM_TR']

# Historical tournament success (example bonus)
df_teams_sorted['TOURNEY_SUCCESS'] = 0.0
for team in ["Duke", "Kentucky", "Kansas", "North Carolina", "Gonzaga", "Michigan St."]:
    if team in df_teams_sorted['TM_KP'].values:
        df_teams_sorted.loc[df_teams_sorted['TM_KP'] == team, 'TOURNEY_SUCCESS'] = 2.0

# Tournament experience bonus (if available)
df_teams_sorted['TOURNEY_EXPERIENCE'] = 0.0
if 'SEED_23' in df_teams_sorted.columns:
    df_teams_sorted.loc[df_teams_sorted['SEED_23'] <= 16, 'TOURNEY_EXPERIENCE'] = 1.0
    df_teams_sorted.loc[df_teams_sorted['SEED_23'] <= 4, 'TOURNEY_EXPERIENCE'] = 2.0

# Assign Regions – simple round-robin assignment
region_names = ['East', 'West', 'South', 'Midwest']
assigned_regions = [region_names[i % 4] for i in range(len(df_teams_sorted))]
df_teams_sorted['Region'] = assigned_regions

# Assign Seeds (1 to 16 within each region)
df_teams_sorted['Seed'] = 0
for reg in region_names:
    region_df = df_teams_sorted[df_teams_sorted['Region'] == reg]
    seeds_for_region = list(range(1, len(region_df) + 1))
    df_teams_sorted.loc[df_teams_sorted['Region'] == reg, 'Seed'] = seeds_for_region

# Build region_teams dictionary
region_teams = {}
for reg in region_names:
    df_reg = df_teams_sorted[df_teams_sorted['Region'] == reg].sort_values('Seed')
    teams_list = df_reg.apply(lambda row: {
        'Team': row['TM_KP'],
        'Seed': int(row['Seed']),
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
        'Region': row['Region']
    }, axis=1).tolist()
    region_teams[reg] = teams_list

def calculate_win_probability(team1, team2):
    team1_off = team1.get('OFF EFF', 1.0) if not pd.isna(team1.get('OFF EFF', 1.0)) else 1.0
    team1_def = team1.get('DEF EFF', 1.0) if not pd.isna(team1.get('DEF EFF', 1.0)) else 1.0
    team2_off = team2.get('OFF EFF', 1.0) if not pd.isna(team2.get('OFF EFF', 1.0)) else 1.0
    team2_def = team2.get('DEF EFF', 1.0) if not pd.isna(team2.get('DEF EFF', 1.0)) else 1.0
    kp_diff = team1['KP_AdjEM'] - team2['KP_AdjEM']
    matchup_adv = (team1_off - team2_def) - (team2_off - team1_def)
    exp_diff = (team1.get('TOURNEY_EXPERIENCE', 0) - team2.get('TOURNEY_EXPERIENCE', 0)) + \
               (team1.get('TOURNEY_SUCCESS', 0) - team2.get('TOURNEY_SUCCESS', 0))
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
    winners = []
    if round_num == 1:
        pairings = [(0, 15), (7, 8), (4, 11), (3, 12), (5, 10), (2, 13), (6, 9), (1, 14)]
    else:
        pairings = [(i, i+1) for i in range(0, len(teams), 2)]
    for i, j in pairings:
        teamA = teams[i]
        teamB = teams[j]
        if use_analytics:
            pA = calculate_win_probability(teamA, teamB)
            pB = 1 - pA
        else:
            diff = teamA['KP_AdjEM'] - teamB['KP_AdjEM']
            pA = 1 / (1 + np.exp(-diff/10))
            pB = 1 - pA
        rand_val = random.random()
        winner = teamA if rand_val < pA else teamB
        winner = winner.copy()
        winner['win_prob'] = pA if winner == teamA else pB
        winners.append(winner)
    return winners

def simulate_region_bracket(teams, region_name, use_analytics=True):
    rounds = {}
    current_round_teams = teams
    num_rounds = int(math.log(len(teams), 2))
    all_games = []
    for r in range(1, num_rounds + 1):
        rounds[r] = current_round_teams
        winners = generate_bracket_round(current_round_teams, r, region_name, use_analytics)
        for i, winner in enumerate(winners):
            if r == 1:
                matchup_idx = [(0, 15), (7, 8), (4, 11), (3, 12), (5, 10), (2, 13), (6, 9), (1, 14)][i]
                team1, team2 = current_round_teams[matchup_idx[0]], current_round_teams[matchup_idx[1]]
            else:
                team1, team2 = current_round_teams[i*2], current_round_teams[i*2+1]
            game_info = {
                'round': r,
                'round_name': {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16", 4: "Elite 8"}[r],
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
    rounds[num_rounds + 1] = current_round_teams
    return rounds, all_games

def run_simulation(use_analytics=True, simulations=1):
    all_results = []
    for sim in range(simulations):
        region_results = {}
        region_champions = {}
        all_games = []
        for reg in region_names:
            rounds, games = simulate_region_bracket(region_teams[reg], reg, use_analytics)
            region_results[reg] = rounds
            region_champions[reg] = rounds[max(rounds.keys())][0]
            all_games.extend(games)
        semifinal_pairs = [('East', 'West'), ('South', 'Midwest')]
        semifinal_results = {}
        final_four_winners = []
        for idx, (regA, regB) in enumerate(semifinal_pairs, start=1):
            team1 = region_champions[regA]
            team2 = region_champions[regB]
            if use_analytics:
                pA = calculate_win_probability(team1, team2)
            else:
                diff = team1['KP_AdjEM'] - team2['KP_AdjEM']
                pA = 1 / (1 + np.exp(-diff/10))
            winner = team1 if random.random() < pA else team2
            winner = winner.copy()
            winner['win_prob'] = pA if winner == team1 else (1-pA)
            semifinal_results[idx] = {'team1': team1, 'team2': team2, 'winner': winner}
            final_four_winners.append(winner)
            all_games.append({
                'round': 5,
                'round_name': "Final Four",
                'region': "National",
                'team1': team1['Team'],
                'seed1': team1['Seed'],
                'team2': team2['Team'],
                'seed2': team2['Seed'],
                'winner': winner['Team'],
                'winner_seed': winner['Seed'],
                'win_prob': winner.get('win_prob', 0.5)
            })
        teamA, teamB = final_four_winners[0], final_four_winners[1]
        if use_analytics:
            pA = calculate_win_probability(teamA, teamB)
        else:
            diff = teamA['KP_AdjEM'] - teamB['KP_AdjEM']
            pA = 1 / (1 + np.exp(-diff/10))
        champion = teamA if random.random() < pA else teamB
        all_games.append({
            'round': 6,
            'round_name': "Championship",
            'region': "National",
            'team1': teamA['Team'],
            'seed1': teamA['Seed'],
            'team2': teamB['Team'],
            'seed2': teamB['Seed'],
            'winner': champion['Team'],
            'winner_seed': champion['Seed'],
            'win_prob': pA if champion == teamA else (1-pA)
        })
        sim_result = {
            'region_champions': region_champions,
            'semifinal_results': semifinal_results,
            'champion': champion,
            'all_games': all_games
        }
        all_results.append(sim_result)
    return all_results

def display_simulation_results(simulation_results, container=None):
    """Display simulation results in a visually appealing way directly in Streamlit"""
    if container is None:
        container = st
        
    # Define colors for rounds using Streamlit's color system
    round_styles = {
        "Round of 64": "blue",
        "Round of 32": "cyan",
        "Sweet 16": "green",
        "Elite 8": "orange",
        "Final Four": "violet",
        "Championship": "red"
    }
    
    if not simulation_results:
        container.warning("No simulation results to display.")
        return
        
    # Get the first simulation result (if multiple)
    result = simulation_results[0]
    games = result['all_games']
    
    # Group games by round
    games_by_round = {}
    for game in games:
        round_name = game['round_name']
        if round_name not in games_by_round:
            games_by_round[round_name] = []
        games_by_round[round_name].append(game)
    
    # Display games by round in expandable sections
    for round_name in ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]:
        if round_name not in games_by_round:
            continue
            
        with container.expander(f"🏀 {round_name}", expanded=(round_name in ["Final Four", "Championship"])):
            for game in games_by_round[round_name]:
                # Format win probability as percentage
                win_prob = f"{game['win_prob']*100:.1f}%" if 'win_prob' in game else "N/A"
                
                # Create matchup text with winner highlighted
                if game['winner'] == game['team1']:
                    matchup_text = f"**({game['seed1']}) {game['team1']}** vs ({game['seed2']}) {game['team2']}"
                else:
                    matchup_text = f"({game['seed1']}) {game['team1']} vs **({game['seed2']}) {game['team2']}**"
                
                # Create colored box with matchup info
                container.markdown(
                    f"""<div style="padding: 8px; border-radius: 5px; margin-bottom: 8px; 
                    background-color: rgba(var(--{round_styles[round_name]}-50), 0.2); 
                    border-left: 4px solid var(--{round_styles[round_name]}-500);">
                    {matchup_text} | Win Prob: {win_prob} | {game.get('region', 'National')} Region
                    </div>""", 
                    unsafe_allow_html=True
                )
    
    # Display champion
    container.markdown("## 🏆 Tournament Champion")
    champion = result['champion']
    container.markdown(
        f"""<div style="padding: 15px; border-radius: 5px; margin: 10px 0; 
        background-color: rgba(var(--red-50), 0.3); border: 2px solid var(--red-500);">
        <h3 style="margin:0;text-align:center;">({champion['Seed']}) {champion['Team']}</h3>
        </div>""", 
        unsafe_allow_html=True
    )

def analyze_simulation_results(all_results):
    num_simulations = len(all_results)
    if num_simulations == 0:
        return {}
    champion_counts = {}
    for result in all_results:
        champion_team = result['champion']['Team']
        champion_counts[champion_team] = champion_counts.get(champion_team, 0) + 1
    champion_probabilities = {team: count / num_simulations for team, count in champion_counts.items()}
    df_champion_probs = pd.DataFrame(list(champion_probabilities.items()), columns=['Team', 'Championship_Probability']).sort_values(by='Championship_Probability', ascending=False)
    region_champion_data = []
    for region in region_names:
        region_champion_counts = {}
        for result in all_results:
            region_champ_team = result['region_champions'][region]['Team']
            region_champion_counts[region_champ_team] = region_champion_counts.get(region_champ_team, 0) + 1
        region_champion_probs = {team: count / num_simulations for team, count in region_champion_counts.items()}
        region_champion_data.append(pd.DataFrame(list(region_champion_probs.items()), columns=['Team', f'{region}_Region_Win_Probability']).sort_values(by=f'{region}_Region_Win_Probability', ascending=False))
    df_region_probs = region_champion_data[0]
    for i in range(1, len(region_champion_data)):
        df_region_probs = pd.merge(df_region_probs, region_champion_data[i], on='Team', how='outer')
    df_region_probs.fillna(0, inplace=True)
    all_games_combined = []
    for result in all_results:
        all_games_combined.extend(result['all_games'])
    games_df_aggregated = pd.DataFrame(all_games_combined)
    games_df_aggregated['upset'] = games_df_aggregated.apply(lambda row: row['winner_seed'] > min(row['seed1'], row['seed2']), axis=1)
    upset_summary_aggregated = games_df_aggregated.groupby(['round_name', 'upset']).size().unstack().fillna(0)
    upset_pct_aggregated = upset_summary_aggregated[True] / (upset_summary_aggregated[True] + upset_summary_aggregated[False]) * 100
    return {
        'champion_probabilities': df_champion_probs,
        'region_probabilities': df_region_probs,
        'games_aggregated': games_df_aggregated,
        'upset_summary_aggregated': upset_summary_aggregated,
        'upset_pct_aggregated': upset_pct_aggregated
    }

def visualize_aggregated_results(analysis_results):
    plt.style.use('dark_background')
    fig = plt.figure(constrained_layout=True, figsize=(14, 12))
    gs = fig.add_gridspec(2, 2)
    
    # Top Teams Championship Win Probability
    ax1 = fig.add_subplot(gs[0, 0])
    champ_probs_df = analysis_results['champion_probabilities'].head(10)
    sns.barplot(x='Championship_Probability', y='Team', data=champ_probs_df, palette="viridis", ax=ax1)
    ax1.set_title('Top Teams Championship Win Probability', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Probability', fontsize=12)
    ax1.set_ylabel('Team', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Aggregated Upset Percentage by Round
    ax2 = fig.add_subplot(gs[0, 1])
    upset_pct_aggregated = analysis_results['upset_pct_aggregated']
    upset_pct_aggregated.plot(kind='bar', color='coral', ax=ax2)
    ax2.set_title('Aggregated Upset Percentage by Round', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Percentage of Games (%)', fontsize=12)
    ax2.set_xlabel('Tournament Round', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # # East Region Win Probability
    # ax3 = fig.add_subplot(gs[1, :])
    # region_probs_df = analysis_results['region_probabilities']
    # if 'East_Region_Win_Probability' in region_probs_df.columns:
    #     east_region_probs_df = region_probs_df[['Team', 'East_Region_Win_Probability']].sort_values(
    #         by='East_Region_Win_Probability', ascending=False).head(10)
    #     sns.barplot(x='East_Region_Win_Probability', y='Team', data=east_region_probs_df, palette="cividis", ax=ax3)
    #     ax3.set_title('Top Teams East Region Win Probability', fontsize=14, fontweight='bold')
    #     ax3.set_xlabel('Probability', fontsize=12)
    #     ax3.set_ylabel('Team', fontsize=12)
    #     ax3.grid(True, linestyle='--', alpha=0.5)
    # else:
    #     ax3.text(0.5, 0.5, "No East Region data", horizontalalignment='center',
    #              verticalalignment='center', fontsize=14, color='white')
    
    # return fig

def create_regional_prob_chart(region_probs):
    """
    Creates a 2x2 subplot bar chart for regional win probabilities.
    Expects a DataFrame with columns: 'Team', 'East_Region_Win_Probability',
    'West_Region_Win_Probability', 'South_Region_Win_Probability', and 'Midwest_Region_Win_Probability'.
    Displays the top 10 teams per region.
    """

    regions = ['East', 'West', 'South', 'Midwest']
    fig = make_subplots(rows=2, cols=2, subplot_titles=regions)
    
    for i, region in enumerate(regions):
        col_name = f"{region}_Region_Win_Probability"
        if col_name in region_probs.columns:
            df_region = region_probs[['Team', col_name]].copy()
            # Sort and select top 10 teams for the region
            df_region = df_region.sort_values(by=col_name, ascending=False).head(10)
            row = i // 2 + 1
            col_idx = i % 2 + 1
            fig.add_trace(go.Bar(
                x=df_region[col_name],
                y=df_region['Team'],
                orientation='h',
                marker=dict(color=df_region[col_name], colorscale='Viridis'),
                name=region
            ), row=row, col=col_idx)
            # Reverse y-axis to show highest probabilities at the top
            fig.update_yaxes(autorange="reversed", row=row, col=col_idx)
    
    fig.update_layout(
        height=600,
        template="plotly_dark",
        title_text="Regional Win Probabilities",
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

def run_tournament_simulation(num_simulations=100, use_analytics=True):
    all_simulation_results = run_simulation(use_analytics=use_analytics, simulations=num_simulations)
    analysis = analyze_simulation_results(all_simulation_results)
    return analysis

# ----------------------------------------------------------------------------
# ------------------ END OF SIMULATION FUNCTIONS ------------------

# ----------------------------------------------------------------------------
# --- App Header & Tabs ---
st.title(":primary[2025 NCAAM BASKETBALL --- MARCH MADNESS]")
st.subheader(":primary[2025 MARCH MADNESS RESEARCH HUB]")
st.caption(":primary[_Cure your bracket brain and propel yourself up the leaderboards by exploring the tabs below:_]")

tab_home, tab_radar, tab_regions, tab_team, tab_conf, tab_pred = st.tabs(["📊 HOME", 
                                                                          "📡 RADAR CHARTS",
                                                                          "🔥 REGIONAL HEATMAPS",
                                                                          "📈 TEAM METRICS",
                                                                          "🏆 CONFERENCE STATS",
                                                                          "🔮 PREDICTIONS"])

# --- Home Tab ---
with tab_home:
    st.subheader(":primary[NCAAM BASKETBALL CONFERENCE TREEMAP]", divider='grey')
    st.caption(":green[_DATA AS OF: 3/15/2025_]")
    treemap = create_treemap(df_main_notnull)
    if treemap is not None:
        st.plotly_chart(treemap, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
        st.subheader("🔍 Team Spotlight", divider='grey')
        selected_team = st.selectbox(
            "Select a team to view detailed metrics:",
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
                    seed_info = f"Seed: {int(team_data['SEED_25'].iloc[0])}" if "SEED_25" in team_data.columns and not pd.isna(team_data["SEED_25"].iloc[0]) else ""
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
                with col2:
                    key_metrics = ["KP_AdjEM", "OFF EFF", "DEF EFF", "TS%", "OPP TS%", "AST/TO%", "STOCKS/GM", "AVG MARGIN"]
                    available_metrics = [m for m in key_metrics if m in team_data.columns]
                    if available_metrics:
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
                        "OFF REB/GM", "DEF REB/GM", "BLKS/GM", "STL/GM", "STOCKS/GM", "STOCKS-TOV/GM"
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
    if "CONFERENCE" in df_main.columns:
        conf_counts = df_main["CONFERENCE"].value_counts().reset_index()
        conf_counts.columns = ["CONFERENCE", "# TEAMS"]
        if "KP_AdjEM" in df_main.columns:
            conf_stats = (
                df_main.groupby("CONFERENCE")["KP_AdjEM"]
                .agg(["count", "max", "mean", "min"])
                .reset_index()
            )
            conf_stats = conf_stats.rename(columns={
                "count": "# TEAMS",
                "max": "MAX AdjEM",
                "mean": "MEAN AdjEM",
                "min": "MIN AdjEM"
            })
            conf_stats = conf_stats.sort_values("MEAN AdjEM", ascending=False)
            st.subheader(":primary[NCAAM BASKETBALL CONFERENCE POWER RANKINGS]", divider='grey')
            with st.expander("*About Conference Power Rankings:*"):
                st.markdown("""
                            Simple-average rollup of each conference:
                            - **MEAN AdjEM**: Average KenPom Adjusted Efficiency Margin within conference
                            - **MAX/MIN AdjEM**: Range of AdjEM values among teams within conference
                            """)
            conf_stats["CONFERENCE"] = conf_stats["CONFERENCE"].apply(get_conf_logo_html)
            styled_conf_stats = (
                conf_stats.style
                .format({
                    "MEAN AdjEM": "{:.2f}",
                    "MIN AdjEM": "{:.2f}",
                    "MAX AdjEM": "{:.2f}",
                })
                .background_gradient(cmap="RdYlGn", subset=["MEAN AdjEM", "MIN AdjEM", "MAX AdjEM"])
                .set_table_styles(detailed_table_styles)
            )
            st.markdown(styled_conf_stats.to_html(escape=False), unsafe_allow_html=True)

# --- Radar Charts Tab ---
with tab_radar:
    st.header("Seed-by-Seed Radar Charts (16x4)")

    # Create the big 16x4 grid figure
    bracket_radar_fig = create_seed_radar_grid(df_main, region_teams)

    if bracket_radar_fig:
        st.plotly_chart(bracket_radar_fig, use_container_width=True)
    else:
        st.warning("No data found for bracket radar grid.")

    with st.expander("About This Radar Grid:"):
        st.markdown("""
        **Each row** represents seeds 1 through 16.<br>
        **Each column** represents one of the four major regions (East, West, South, Midwest).<br>
        Each subplot compares the team to the national average (red) and their conference average (green).<br>
        - Radial scale 0..10, where 5 is NCAA average.
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
    east_teams_2025 = [
        "Duke", "Tennessee", "Iowa St.", "Maryland", "Texas A&M", "Kansas", "UCLA", "Mississippi St.",
        "Georgia", "Ohio St.", "New Mexico", "Indiana", "Memphis", "Villanova", "Santa Clara", "Pittsburgh",
        "TOURNEY AVG"
    ]
    west_teams_2025 = [
        "Auburn", "Alabama", "Gonzaga", "Purdue", "Illinois", "Saint Mary's", "Marquette", "Michigan",
        "Connecticut", "Oklahoma", "Xavier", "Northwestern", "Boise St.", "West Virginia", "Drake", "Liberty",
        "TOURNEY AVG"
    ]
    south_teams_2025 = [
        "Houston", "Texas Tech", "Kentucky", "St. John's", "Clemson", "Louisville", "Mississippi", "VCU",
        "North Carolina", "UC San Diego", "San Diego St.", "Vanderbilt", "Colorado St.", "Nebraska", "Penn St.", "Iowa",
        "TOURNEY AVG"
    ]
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
        "STOCKS/GM": "RdBu"
    }
    for region_name, team_list in regions.items():
        teams_found = [tm for tm in team_list if tm in df_heat_T.columns]
        if teams_found:
            region_df = df_heat_T[teams_found].copy()
            st.subheader(region_name)
            region_styler = region_df.style.format(safe_format)
            for row_label, cmap in color_map_dict.items():
                if row_label in region_df.index:
                    region_styler = region_styler.background_gradient(cmap=cmap, subset=pd.IndexSlice[[row_label], :])
            region_styler = region_styler.set_table_styles(detailed_table_styles)
            with st.expander("📊 Key Metrics Explained"):
                st.markdown("""
                            - **KP_Rank**: KenPom team ranking (lower is better)
                            - **KP_AdjEM**: Adjusted efficiency margin (higher is better)
                            - **WIN% CLOSE GM**: Win percentage in games decided by 5 points or less
                            - **OFF/DEF EFF**: Points scored/allowed per 100 possessions
                            - **eFG%/TS%**: Effective field goal & true shooting percentages
                            """)
            st.markdown(region_styler.to_html(), unsafe_allow_html=True)
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
                styled_table = styled_table.set_caption("Team Performance Metrics Comparison")
                st.markdown(styled_table.to_html(), unsafe_allow_html=True)
                
            else:
                radar_fig = create_radar_chart(selected_teams, df_main)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
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
        
# --- 🔮 PREDICTIONS Tab (Bracket Simulation) ---
# 1) Define a small helper function to color-code round logs
# 1) Define a small helper function to color-code round logs
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
        # 2) Let the user decide if they want to see detailed logs (single-run example)
        show_detailed_logs = st.checkbox("Show Detailed Logs (Single Simulation Recommended)", value=False)
        st.write("Run the tournament simulation across multiple iterations to see aggregated outcomes.")
        
        # 3) Single button with a unique key argument
        if st.button("Run Bracket Simulation", key="btn_run_bracket"):
            with st.spinner("Simulating tournament..."):
                # Run your existing multi-simulation
                aggregated_analysis = run_tournament_simulation(num_simulations=100, use_analytics=True)
                
                # Also run a single simulation to display detailed results
                single_sim_results = run_simulation(use_analytics=True, simulations=1)
                st.session_state['single_sim_results'] = single_sim_results

            st.success("Simulation complete!")

            # -----------------------------------------
            # Apply advanced styling to two result DataFrames:
            #   (1) Championship Win Probabilities
            #   (2) Upset Summary
            # for consistency with the rest of the app.
            # -----------------------------------------

            # (A) Championship Win Probabilities
            st.subheader("Championship Win Probabilities")
            champ_df = aggregated_analysis['champion_probabilities'].copy()
            
            # Convert numeric columns to background gradient
            numeric_cols_champ = champ_df.select_dtypes(include=[float, int]).columns
            styled_champ = (
                champ_df.style
                .format("{:.2%}", subset=["Championship_Probability"])  # percentage style
                .background_gradient(cmap="RdYlGn", subset=numeric_cols_champ)
                .set_table_styles(detailed_table_styles)
                .set_caption("Championship Win Probabilities by Team")
            )
            # Render as HTML
            st.markdown(styled_champ.to_html(), unsafe_allow_html=True)

            # (B) Regional Win Probabilities Chart (plotly)
            st.subheader("Regional Win Probabilities")
            if 'region_probabilities' in aggregated_analysis:
                fig_regional = create_regional_prob_chart(aggregated_analysis['region_probabilities'])
                st.plotly_chart(fig_regional, use_container_width=True)
            else:
                st.warning("Regional win probabilities data not available.")

            # (C) Upset Analysis
            st.subheader("Aggregated Upset Analysis")
            upset_summary_df = pd.DataFrame({
                'Round': aggregated_analysis['upset_pct_aggregated'].index,
                'Upset %': aggregated_analysis['upset_pct_aggregated'].values.round(1)
            })
            # Style that similarly
            numeric_cols_upsets = upset_summary_df.select_dtypes(include=[float, int]).columns
            styled_upsets = (
                upset_summary_df.style
                .format("{:.1f}", subset=["Upset %"])
                .background_gradient(cmap="RdYlGn", subset=numeric_cols_upsets)
                .set_table_styles(detailed_table_styles)
                .set_caption("Upset Analysis by Round")
            )
            st.markdown(styled_upsets.to_html(), unsafe_allow_html=True)

            # (D) Show aggregated matplotlib figure with uniform styling
            try:
                agg_viz_fig = visualize_aggregated_results(aggregated_analysis)
                st.pyplot(agg_viz_fig, use_container_width=True)  # Pass figure explicitly
            except Exception as e:
                st.error(f"Could not generate aggregated visualizations: {e}")

            # (E) Enhanced detailed simulation display for single run results
            if show_detailed_logs and 'single_sim_results' in st.session_state:
                st.info("Detailed game outcomes for a single simulation run:")
                
                # Use our new enhanced display function
                display_simulation_results(st.session_state['single_sim_results'], st)
            
            # Legacy detailed logs (can keep both or replace with enhanced version)
            elif show_detailed_logs:
                st.info("Detailed logs below show one example simulation's game outcomes:")
                single_run_results = run_simulation(use_analytics=True, simulations=1)[0]
                detailed_games = single_run_results["all_games"]

                st.write("---")
                st.write("### Detailed Game Outcomes (Single Simulation):")

                # Print each round's outcome in color-coded HTML
                for gm in detailed_games:
                    round_label   = gm["round_name"]
                    t1, t2        = gm["team1"], gm["team2"]
                    winner        = gm["winner"]
                    prob_to_win   = gm["win_prob"]
                    colored_label = color_log_text(round_label, f"[{round_label}]")

                    # Mark upsets for emphasis
                    upset_flag = ""
                    if gm["winner_seed"] > min(gm["seed1"], gm["seed2"]):
                        upset_flag = "<b>(UPSET!)</b>"

                    line_html = (
                        f"{colored_label} &nbsp;"
                        f"{t1} vs. {t2} → "
                        f"<b>Winner:</b> {winner} "
                        f"<small>(win prob {prob_to_win:.1%})</small> "
                        f"{upset_flag}"
                    )
                    st.markdown(line_html, unsafe_allow_html=True)

    # Add bracket visualization in the second column or below the simulation section
    with viz_col:
        st.subheader("Bracket Visualization")
        st.markdown("Select a view to explore the tournament teams")
        
        viz_type = st.radio(
            "Visualization Type", 
            ["Team Stats", "Bracket Overview"],
            horizontal=True
        )
        
        if viz_type == "Team Stats":
            # Display a team selector dropdown
            all_tourney_teams = TR_df[TR_df['SEED_25'].notna()]['TM_TR'].tolist()
            selected_team = st.selectbox("Select Team", sorted(all_tourney_teams))
            
            # Display the radar for just this team
            team_data = TR_df[TR_df['TM_TR'] == selected_team].iloc[0]
            create_team_radar(team_data, dark_mode=True)
            
            # Show key team stats below the radar
            st.markdown("### Key Team Stats")
            
            # Extract and display key stats in a neat format
            key_stats = {
                "Seed": f"{int(team_data['REG_SEED_25'])} ({team_data['REGION_25']})",
                "Record": f"{team_data['WIN_25']}-{team_data['LOSS_25']}",
                "NET Rank": f"{int(team_data['NET_25'])}",
                "KenPom Rank": f"{int(team_data['KP_Rank'])}",
                "Offense Rank": f"{int(team_data['TR_ORk_25'])}",
                "Defense Rank": f"{int(team_data['TR_DRk_25'])}"
            }
            
            # Create two columns for stats
            stat_col1, stat_col2 = st.columns(2)
            
            # Display stats in columns
            for i, (stat, value) in enumerate(key_stats.items()):
                if i % 2 == 0:
                    stat_col1.metric(stat, value)
                else:
                    stat_col2.metric(stat, value)
                    
        else:
            # Show button to load full bracket visualization
            if st.button("Load Full Bracket Visualization"):
                with st.spinner("Generating bracket visualization..."):
                    create_bracket_radar_grid()

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