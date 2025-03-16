import streamlit as st
#import from streamlit_plotly_events import plotly_events
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import base64
from io import BytesIO
from PIL import Image

import os, math

# --- Streamlit Setup ---
st.set_page_config(page_title="NCAA BASKETBALL -- MARCH MADNESS 2025",
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
core_cols = [
    "KP_Rank", "WIN_25", "LOSS_25", "WIN% ALL GM", "WIN% CLOSE GM",
    "KP_AdjEM", "KP_SOS_AdjEM", "OFF EFF", "DEF EFF", "OFF REB/GM", "DEF REB/GM",
    "BLKS/GM", "STL/GM", "AST/GM", "TO/GM", "AVG MARGIN", "PTS/GM", "OPP PTS/GM",
    "eFG%", "OPP eFG%", "TS%", "OPP TS%", "AST/TO%", "STOCKS/GM", "STOCKS-TOV/GM"
]
extra_cols_for_treemap = ["CONFERENCE", "TM_KP", "SEED_25"]
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

conference_logo_map = {"ACC": ACC_logo, "AAC": AAC_logo, "AEC": AEC_logo, "ASUN": ASUN_logo, "B10": B10_logo, "B12": B12_logo, "BE": BE_logo,
                       "Big South": BSouth_logo, "Big Sky": BSky_logo, "Big West": BWest_logo, "CAA": CAA_logo, "CUSA": CUSA_logo, "Horizon": Horizon_logo,
                       "Ivy": Ivy_logo, "MAAC": MAAC_logo, "MAC": MAC_logo, "MEAC": MEAC_logo, "MVC": MVC_logo, "MWC": MWC_logo, "NEC": NEC_logo,
                       "OVC": OVC_logo, "Patriot": Patriot_logo, "SBC": SBC_logo, "SEC": SEC_logo, "SoCon": SoCon_logo, "Southland": Southland_logo, "Summit": Summit_logo,
                       "SWAC": SWAC_logo, "WAC": WAC_logo, "WCC": WCC_logo,
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
    return ['AVG MARGIN',
            'KP_AdjEM', 
            'OFF EFF', 'DEF EFF',
            'OFF REB/GM', 'DEF REB/GM',
            'AST/TO%', 'STOCKS/GM',
            #'BLKS/GM', 'STL/GM', 'AST/GM', 'TO/GM',
            ]

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
        vertical_spacing=0.20
    )
    fig.update_layout(
        height=fig_height,
        title={
            'text': "Team Performance Radar Charts",
            'font': {'size': 24, 'family': 'Arial, sans-serif', 'color': 'white'},
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template='plotly_dark',
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.1)"
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor="rgba(0,0,0,0.1)",
        plot_bgcolor="rgba(0,0,0,0.1)"
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
        bgcolor="rgba(0,0,0,0.2)"
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
            font=dict(size=12, color="white"),
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
                f"KP Rank: {int(x['KP_Rank'])}<br>"
                f"Record: {int(x['WIN_25'])}-{int(x['LOSS_25'])}<br>"
                f"AdjEM: {x['KP_AdjEM']:.1f}<br>"
            )
            if "OFF EFF" in x and "DEF EFF" in x:
                base += f"OFF EFF: {x['OFF EFF']:.1f}<br>DEF EFF: {x['DEF EFF']:.1f}<br>"
            if "SEED_25" in x and not pd.isna(x["SEED_25"]):
                base += f"Seed: {int(x['SEED_25'])}"
            return base

        treemap_data['hover_text'] = treemap_data.apply(hover_text_func, axis=1)
        treemap = px.treemap(
            treemap_data,
            path=["CONFERENCE", "TM_KP"],
            values="KP_AdjEM",
            color="KP_AdjEM",
            color_continuous_scale=px.colors.diverging.RdBu_r,
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
                title="AdjEM",
                thicknessmode="pixels",
                thickness=15,
                lenmode="pixels",
                len=300,
                yanchor="top",
                y=1,
                ticks="outside"
            ),
            template="plotly_dark"
        )
        return treemap
    except Exception as e:
        st.error(f"An error occurred while generating treemap: {e}")
        return None

# ----------------------------------------------------------------------------
# --- App Header & Tabs --- #
st.title(":primary[2025 NCAAM BASKETBALL --- MARCH MADNESS]")
st.subheader(":primary[2025 MARCH MADNESS RESEARCH HUB]")
st.caption(":green[_Cure your bracket brain and propel your bracket up the leaderboards by exploring the tabs below:_]")

tab_home, tab_radar, tab_regions, tab_hist, tab_corr, tab_conf, tab_team, tab_tbd = st.tabs(["HOME", "RADAR CHARTS",
                                                                                             "REGIONAL HEATMAPS",
                                                                                             "HISTOGRAM", "CORRELATION HEATMAP",
                                                                                             "CONFERENCE COMPARISON",
                                                                                             "TEAM METRICS COMPARISON",
                                                                                             "TBU",
                                                                                             ])

# --- Home Tab --- #
treemap = create_treemap(df_main_notnull)
with tab_home:
    st.subheader(":primary[NCAAM BASKETBALL CONFERENCE TREEMAP]", divider='grey')
    st.caption(":green[_DATA AS OF: 3/12/2025_]")
    treemap = create_treemap(df_main_notnull)
    if treemap is not None:
        st.plotly_chart(treemap, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
        st.caption(":green[_DATA AS OF: 3/12/2025_]")
    else:
        st.warning("TREEMAP OVERHEATED.")

    with st.expander("*About Conference Treemap:*"):
        st.markdown("""
                    This table shows a summary of each conference:
                    - **Avg AdjEM**: Average KenPom Adjusted Efficiency Margin for the conference
                    - **Min/Max**: Range of AdjEM values among teams in that conference
                    - **Count**: Number of teams in the conference
                    """)
    st.caption(":green[_DATA AS OF: 3/12/2025_]")

if "CONFERENCE" in df_main.columns:
    # Compute team counts per conference
    conf_counts = df_main["CONFERENCE"].value_counts().reset_index()
    conf_counts.columns = ["CONFERENCE", "# TEAMS"]

    if "KP_AdjEM" in df_main.columns:
        # Compute aggregated KP_AdjEM statistics by conference
        conf_stats = (
            df_main.groupby("CONFERENCE")["KP_AdjEM"]
            .agg(["count", "max", "mean", "min"])
            .reset_index()
        )
        # Rename columns in a Pythonic way
        conf_stats = conf_stats.rename(columns={
            "count": "# TEAMS",
            "max": "MAX AdjEM",
            "mean": "MEAN AdjEM",
            "min": "MIN AdjEM"
        })
        conf_stats = conf_stats.sort_values("MEAN AdjEM", ascending=False)

        st.subheader(":primary[NCAAM BASKETBALL CONFERENCE POWER RANKINGS]", divider='grey')
        with st.expander("*About Conference Power Rankings:*"):
            st.markdown(
                """
                Simple-average rollup of each conference:
                - **MEAN AdjEM**: Average KenPom Adjusted Efficiency Margin within conference
                - **MAX/MIN AdjEM**: Range of AdjEM values among teams within conference
                """
            )

        # Replace the conference name with an HTML snippet including the logo
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



# --- Radar Charts Tab --- #
with tab_radar:
    st.header("TEAM RADAR CHARTS")
    radar_metrics = get_default_metrics()
    available_radar_metrics = [m for m in radar_metrics if m in df_main.columns]
    if len(available_radar_metrics) < 3:
        st.warning(f"Not enough radar metrics available. Need at least 4: {', '.join(radar_metrics)}")
    else:
        if "TM_KP" in df_main.columns:
            all_teams = sorted(df_main["TM_KP"].dropna().unique().tolist())
            default_teams = ['Duke', 'Kansas', 'Auburn', 'Houston']
            if "KP_AdjEM" in df_main.columns:
                top_teams = df_main.sort_values("KP_AdjEM", ascending=False).head(4)
                if "TM_KP" in top_teams.columns:
                    default_teams = top_teams["TM_KP"].tolist()
            if not default_teams and all_teams:
                default_teams = all_teams[:4]
            selected_teams = st.multiselect(
                "Select Teams to Compare:",
                options=all_teams,
                default=default_teams
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

# --- Regional Heatmaps Tab --- #
with tab_regions:
    st.header("BRACKET ANALYSIS")
    st.write("REGIONAL HEATFRAMES (W, X, Y, Z)")
    df_heat = df_main.copy()
    # Compute mean for numeric columns and reindex to match all columns to avoid index length mismatches
    numeric_cols_heat = df_heat.select_dtypes(include=np.number).columns
    mean_series = df_heat.mean(numeric_only=True)
    mean_series = mean_series.reindex(df_heat.columns, fill_value=np.nan)
    df_heat.loc["TOURNEY AVG"] = mean_series
    df_heat_T = df_heat.T

    # Region seeds
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

# --- Histogram Tab --- #
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
    with st.expander("About Histogram Metrics:"):
        st.markdown("""
        **Histogram Metric Description:**
        - **KP_AdjEM**: Adjusted efficiency margin from KenPom ratings.
        - **OFF EFF/DEF EFF**: Offensive/Defensive efficiency.
        - Other metrics follow similar definitions as per NCAA advanced statistics.
        """)

# --- Correlation Heatmap Tab --- #
with tab_corr:
    st.header("Correlation Heatmap")
    numeric_cols = [c for c in core_cols if c in df_main.columns and pd.api.types.is_numeric_dtype(df_main[c])]
    default_corr_metrics = [m for m in ["KP_AdjEM", "OFF EFF", "DEF EFF", "PTS/GM", "OPP PTS/GM"] if m in numeric_cols]
    selected_corr_metrics = st.multiselect("Select Metrics for Correlation Analysis", options=numeric_cols, default=default_corr_metrics)
    if len(selected_corr_metrics) >= 2:
        df_for_corr = df_main[selected_corr_metrics].dropna()
        corr_mat = df_for_corr.corr().round(2)
        fig_corr = px.imshow(
            corr_mat,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix",
            template="plotly_dark"
        )
        fig_corr.update_layout(width=800, height=700)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Please select at least 2 metrics for correlation analysis.")
    with st.expander("About Correlation Metrics:"):
        st.markdown("""
        **Correlation Heatmap Glossary:**
        - **Correlation Coefficient**: Measures linear relationship between two variables.
        - **Positive/Negative Correlation**: Indicates the direction of the relationship.
        - **Metrics**: Derived from advanced team statistics.
        """)

# --- Conference Comparison Tab --- #
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
    with st.expander("About Conference Comparisons:"):
        st.markdown("""
        **Conference Comparison Glossary:**
        - **Avg AdjEM**: Average Adjusted Efficiency Margin.
        - **Conference**: Grouping of teams by their athletic conferences.
        - Metrics provide insight into relative conference strength.
        """)

# --- Team Metrics Comparison Tab --- #
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
        fig_scatter = px.scatter(
            df_main.reset_index(), x=x_metric, y=y_metric, color="CONFERENCE",
            hover_name="TEAM",
            size="KP_AdjEM_Offset" if "KP_AdjEM_Offset" in df_main.columns else None,
            size_max=15, opacity=0.8, template="plotly_dark",
            title=f"{y_metric} vs {x_metric}", height=700
        )
    else:
        fig_scatter = px.scatter(
            df_main.reset_index(), x=x_metric, y=y_metric,
            hover_name="TEAM", opacity=0.8,
            template="plotly_dark", title=f"{y_metric} vs {x_metric}"
        )
    if (("OFF" in x_metric and "DEF" in y_metric) or ("DEF" in x_metric and "OFF" in y_metric)):
        x_avg, y_avg = df_main[x_metric].mean(), df_main[y_metric].mean()
        fig_scatter.add_hline(y=y_avg, line_dash="dash", line_color="white", opacity=0.4)
        fig_scatter.add_vline(x=x_avg, line_dash="dash", line_color="white", opacity=0.4)
        quadrants = [
            {"x": x_avg * 0.9, "y": y_avg * 0.9, "text": "WEAK OFF/DEF"},
            {"x": x_avg * 1.1, "y": y_avg * 0.9, "text": "SOLID OFF/DEF"},
            {"x": x_avg * 0.9, "y": y_avg * 1.1, "text": "SOLID DEF"},
            {"x": x_avg * 1.1, "y": y_avg * 1.1, "text": "SOLID OFF"}
        ]
        for q in quadrants:
            fig_scatter.add_annotation(
                x=q["x"], y=q["y"], text=q["text"], showarrow=False,
                font=dict(color="white", size=10), opacity=0.7
            )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- TBD Tab ---
with tab_tbd:
    st.header("BRACKET SIMULATIONS -- TO BE UPDATED")
    st.info("Additional visualizations, bracket simulations coming soon.")

# ----------------------------------------------------------------------------
# CONFERENCE LOGOS GRID
conference_logos = [
    [AAC_logo, ACC_logo, AEC_logo, ASUN_logo, B10_logo, B12_logo, BE_logo],
    [BSouth_logo, BSky_logo, BWest_logo, CAA_logo, CUSA_logo, Horizon_logo, Ivy_logo],
    [MAAC_logo, MAC_logo, MEAC_logo, MVC_logo, MWC_logo, NEC_logo, OVC_logo],
    [Patriot_logo, SBC_logo, SEC_logo, Summit_logo, SWAC_logo, WAC_logo, WCC_logo],
]
for row in conference_logos:
    cols = st.columns(len(row))
    for i, logo in enumerate(row):
        if logo:
            with cols[i]:
                st.image(logo, width=75)

if FinalFour25_logo:
    st.image(FinalFour25_logo, use_container_width=True)

# GitHub Link & App Footer
st.markdown("---")
st.caption("Python code framework available on [GitHub](https://github.com/nehat312/march-madness-2025)")
st.caption("DATA SOURCED FROM: [TeamRankings](https://www.teamrankings.com/ncaa-basketball/ranking/predictive-by-other/), [KenPom](https://kenpom.com/)")
st.stop()
