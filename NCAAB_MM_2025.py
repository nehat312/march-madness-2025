import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

import base64
import copy
from io import BytesIO
from PIL import Image

import os, math, logging, random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
mm_database_csv = abs_path + '/data/mm_2025_database_vF.csv'

@st.cache_data
def load_data():
    df = pd.read_csv(mm_database_csv, index_col=0)
    df.index.name = "TEAM"
    return df

mm_database_2025 = load_data()

# ----------------------------------------------------------------------------
# Select Relevant Columns (including radar metrics)
core_cols = ["WIN_25", "LOSS_25", "WIN% ALL GM", "WIN% CLOSE GM",
             "KP_Rank", "NET_25", "BPI_25", "BPI_Rk_25", "SEED_25", 'REGION_25',
             "KP_AdjEM", "KP_SOS_AdjEM",
             "OFF EFF", "DEF EFF",
             "KP_AdjO", "KP_AdjD",
             #'TR_ORk_25', 'TR_DRk_25',  
             "AVG MARGIN", "PTS/GM", "OPP PTS/GM",
            "TS%", "OPP TS%", 
             "OFF REB/GM", "DEF REB/GM",
             "BLKS/GM", "STL/GM", "AST/GM", "TO/GM", 
             "AST/TO%", "STOCKS/GM", "STOCKS-TOV/GM",
             "FT%", "3PT%", "3PTA/GM", #"3PTM/GM", 
             "NET_eFG%", "eFG%", "OPP eFG%",
             ]

# ----------------------------------------------------------------------------
# Prepare Data for Treemap / Path configurations
extra_cols_for_treemap = ["CONFERENCE", "TM_KP"] #, "SEED_25"  "REGION_25"
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

required_path_cols = ["CONFERENCE", "TM_KP", "KP_AdjEM"]
if all(col in df_main.columns for col in required_path_cols):
    df_main_notnull = df_main.dropna(subset=required_path_cols, how="any").copy()
else:
    df_main_notnull = df_main.copy()

# ----------------------------------------------------------------------------
# Logo Loading / Image Configuration
logo_path = "images/NCAA_logo1.png"
FinalFour25_logo_path = "images/ncaab_mens_finalfour2025_logo.png"
Conferences25_logo_path = "images/ncaab_conferences_2025.png"
Banner_logo_path = "images/MM2025_banner1.png"

## FINAL FOUR ##
Duke_BlueDevils = "images/Duke_BlueDevils.png"
Houston_Cougars = "images/Houston_Cougars.png"
Florida_Gators = "images/Florida_Gators.png"
Auburn_Tigers = "images/Auburn_Tigers.png"

## HEADER // FOOTER IMAGES ##
UAP_court1 = "images/UAP_court1.png"
UAP_court2 = "images/UAP_court2.png"
Space_court1 = "images/Space_court1.png"
Space_court2 = "images/Space_court2.png"

## CONFERENCES ##
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
Banner_logo = Image.open(Banner_logo_path) if os.path.exists(Banner_logo_path) else None

Duke_BlueDevils_logo = Image.open(Duke_BlueDevils) if os.path.exists(Duke_BlueDevils) else None
Houston_Cougars_logo = Image.open(Houston_Cougars) if os.path.exists(Houston_Cougars) else None
Florida_Gators_logo = Image.open(Florida_Gators) if os.path.exists(Florida_Gators) else None
Auburn_Tigers_logo = Image.open(Auburn_Tigers) if os.path.exists(Auburn_Tigers) else None

## HEADER // FOOTER IMAGES ##
UAP_court1_banner = Image.open(UAP_court1) if os.path.exists(UAP_court1) else None
UAP_court2_banner = Image.open(UAP_court2) if os.path.exists(UAP_court2) else None
Space_court1_banner = Image.open(Space_court1) if os.path.exists(Space_court1) else None
Space_court2_banner = Image.open(Space_court2) if os.path.exists(Space_court2) else None

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

final_four_logo_map = {"Duke": Duke_BlueDevils, "Houston": Houston_Cougars,
                       "Florida": Florida_Gators, "Auburn": Auburn_Tigers,
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

def get_team_logo_html(conf_name):  # Return HTML <img> + conference name for table column 
    img_obj = final_four_logo_map.get(conf_name, None)
    if img_obj:
        encoded = image_to_base64(img_obj)
        if encoded:
            return f'<img src="data:image/png;base64,{encoded}" width="40" style="vertical-align: middle;" /> {conf_name}'
    return conf_name

def get_interpretive_insights(row, df_all):
                    lines = []
                    t_avgs, t_stdevs = compute_tournament_stats(df_all)
                    for metric in get_default_metrics():
                        if metric in row:
                            mean_val = t_avgs.get(metric, 0)
                            std_val = max(t_stdevs.get(metric, 1), 1e-6)
                            team_val = row[metric]
                            z = (team_val - mean_val) / std_val
                            if metric in ["DEF EFF", "TO/GM", "KP_AdjD", "KP_SOS_AdjEM"]:
                                z = -z
                            if abs(z) < 0.3:
                                lines.append(f"**{metric}** | Near NCAA average.")
                            elif z >= 1.0:
                                lines.append(f"**{metric}** | Clear strength.")
                            elif 0.3 <= z < 1.0:
                                lines.append(f"**{metric}** | Above NCAA average.")
                            elif -1.0 < z <= -0.3:
                                lines.append(f"**{metric}** | Below NCAA average.")
                            else:
                                lines.append(f"**{metric}** | Notable weakness.")
                    return lines

def get_interpretive_insights_opp(row, df_all):
    lines = []
    t_avgs, t_stdevs = compute_tournament_stats(df_all)
    for metric in get_default_metrics():
        if metric in row:
            mean_val = t_avgs.get(metric, 0)
            std_val = max(t_stdevs.get(metric, 1), 1e-6)
            val = row[metric]
            z = (val - mean_val) / std_val
            if metric in ["DEF EFF", "TO/GM", "KP_AdjD", "KP_SOS_AdjEM"]:
                z = -z
            if abs(z) < 0.3:
                lines.append(f"**{metric}** | Near NCAA average.")
            elif z >= 1.0:
                lines.append(f"**{metric}** | Clear strength.")
            elif 0.3 <= z < 1.0:
                lines.append(f"**{metric}** | Above NCAA average.")
            elif -1.0 < z <= -0.3:
                lines.append(f"**{metric}** | Below NCAA average.")
            else:
                lines.append(f"**{metric}** | Notable weakness.")
    return lines

# Helper to compute performance badge with revised thresholds
def compute_performance_badge(row, df_all):
    t_avgs, t_stdevs = compute_tournament_stats(df_all)
    metrics = get_default_metrics()
    z_vals = []
    for m in metrics:
        if m in row and m in t_avgs and m in t_stdevs:
            std = t_stdevs[m] if t_stdevs[m] > 1e-12 else 1e-6
            val = row[m]
            mean_ = t_avgs[m]
            z = (val - mean_) / std
            # Invert if lower-is-better
            if m in ["DEF EFF", "KP_AdjD", "TO/GM", "KP_SOS_AdjEM"]:
                z = -z
            z_vals.append(z)
    if not z_vals:
        return {"text": "NO DATA", "class": "badge-mid"}
    avg_z = sum(z_vals) / len(z_vals)
    # Adjust thresholds as you see fit
    if avg_z >= 0.75:
        return {"text": "ELITE", "class": "badge-elite"}
    elif avg_z >= 0.25:
        return {"text": "SOLID", "class": "badge-solid"}
    elif avg_z >= -0.25:
        return {"text": "MID", "class": "badge-mid"}
    elif avg_z >= -0.75:
        return {"text": "SUBPAR", "class": "badge-subpar"}
    else:
        return {"text": "WEAK", "class": "badge-weak"}

# ----------------------------------------------------------------------------
# Global visualization settings
viz_margin_dict = dict(l=20, r=20, t=50, b=20)
viz_bg_color = '#0360CE'
viz_font_dict = dict(size=12, color='#FFFFFF')
RdYlGn = px.colors.diverging.RdYlGn
Spectral = px.colors.diverging.Spectral
RdBu_r = px.colors.diverging.RdBu_r

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
        ('border-bottom', '2px dashed #000000'),
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

# ----------------------------------------------------------------------------
# Radar Chart Functions
def get_default_metrics():
    """Return metrics to be used in radar charts z-score logic."""
    return [
        'AVG MARGIN', 'KP_AdjEM',
        'BPI_25',
        #'NET_25'
        'KP_AdjO', 'KP_AdjD',
        'OFF EFF', 'DEF EFF',
        'AST/TO%', 'STOCKS-TOV/GM',
        ]

# --- 2025 NCAA TOURNAMENT UPDATED RESULTS (ELITE 8)---

completed_results_2025 = {
    'First Four': [
        ('Alabama St.', 'Saint Francis', 70, 68),
        ('North Carolina', 'San Diego St.', 95, 68),
        ('Mount St. Mary\'s', 'American University', 83, 72),
        ('Xavier', 'Texas', 86, 80)
    ],
    'Round of 64': [
        ('Creighton', 'Louisville', 89, 75),
        ('Purdue', 'High Point', 75, 63),
        ('Wisconsin', 'Montana', 85, 66),
        ('Houston', 'SIU Edwardsville', 78, 40),
        ('Auburn', 'Alabama St.', 83, 63),
        ('McNeese', 'Clemson', 69, 67),
        ('BYU', 'VCU', 80, 71),
        ('Gonzaga', 'Georgia', 89, 68),
        ('Tennessee', 'Wofford', 77, 62),
        ('Arkansas', 'Kansas', 79, 72),
        ('Texas A&M', 'Yale', 80, 71),
        ('Drake', 'Missouri', 67, 57),
        ('UCLA', 'Utah St.', 72, 47),
        ('St. John\'s', 'Omaha', 83, 53),
        ('Michigan', 'UC San Diego', 68, 65),
        ('Texas Tech', 'UNC Wilmington', 82, 72),
        ('Baylor', 'Mississippi St.', 75, 72),
        ('Alabama', 'Robert Morris', 90, 81),
        ('Iowa St.', 'Lipscomb', 82, 55),
        ('Colorado St.', 'Memphis', 78, 70),
        ('Duke', 'Mount St. Mary\'s', 93, 49),
        ('Saint Mary\'s', 'Vanderbilt', 59, 56),
        ('Mississippi', 'North Carolina', 71, 64),
        ('Maryland', 'Grand Canyon', 81, 49),
        ('Florida', 'Norfolk St.', 95, 69),
        ('Kentucky', 'Troy', 76, 57),
        ('New Mexico', 'Marquette', 75, 66),
        ('Arizona', 'Akron', 93, 65),
        ('UConn', 'Oklahoma', 67, 59),
        ('Illinois', 'Xavier', 86, 73),
        ('Michigan St.', 'Bryant', 87, 62),
        ('Oregon', 'Liberty', 81, 52)
    ],
    'Round of 32': [
        ('Purdue', 'McNeese', 76, 62),
        ('Arkansas', 'St. John\'s', 75, 66),
        ('Michigan', 'Texas A&M', 91, 79),
        ('Texas Tech', 'Drake', 77, 64),
        ('Auburn', 'Creighton', 82, 70),
        ('BYU', 'Wisconsin', 91, 89),
        ('Houston', 'Gonzaga', 81, 76),
        ('Tennessee', 'UCLA', 67, 58),
        ('Florida', 'UConn', 77, 75),
        ('Duke', 'Baylor', 89, 66),
        ('Kentucky', 'Illinois', 84, 75),
        ('Alabama', 'Saint Mary\'s', 80, 66),
        ('Maryland', 'Colorado St.', 72, 71),
        ('Mississippi', 'Iowa St.', 91, 78),
        ('Michigan St.', 'New Mexico', 71, 63),
        ('Arizona', 'Oregon', 87, 83)
    ],
    'Sweet 16': [
        ('Texas Tech', 'Arkansas', 85, 83),
        ('Auburn', 'Michigan', 78, 65),
        ('Houston', 'Purdue', 62, 60),
        ('Tennessee', 'Kentucky', 78, 65),
        ('Florida', 'Maryland', 87, 71),
        ('Duke', 'Arizona', 100, 93),
        ('Alabama', 'BYU', 113, 88),
        ('Michigan St.', 'Mississippi', 73, 70),
    ],
    'Elite 8': [
        ('Florida', 'Texas Tech', 84, 79),
        ('Duke', 'Alabama', 85, 65),
        ('Auburn', 'Michigan St.', 70, 64),
        ('Houston', 'Tennessee', 69, 50),
    ],
    'Final Four': [
        ('Florida', 'Auburn', 79, 73),
        ('Houston', 'Duke', 70, 67),
    ],
}

def apply_completed_results(bracket, completed_results):
    """Remove teams eliminated in completed rounds from the bracket, ensuring seed integrity."""
    eliminated_teams = set()
    for round_results in completed_results.values():
        for winner, loser, _, _ in round_results:
            eliminated_teams.add(loser)

    for region in bracket:
        bracket[region] = [team for team in bracket[region] if team['team'] not in eliminated_teams]


### --- STATS COMPUTATION --- ###
def compute_tournament_stats(df):
    """Compute overall averages and standard deviations for radar metrics."""
    metrics = get_default_metrics()
    avgs = {m: df[m].mean() for m in metrics if m in df.columns}
    stdevs = {m: df[m].std() for m in metrics if m in df.columns}
    return avgs, stdevs

def compute_performance_text(team_row, t_avgs, t_stdevs):
    """
    Returns a dict with performance text and badge class based on the average z-score
    across default metrics. Inverts metrics for which lower is better (e.g. DEF EFF).
    Thresholds are aligned with interpretive_insights to reduce 'false WEAK' labels.
    """
    metrics = get_default_metrics()  # same function as in your code
    z_vals = []
    for m in metrics:
        if m in team_row and m in t_avgs and m in t_stdevs:
            std = t_stdevs[m] if t_stdevs[m] > 1e-12 else 1e-6
            team_val = team_row[m]
            mean_val = t_avgs[m]
            z = (team_val - mean_val) / std
            # Invert if lower is better:
            if m in ["DEF EFF", "KP_AdjD", "TO/GM", "KP_SOS_AdjEM"]:
                z = -z
            z_vals.append(z)

    if not z_vals:
        # No data => default to "MID"
        return {"text": "NO DATA", "class": "badge-mid"}

    avg_z = sum(z_vals) / len(z_vals)

    # Aligned thresholds with interpretive_insights
    if avg_z >= 1.0:
        return {"text": "ELITE", "class": "badge-elite"}
    elif avg_z >= 0.3:
        return {"text": "SOLID", "class": "badge-solid"}
    elif avg_z > -0.3:
        return {"text": "MID", "class": "badge-mid"}
    elif avg_z > -1.0:
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
        #conf['CONFERENCE'] = ['CONFERENCE'].apply(get_conf_logo_html)
        seed_str = ""
        if "SEED_25" in row and not pd.isna(row["SEED_25"]):
            seed_str = f" | SEED #{int(row['SEED_25'])}"
        perf_data = compute_performance_text(row, t_avgs, t_stdevs)
        subplot_titles.append(f"{team_name} | {conf}{seed_str}")
    fig = make_subplots(
        rows=row_count, cols=col_count,
        specs=[[{'type': 'polar'}] * col_count for _ in range(row_count)],
        subplot_titles=subplot_titles,

        horizontal_spacing=0.15,
        vertical_spacing=0.2
    )
    fig.update_layout(
        #title=dict(title=subplot_titles, x=0.5, y=0.95, xanchor='center', yanchor='top'),
        height=fig_height,
        template='plotly_dark',
        font=dict(family="Arial, sans-serif", size=12, color='white'),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=.95, xanchor="right", x=1, bgcolor="rgba(0,0,0,0.1)"),
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
# TREEMAP
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
                #f"<b>Record:</b> {int(x['WIN_25'])}-{int(x['LOSS_25'])}<br>"
                f"<b>KenPom AdjEM:</b> {x['KP_AdjEM']:.1f}<br>"
                f"<b>ESPN BPI:</b> {x['BPI_25']:.1f}<br>"
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
            title="<b>2025 KenPom AdjEM by Conference (Top 150)</b>"
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

def get_matchups_by_round():
    """
    Returns the standard NCAA bracket matchups (seed pairings or indices) per round.
    Indices assume winners from the previous round are ordered correctly.
    """
    round_64 = { # Seed pairings
        'West':    [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)],
        'East':    [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)],
        'South':   [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)],
        'Midwest': [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
    }
    # Subsequent rounds use indices relative to the winners of the previous round
    round_32 = { # Index pairings from R64 winners (0=1/16 winner, 1=8/9 winner, etc.)
        'West':    [(0,1),(2,3),(4,5),(6,7)],
        'East':    [(0,1),(2,3),(4,5),(6,7)],
        'South':   [(0,1),(2,3),(4,5),(6,7)],
        'Midwest': [(0,1),(2,3),(4,5),(6,7)]
    }
    sweet_16 = { # Index pairings from R32 winners
        'West':    [(0,1),(2,3)],
        'East':    [(0,1),(2,3)],
        'South':   [(0,1),(2,3)],
        'Midwest': [(0,1),(2,3)]
    }
    elite_8 = { # Index pairings from S16 winners
        'West':    [(0,1)],
        'East':    [(0,1)],
        'South':   [(0,1)],
        'Midwest': [(0,1)]
    }
    # Final Four uses indices relative to the list of region winners [West, East, South, Midwest]
    final_four   = [(0,1), (2,3)]  # (West vs East, South vs Midwest)
    # Championship uses indices relative to the Final Four winners
    championship = [(0,1)] # (Winner of WE vs Winner of SM)
    return round_64, round_32, sweet_16, elite_8, final_four, championship

def simulate_game(team1, team2):
    """
    Simulate one game between team1 and team2.
    Returns a *copy* of the winner's dictionary. Handles missing 'team' key.
    """
    # Ensure 'team' key exists for logging/debugging - use index name or default if missing
    t1_name = team1.get('team', 'Unknown Team 1')
    t2_name = team2.get('team', 'Unknown Team 2')

    # Ensure necessary keys for probability calculation exist, even if using defaults
    if 'KP_AdjEM' not in team1 or 'KP_AdjEM' not in team2 or 'seed' not in team1 or 'seed' not in team2:
        sim_logger.warning(f"Missing critical data (KP_AdjEM or seed) for matchup: {t1_name} vs {t2_name}. Using 50/50 prob.")
        pA = 0.5
    else:
        pA = calculate_win_probability(team1, team2)

    # Determine winner based on calculated probability
    winner = team1 if random.random() < pA else team2
    loser = team2 if winner is team1 else team1
    sim_logger.info(f"Sim Game: {t1_name} vs {t2_name}. Winner: {winner.get('team', 'Unknown')}. Prob T1 ({t1_name}): {pA:.2f}")

    # Return a deep copy to prevent modifications affecting subsequent rounds
    return copy.deepcopy(winner)

def simulate_tournament(bracket, num_simulations=1000):
    """
    Runs multiple full-bracket simulations from Round of 64 to Championship
    based on the provided bracket structure and returns aggregated win probabilities.
    """
    if not bracket or len(bracket) != 4:
        sim_logger.error("Invalid or incomplete bracket provided to simulate_tournament. Expected 4 regions.")
        return {} # Return empty dict if bracket is invalid

    round_64_pairs, round_32_pairs, sweet_16_pairs, elite_8_pairs, final_four_pairs, championship_pairs = get_matchups_by_round()

    rounds_list = [
        "Round of 64", "Round of 32", "Sweet 16", "Elite 8",
        "Final Four", "Championship", "Champion"
    ]
    # Initialize aggregator for win counts in each round
    results = {round_name: {} for round_name in rounds_list}
    # Track region champions separately
    results["Region"] = {region: {} for region in bracket.keys()}

    # Define the order for Final Four/Championship matchups
    region_order = ["West", "East", "South", "Midwest"] # MUST match the order assumed in final_four_pairs/championship_pairs

    sim_logger.info(f"Starting {num_simulations} tournament simulations...")

    for i in range(num_simulations):
        # --- Initialize Simulation Run ---
        # Deep copy the initial bracket state for this simulation run
        current_winners = {region: [copy.deepcopy(t) for t in teams] for region, teams in bracket.items()}
        round_advancers = {} # Stores winners for the *next* round

        # --- Simulate Regional Rounds ---
        for region in bracket.keys():
            # Round of 64
            r64_winners_region = []
            teams_r64 = current_winners[region]
            matchups = round_64_pairs.get(region, [])
            for (s1, s2) in matchups:
                t1 = next((t for t in teams_r64 if t.get('seed') == s1), None)
                t2 = next((t for t in teams_r64 if t.get('seed') == s2), None)
                if t1 and t2:
                    w = simulate_game(t1, t2)
                    w_name = w.get('team')
                    if w_name:
                        results["Round of 64"][w_name] = results["Round of 64"].get(w_name, 0) + 1
                        r64_winners_region.append(w)
                else:
                     sim_logger.warning(f"Sim {i}, R64, {region}: Missing team for seed pairing ({s1} vs {s2}).")
            round_advancers[region] = {'r32': r64_winners_region} # Store for next round

            # Round of 32
            r32_winners_region = []
            teams_r32 = round_advancers[region]['r32']
            matchups = round_32_pairs.get(region, [])
            for (idx1, idx2) in matchups:
                if idx1 < len(teams_r32) and idx2 < len(teams_r32):
                    t1, t2 = teams_r32[idx1], teams_r32[idx2]
                    w = simulate_game(t1, t2)
                    w_name = w.get('team')
                    if w_name:
                        results["Round of 32"][w_name] = results["Round of 32"].get(w_name, 0) + 1
                        r32_winners_region.append(w)
                else:
                    sim_logger.warning(f"Sim {i}, R32, {region}: Index out of bounds for pairing ({idx1} vs {idx2}).")
            round_advancers[region]['s16'] = r32_winners_region

            # Sweet 16
            s16_winners_region = []
            teams_s16 = round_advancers[region]['s16']
            matchups = sweet_16_pairs.get(region, [])
            for (idx1, idx2) in matchups:
                 if idx1 < len(teams_s16) and idx2 < len(teams_s16):
                    t1, t2 = teams_s16[idx1], teams_s16[idx2]
                    w = simulate_game(t1, t2)
                    w_name = w.get('team')
                    if w_name:
                        results["Sweet 16"][w_name] = results["Sweet 16"].get(w_name, 0) + 1
                        s16_winners_region.append(w)
                 else:
                    sim_logger.warning(f"Sim {i}, S16, {region}: Index out of bounds for pairing ({idx1} vs {idx2}).")
            round_advancers[region]['e8'] = s16_winners_region

            # Elite 8 (Regional Final)
            e8_winner_region = None
            teams_e8 = round_advancers[region]['e8']
            matchups = elite_8_pairs.get(region, [])
            for (idx1, idx2) in matchups: # Should only be one matchup
                if idx1 < len(teams_e8) and idx2 < len(teams_e8):
                    t1, t2 = teams_e8[idx1], teams_e8[idx2]
                    w = simulate_game(t1, t2)
                    e8_winner_region = w # This is the regional champion
                    w_name = w.get('team')
                    if w_name:
                        results["Elite 8"][w_name] = results["Elite 8"].get(w_name, 0) + 1
                        # Record the region champion win count
                        results["Region"][region][w_name] = results["Region"][region].get(w_name, 0) + 1
                else:
                    sim_logger.warning(f"Sim {i}, E8, {region}: Index out of bounds for pairing ({idx1} vs {idx2}).")
            round_advancers[region]['f4'] = e8_winner_region # Store the single region winner

        # --- Simulate Final Four and Championship ---
        # Collect region winners based on the defined region_order
        final_four_teams = [round_advancers.get(reg, {}).get('f4', None) for reg in region_order]

        # Check if all four regional winners are present
        if None in final_four_teams:
            sim_logger.warning(f"Sim {i}: Missing one or more regional champions for Final Four.")
            continue # Skip Final Four/Championship for this simulation run

        # Final Four
        ff_winners = []
        for (idx1, idx2) in final_four_pairs:
            t1, t2 = final_four_teams[idx1], final_four_teams[idx2]
            w = simulate_game(t1, t2)
            w_name = w.get('team')
            if w_name:
                results["Final Four"][w_name] = results["Final Four"].get(w_name, 0) + 1
                ff_winners.append(w)

        # Championship
        champion = None
        if len(ff_winners) == 2: # Should always be 2 if FF simulation was successful
             for (idx1, idx2) in championship_pairs: # Should only be one pairing (0, 1)
                t1, t2 = ff_winners[idx1], ff_winners[idx2]
                w = simulate_game(t1, t2)
                champion = w
                w_name = w.get('team')
                if w_name:
                    results["Championship"][w_name] = results["Championship"].get(w_name, 0) + 1
                    results["Champion"][w_name] = results["Champion"].get(w_name, 0) + 1
        else:
             sim_logger.warning(f"Sim {i}: Incorrect number of Final Four winners ({len(ff_winners)}), cannot simulate Championship.")

    sim_logger.info("Simulations finished. Converting counts to percentages...")

    # --- Convert Counts to Percentages ---
    # For advancing to each round
    for round_name in rounds_list:
        total_wins_in_round = sum(results[round_name].values())
        # Avoid division by zero if a round had no winners (shouldn't happen in full sim)
        # Normalization should be based on num_simulations * potential winners in that round,
        # but dividing by num_simulations gives probability of *reaching* that round.
        for team_name in results[round_name]:
            results[round_name][team_name] = (results[round_name][team_name] / num_simulations) * 100.0

    # For winning each region
    for region in results["Region"]:
        for team_name in results["Region"][region]:
            results["Region"][region][team_name] = (results["Region"][region][team_name] / num_simulations) * 100.0

    return results


def prepare_tournament_data(df):
    """
    Filters and prepares the main DataFrame for simulation.
    Extracts seeded teams (1-16) for the four main regions.
    Returns a bracket dictionary: {'Region': [list_of_team_dicts], ...}
    Each team_dict contains necessary stats for simulation.
    """
    # Define required and desired optional columns for simulation
    required_cols = ['SEED_25', 'REGION_25', 'KP_AdjEM']
    # Add more columns used in calculate_win_probability
    optional_cols = [
        'TM_KP', 'TEAM', # Need at least one identifier
        'BPI_25', 'NET_25', 'OFF EFF', 'DEF EFF', 'KP_AdjO', 'KP_AdjD',
        'WIN% ALL GM', 'WIN% CLOSE GM', 'AVG MARGIN', 'KP_SOS_AdjEM',
        'eFG%', 'OPP eFG%', 'TS%', 'OPP TS%', 'AST/TO%', 'TO/GM',
        'STOCKS/GM', 'STOCKS-TOV/GM', 'OFF REB/GM', 'DEF REB/GM',
        # Add ranks if available and potentially useful for thresholds
        'KP_Rank', 'KP_AdjO_Rk', 'KP_AdjD_Rk', 'KP_AdjEM_Rk', 'BPI_Rk_25',
        "FT%", "3PT%", "3PTA/GM", #"3PTM/GM",
        'CONFERENCE', # Make sure CONFERENCE is selected if used later
    ]

    # Check for required columns
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        sim_logger.error(f"FATAL: Missing required columns for simulation: {missing_req}. Cannot prepare bracket.")
        st.error(f"Data is missing required columns for simulation: {missing_req}")
        return None

    # Select columns, prioritize TM_KP as identifier if available
    sim_cols = list(set(required_cols + optional_cols))
    actual_sim_cols = [c for c in sim_cols if c in df.columns]

    # Ensure core identifiers are present before filtering
    if 'TM_KP' not in actual_sim_cols and 'TEAM' not in actual_sim_cols and (df.index.name is None or df.index.name.upper() != 'TEAM'):
         sim_logger.error("FATAL: No valid team identifier column (TM_KP or TEAM or index named TEAM) found in available columns.")
         st.error("Could not identify team names in the available data columns.")
         return None

    # Initial selection
    bracket_teams = df[actual_sim_cols].copy()

    # --- **NEW**: Check and handle duplicate columns ---
    if bracket_teams.columns.duplicated().any():
        duplicated_cols = bracket_teams.columns[bracket_teams.columns.duplicated()].unique()
        sim_logger.warning(f"Duplicate columns found in input data: {list(duplicated_cols)}. Keeping first instance.")
        # Keep the first instance of each column name
        bracket_teams = bracket_teams.loc[:, ~bracket_teams.columns.duplicated(keep='first')]
        # Update actual_sim_cols to reflect the columns that actually remain AFTER deduplication
        current_columns = bracket_teams.columns.tolist()
        actual_sim_cols = [col for col in actual_sim_cols if col in current_columns]
        # Re-check required columns AFTER deduplication
        missing_req_after_dedup = [c for c in required_cols if c not in bracket_teams.columns]
        if missing_req_after_dedup:
             sim_logger.error(f"FATAL: Required columns missing after duplicate removal: {missing_req_after_dedup}.")
             st.error(f"Data integrity issue: Required columns lost after handling duplicates: {missing_req_after_dedup}")
             return None
        sim_logger.info(f"Columns after deduplication: {bracket_teams.columns.tolist()}") # Log remaining columns
    # --- End New Section ---

    # Identify the team name column to use
    name_col_found = False
    if 'TM_KP' in bracket_teams.columns:
        name_col = 'TM_KP'
        name_col_found = True
    elif 'TEAM' in bracket_teams.columns:
        name_col = 'TEAM'
        name_col_found = True
    elif bracket_teams.index.name is not None and bracket_teams.index.name.upper() == 'TEAM':
         if 'TEAM' not in bracket_teams.columns:
             bracket_teams['TEAM'] = bracket_teams.index
         name_col = 'TEAM'
         name_col_found = True

    if not name_col_found:
         sim_logger.error("FATAL: No valid team identifier column could be confirmed after selection/deduplication.")
         st.error("Could not identify team names in the data.")
         return None

    # Use a consistent internal name ('TM_KP')
    if name_col != 'TM_KP':
        # Ensure 'TM_KP' doesn't already exist from a duplicate before renaming
        if 'TM_KP' in bracket_teams.columns and name_col in bracket_teams.columns:
             sim_logger.warning(f"Both '{name_col}' and 'TM_KP' exist. Dropping 'TM_KP' before renaming.")
             bracket_teams = bracket_teams.drop(columns=['TM_KP'])
             actual_sim_cols.remove('TM_KP') # Remove from list if it was there

        bracket_teams.rename(columns={name_col: 'TM_KP'}, inplace=True)
        # Update actual_sim_cols list after rename
        if 'TM_KP' not in actual_sim_cols:
             actual_sim_cols.append('TM_KP')
        if name_col in actual_sim_cols:
             actual_sim_cols.remove(name_col)

    # Ensure TM_KP is definitely in the list if it exists in the DataFrame
    if 'TM_KP' in bracket_teams.columns and 'TM_KP' not in actual_sim_cols:
         actual_sim_cols.append('TM_KP')


    # --- Data Cleaning and Preparation ---
    if 'SEED_25' in bracket_teams.columns:
        bracket_teams['SEED_25'] = pd.to_numeric(bracket_teams['SEED_25'], errors='coerce')
        bracket_teams.dropna(subset=['SEED_25'], inplace=True)
        bracket_teams['SEED_25'] = bracket_teams['SEED_25'].astype(int)
    else:
        sim_logger.error("FATAL: 'SEED_25' column missing, cannot prepare bracket.")
        st.error("Missing 'SEED_25' column.")
        return None

    bracket_teams = bracket_teams[(bracket_teams['SEED_25'] >= 1) & (bracket_teams['SEED_25'] <= 16)]

    if 'REGION_25' in bracket_teams.columns:
        bracket_teams['REGION_25'] = bracket_teams['REGION_25'].astype(str).str.strip()
    else:
        sim_logger.error("FATAL: 'REGION_25' column missing, cannot prepare bracket.")
        st.error("Missing 'REGION_25' column.")
        return None

    # --- Numeric Conversion Loop ---
    columns_to_skip_conversion = ['REGION_25', 'TM_KP', 'CONFERENCE', 'TEAM'] # Add potential original name 'TEAM' just in case
    sim_logger.info(f"Attempting numeric conversion. Skipping: {columns_to_skip_conversion}")
    sim_logger.info(f"Columns to process: {actual_sim_cols}")

    for col in actual_sim_cols:
        # Check existence and skip non-numeric
        if col in bracket_teams.columns and col not in columns_to_skip_conversion:
             sim_logger.debug(f"Converting column '{col}' to numeric.")
             try:
                 # Check the type before conversion attempt
                 if not isinstance(bracket_teams[col], (pd.Series, pd.Index)):
                      sim_logger.warning(f"Column '{col}' is type {type(bracket_teams[col])}, not Series/Index. Skipping conversion.")
                      continue # Skip this column if it's not a Series

                 bracket_teams[col] = pd.to_numeric(bracket_teams[col], errors='coerce')

             except TypeError as te:
                  # --- **NEW** Final Safety Net ---
                  sim_logger.error(f"*** Persistent TypeError converting column '{col}'. Type was: {type(bracket_teams[col])}. Skipping conversion for this column. Error: {te}")
                  # Optionally, you could try converting element by element if necessary, but skipping is safer now.
                  # For example:
                  # bracket_teams[col] = bracket_teams[col].apply(pd.to_numeric, errors='coerce')
                  continue # Skip to the next column
             except Exception as e:
                  sim_logger.error(f"Unexpected error converting column '{col}'. Error: {e}. Skipping conversion.")
                  continue # Skip to the next column
        elif col not in bracket_teams.columns:
             sim_logger.warning(f"Column '{col}' was in actual_sim_cols but not found in bracket_teams just before conversion loop.")
        else:
             sim_logger.debug(f"Skipping non-numeric column '{col}'.")


    # Define default values for potential NaNs AFTER conversion
    default_values = {
        'BPI_25': 0, 'NET_25': 150, 'OFF EFF': 100, 'DEF EFF': 100, 'KP_AdjO': 100, 'KP_AdjD': 100,
        'WIN% ALL GM': 0.5, 'WIN% CLOSE GM': 0.5, 'AVG MARGIN': 0, 'KP_SOS_AdjEM': 0,
        'eFG%': 0.5, 'OPP eFG%': 0.5, 'TS%': 0.5, 'OPP TS%': 0.5, 'AST/TO%': 1.0, 'TO/GM': 12,
        'STOCKS/GM': 8, 'STOCKS-TOV/GM': 0.7,
        'OFF REB/GM': 10, 'DEF REB/GM': 25,
        'KP_Rank': 300, 'KP_AdjO_Rk': 300, 'KP_AdjD_Rk': 300, 'KP_AdjEM_Rk': 300, 'BPI_Rk_25': 300,
        'FT%': 0.7, '3PT%': 0.33, '3PTA/GM': 20,
    }
    for col, default in default_values.items():
        if col in bracket_teams.columns:
            if pd.api.types.is_numeric_dtype(bracket_teams[col]):
                 bracket_teams[col].fillna(default)
            else:
                 # Log if a column expected to be numeric (because it has a default) isn't
                 sim_logger.warning(f"Column '{col}' was expected to be numeric for fillna but is type {bracket_teams[col].dtype}. Skipping fillna.")


    # --- Build Bracket Dictionary ---
    bracket = {}
    main_regions = ['West', 'East', 'South', 'Midwest']
    all_teams_added = []

    for region in main_regions:
        region_df = bracket_teams[bracket_teams['REGION_25'].str.lower() == region.lower()].copy()

        if region_df.empty:
            sim_logger.warning(f"No teams found for region '{region}'. Skipping this region.")
            continue

        if region_df['SEED_25'].duplicated().any():
             sim_logger.warning(f"Region '{region}' contains duplicate seeds. Deduplicating based on SEED, keeping first.")
             region_df = region_df.drop_duplicates(subset=['SEED_25'], keep='first')

        if len(region_df) != 16:
             sim_logger.warning(f"Region '{region}' has {len(region_df)} teams after potential deduplication, expected 16. Simulation may be incomplete.")

        region_df = region_df.sort_values('SEED_25')

        region_list = []
        for _, row in region_df.iterrows():
            # Check for duplicates across regions before processing
            team_name = row['TM_KP']
            if team_name in all_teams_added:
                sim_logger.warning(f"Team '{team_name}' (Seed {row['SEED_25']}, Region {region}) appears to be a duplicate across regions. Skipping this instance.")
                continue

            team_dict = row.to_dict()

            # Ensure key stats needed for simulation are floats/ints
            numeric_simulation_keys = list(default_values.keys()) + ['KP_AdjEM', 'SEED_25'] # Add essential keys
            for key in numeric_simulation_keys:
                if key in team_dict:
                    try:
                        numeric_val = pd.to_numeric(team_dict[key], errors='coerce')
                        if pd.isna(numeric_val):
                            team_dict[key] = float(default_values.get(key, 0.0))
                        else:
                             team_dict[key] = int(numeric_val) if key == 'SEED_25' else float(numeric_val)
                    except (ValueError, TypeError):
                        sim_logger.warning(f"Could not convert {key}='{team_dict[key]}' to numeric for team {team_name} during dict creation. Using default.")
                        team_dict[key] = float(default_values.get(key, 0.0))

            # Rename columns for simulation logic AFTER processing
            team_dict['team'] = team_dict.pop('TM_KP', 'Unknown Team')
            team_dict['seed'] = int(team_dict.pop('SEED_25', 99))

            region_list.append(team_dict)
            all_teams_added.append(team_name) # Mark team as added

        bracket[region] = region_list

    # Final checks after processing all regions
    if len(bracket) != 4:
         st.warning(f"Bracket preparation resulted in {len(bracket)} valid regions. Simulation might produce unexpected results.")
    elif len(all_teams_added) != 64:
         st.warning(f"Bracket preparation processed {len(all_teams_added)} unique teams, expected 64. Check for duplicate teams in source data or missing teams.")

    sim_logger.info("Bracket preparation finished.")
    return bracket


def calculate_win_probability(t1, t2):
    """
    H2H matchup win probability calculation.
    Integrates historical seed-based expectations with current efficiency metrics.
    Combines threshold evaluations and advanced analytics via logistic transformations.
    Slightly reduced base seeding bias.
    """

    # --- METRIC DIFFERENCES & THRESHOLD LOGIC --- #
    kp_diff = float(t1['KP_AdjEM']) - float(t2['KP_AdjEM'])
    bpi_diff = float(t1.get('BPI_25', 0)) - float(t2.get('BPI_25', 0))
    net_diff = float(t1.get('NET_25', 0)) - float(t2.get('NET_25', 0))
    t1_off = float(t1.get('OFF EFF', 1.0))
    t1_def = float(t1.get('DEF EFF', 1.0))
    t2_off = float(t2.get('OFF EFF', 1.0))
    t2_def = float(t2.get('DEF EFF', 1.0))
    off_advantage = t1_off - t2_def
    def_advantage = t1_def - t2_off
    adjO_diff = float(t1.get('KP_AdjO', 0)) - float(t2.get('KP_AdjO', 0))
    adjD_diff = float(t2.get('KP_AdjD', 0)) - float(t1.get('KP_AdjD', 0))
    # Schedule / Record
    win_pct_diff = (float(t1.get('WIN% ALL GM', 0.5)) - float(t2.get('WIN% ALL GM', 0.5)))
    close_pct_diff = (float(t1.get('WIN% CLOSE GM', 0.5)) - float(t2.get('WIN% CLOSE GM', 0.5)))
    margin_diff = float(t1.get('AVG MARGIN', 0)) - float(t2.get('AVG MARGIN', 0))
    sos_diff = float(t1.get('KP_SOS_AdjEM', 0)) - float(t2.get('KP_SOS_AdjEM', 0))
    # Experience / Intangibles
    exp_diff = (float(t1.get('TOURNEY_EXPERIENCE', 0)) - float(t2.get('TOURNEY_EXPERIENCE', 0))) * 0.5
    success_diff = (float(t1.get('TOURNEY_SUCCESS', 0)) - float(t2.get('TOURNEY_SUCCESS', 0)))
    # Shooting differentials
    efg_diff = float(t1.get('eFG%', 0.5)) - float(t2.get('eFG%', 0.5))
    # Ball control / Misc differentials
    ast_to_diff = float(t1.get('AST/TO%', 1.0)) - float(t2.get('AST/TO%', 1.0))
    net_ast_to_ratio_diff = float(t1.get('NET AST/TOV RATIO', 1.0)) - float(t2.get('NET AST/TOV RATIO', 1.0))
    stocks_to_diff = float(t1.get('STOCKS-TOV/GM', 1.0)) - float(t2.get('STOCKS-TOV/GM', 1.0))

    
    # --- Historical threshold constants (from KP metrics) --- #
    KP_AdjEM_Rk_THRESHOLD = 25 # actual=5.7; no outliers=3.3    
    KP_AdjO_Rk_THRESHOLD = 50 # actual=39      
    KP_AdjD_Rk_THRESHOLD = 35 # actual=25
    KP_AdjEM_champ_live = 22.5 # prior=21.5
    KP_AdjEM_champ_avg = 27.9 # prior=27.9      
    KP_AdjEM_champ_min = 20 # prior=19.1      
    KP_AdjEM_champ_max = 50 # prior=35.7      
    KP_AdjO_THRESHOLD = 112.5 # prior=115
    KP_AdjD_THRESHOLD = 102.5 # prior=100

    # --- FACTOR WEIGHTING --- #
    factor = 0
    factor += 0.2 * kp_diff            # KP_AdjEM difference
    factor += 0.2 * bpi_diff           # ESPN BPI_25 difference
    factor += 0.00 * net_diff           # NCAA NET_25 difference
    factor += 0.10 * off_advantage    
    factor += 0.10 * (-def_advantage) 
    factor += 0.00 * adjO_diff       
    factor += 0.00 * adjD_diff       
    factor += 0.00 * win_pct_diff * 100    
    factor += 0.05 * close_pct_diff * 100  
    factor += 0.15 * margin_diff      # AVG MARGIN difference
    factor += 0.00 * sos_diff        
    factor += 0.00 * exp_diff
    factor += 0.00 * success_diff
    factor += 0.05 * efg_diff * 100  
    factor += 0.05 * ast_to_diff
    factor += 0.00 * net_ast_to_ratio_diff
    factor += 0.05 * stocks_to_diff

    # --- THRESHOLD EVALUATION (using KP metrics) --- #
    # Adds bonus points based on meeting typical contender/champion thresholds
    def threshold_evaluation(team):
        score = 0
        # Use .get() with safe defaults
        kp_adjEM = float(team.get('KP_AdjEM', 0))
        kp_adjO = float(team.get('KP_AdjO', 0))
        kp_adjD = float(team.get('KP_AdjD', 150)) # High default for lower-is-better
        kp_adjEM_rk = float(team.get('KP_AdjEM_Rk', 999)) # High default for rank
        kp_adjO_rk = float(team.get('KP_AdjO_Rk', 999))
        kp_adjD_rk = float(team.get('KP_AdjD_Rk', 999))

        # AdjEM thresholds
        if kp_adjEM >= KP_AdjEM_champ_min: score += 0.1
        if kp_adjEM >= KP_AdjEM_champ_live: score += 0.1
        if kp_adjEM_rk <= KP_AdjEM_Rk_THRESHOLD: score += 0.1 # Bonus for being elite rank
        elif kp_adjEM_rk <= 20: score += 0.05 # Smaller bonus for being very good

        # AdjO thresholds
        if kp_adjO >= KP_AdjO_THRESHOLD: score += 0.1
        elif kp_adjO >= (KP_AdjO_THRESHOLD - 5): score += 0.05 # Near elite Offense
        if kp_adjO_rk <= KP_AdjO_Rk_THRESHOLD: score += 0.1

        # AdjD thresholds
        if kp_adjD <= KP_AdjD_THRESHOLD: score += 0.1
        elif kp_adjD <= (KP_AdjD_THRESHOLD + 5): score += 0.05 # Near elite Defense
        if kp_adjD_rk <= KP_AdjD_Rk_THRESHOLD: score += 0.1

        return score

    t1_threshold_score = threshold_evaluation(t1)
    t2_threshold_score = threshold_evaluation(t2)
    threshold_diff = t1_threshold_score - t2_threshold_score
    factor += 0.10 * threshold_diff # Add threshold difference to the overall factor

    # def_eff_diff = t2_def - t1_def  
    # factor += 0.05 * def_eff_diff

    # --- HISTORICAL SEED-BASED BASE PROBABILITY ---
    seed1_raw = t1.get('seed', 0)
    seed2_raw = t2.get('seed', 0)
    if pd.isna(seed1_raw): seed1_raw = 99
    if pd.isna(seed2_raw): seed2_raw = 99
    seed1 = int(seed1_raw)
    seed2 = int(seed2_raw)
    if seed1 < seed2:
        base_seed_prob = 0.51
    else:
        base_seed_prob = 0.49

    # --- APPLYING LOGISTIC TRANSFORMATION ---
    # The logistic function converts the weighted factor sum into a probability adjustment (0 to 1)
    # The divisor controls the steepness of the curve (sensitivity to factor changes)

    sensitivity = 10 # Lower value = more sensitive; Higher value = less sensitive
    adjustment_t1 = 1.0 / (1.0 + np.exp(-factor) / sensitivity)
    adjustment_t2 = 1.0 / (1.0 + np.exp(factor) / sensitivity)
    adjusted_t1 = base_seed_prob * adjustment_t1
    adjusted_t2 = (1 - base_seed_prob) * adjustment_t2
    total = adjusted_t1 + adjusted_t2
    final_prob = adjusted_t1 / total if total > 0 else base_seed_prob

    # --- SEED-BASED HISTORICAL UPSET PATTERNS (RELAXED & WEIGHTED) ---
    seed_diff = seed2 - seed1

    # Use a blend (weighted average) instead of a hard floor:
    if t1['seed'] == 1 and t2['seed'] == 16:
        final_prob = 0.7 * final_prob + 0.3 * 0.95
    elif t1['seed'] == 2 and t2['seed'] == 15:
        final_prob = 0.7 * final_prob + 0.3 * 0.85
    elif t1['seed'] == 3 and t2['seed'] == 14:
        final_prob = 0.7 * final_prob + 0.3 * 0.75
    elif t1['seed'] == 4 and t2['seed'] == 13:
        final_prob = 0.7 * final_prob + 0.3 * 0.70
    elif t1['seed'] == 5 and t2['seed'] == 12:
        final_prob = 0.7 * final_prob + 0.3 * 0.65
    elif t1['seed'] == 6 and t2['seed'] == 11:
        final_prob = 0.7 * final_prob + 0.3 * 0.60
    elif t1['seed'] == 7 and t2['seed'] == 10:
        final_prob = 0.7 * final_prob + 0.3 * 0.58
    elif (seed1, seed2) in [(8, 9), (9, 8)]:
        # For 8 vs 9, nudge toward a near coin flip:
        final_prob = 0.8 * final_prob + 0.2 * 0.50
    elif seed_diff > 8:
        # For larger gaps beyond 8, apply a single tiered adjustment:
        seed_factor = min(0.03 * seed_diff, 0.15)
        final_prob = min(final_prob + seed_factor, 0.90)

    # --- ROUND-BASED ADJUSTMENT ---
    # For later rounds (e.g. Round of 32 and beyond), slightly reduce the forced floor.
    # if current_round >= 2:
    #     final_prob = max(final_prob, 0.85)

    # --- FINAL SANITY CHECKS & LIGHTENED ADJUSTMENTS ---
    # Apply one final tier of adjustments if needed without double–counting:
    if seed_diff >= 10 and final_prob < 0.80:
        final_prob = min(final_prob + 0.05, 0.85)
    elif seed_diff >= 5 and final_prob < 0.65:
        final_prob = min(final_prob + 0.03, 0.75)
    elif seed_diff <= -5 and final_prob > 0.35:
        final_prob = max(final_prob - 0.03, 0.30)

    # Return final probability clamped to a slightly wider range for increased randomness.
    final_prob = max(0.05, min(0.95, final_prob))
    return final_prob

def run_games(team_list, pairing_list, round_name, region_name, use_analytics=True):
    """
    Simulate a series of matchups for the specified round.
    Returns a tuple (winners, round_games) where round_games contains detailed logs.
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
                'team1':      tA['team'],
                'seed1':      tA['seed'],
                'team2':      tB['team'],
                'seed2':      tB['seed'],
                'winner':     winner['team'],
                'winner_seed': winner['seed'],
                'win_prob':  pA if (winner is tA) else (1 - pA),
            })
            winners.append(winner)
    return winners, round_games

def generate_bracket_round(teams, round_num, region, use_analytics=True):
    """
    Given a list of teams in bracket order for a round, simulate each matchup.
    For round 1, use hard-coded pairings; for subsequent rounds, pair consecutively.
    """
    winners = []
    if round_num == 1:
        pairings = [(0,15), (7,8), (4,11), (3,12),
                    (5,10), (2,13), (6,9), (1,14)]
    else:
        pairings = [(i, i+1) for i in range(0, len(teams), 2)]
    for (i, j) in pairings:
        if i < len(teams) and j < len(teams):
            teamA = teams[i]
            teamB = teams[j]
            pA = calculate_win_probability(teamA, teamB) if use_analytics else 1/(1+np.exp(-(teamA['KP_AdjEM']-teamB['KP_AdjEM'])/10))
            winner = teamA if random.random() < pA else teamB
            w = winner.copy()
            w['win_prob'] = pA if (winner is teamA) else (1 - pA)
            winners.append(w)
    return winners

def simulate_region_bracket(teams, region_name, use_analytics=True):
    """
    Simulate a single region’s bracket (16 seeds) and return:
      - rounds_dict: {1:[r64 winners], 2:[r32 winners], 3:[Sweet 16 winners], 4:[Elite 8 winners], 5:[region champion] }
      - all_games: a list of game log dictionaries.
    """
    pairings_r64 = [(0,15),(7,8),(4,11),(3,12),(5,10),(2,13),(6,9),(1,14)]
    pairings_r32 = [(0,1),(2,3),(4,5),(6,7)]
    pairings_s16 = [(0,1),(2,3)]
    pairings_e8  = [(0,1)]

    r64_winners, g64 = run_games(teams, pairings_r64, "Round of 64", region_name, use_analytics)
    r32_winners, g32 = run_games(r64_winners, pairings_r32, "Round of 32", region_name, use_analytics)
    s16_winners, g16 = run_games(r32_winners, pairings_s16, "Sweet 16", region_name, use_analytics)
    e8_winners,  g8  = run_games(s16_winners, pairings_e8,  "Elite 8", region_name, use_analytics)
    region_champion = e8_winners[0] if e8_winners else None

    rounds_dict = {1: r64_winners, 2: r32_winners, 3: s16_winners, 4: e8_winners, 5: [region_champion] if region_champion else []}
    all_games = g64 + g32 + g16 + g8
    return rounds_dict, all_games

def simulate_final_four_and_championship(region_champions, use_analytics=True):
    """
    Given a dict of region champions, simulate the Final Four and Championship.
    Returns a tuple (champion_dict, final_games_list).
    """
    required = ["West", "East", "South", "Midwest"]
    for r in required:
        if r not in region_champions:
            return None, []
    west_champ    = region_champions["West"]
    east_champ    = region_champions["East"]
    south_champ   = region_champions["South"]
    midwest_champ = region_champions["Midwest"]

    sf1_prob   = calculate_win_probability(west_champ, east_champ) if use_analytics else 0.5
    sf1_winner = west_champ if random.random() < sf1_prob else east_champ
    sf1_game = {
        "round_name":  "Final Four",
        "region":      "Final Four",
        "team1":       west_champ["team"],
        "seed1":       west_champ["seed"],
        "team2":       east_champ["team"],
        "seed2":       east_champ["seed"],
        "winner":      sf1_winner["team"],
        "winner_seed": sf1_winner["seed"],
        "win_prob":    sf1_prob if sf1_winner is west_champ else (1 - sf1_prob),
    }

    sf2_prob   = calculate_win_probability(south_champ, midwest_champ) if use_analytics else 0.5
    sf2_winner = south_champ if random.random() < sf2_prob else midwest_champ
    sf2_game = {
        "round_name":  "Final Four",
        "region":      "Final Four",
        "team1":       south_champ["team"],
        "seed1":       south_champ["seed"],
        "team2":       midwest_champ["team"],
        "seed2":       midwest_champ["seed"],
        "winner":      sf2_winner["team"],
        "winner_seed": sf2_winner["seed"],
        "win_prob":    sf2_prob if sf2_winner is south_champ else (1 - sf2_prob),
    }

    final_prob = calculate_win_probability(sf1_winner, sf2_winner) if use_analytics else 0.5
    champion   = sf1_winner if random.random() < final_prob else sf2_winner
    champ_game = {
        "round_name":  "Championship",
        "region":      "Championship",
        "team1":       sf1_winner["team"],
        "seed1":       sf1_winner["seed"],
        "team2":       sf2_winner["team"],
        "seed2":       sf2_winner["seed"],
        "winner":      champion["team"],
        "winner_seed": champion["seed"],
        "win_prob":    final_prob if champion is sf1_winner else (1 - final_prob),
    }
    final_games = [sf1_game, sf2_game, champ_game]
    return champion, final_games

def run_simulation(use_analytics=True, simulations=1):
    """
    Runs a single simulation of the full tournament.
    Returns a list (one per simulation) with:
      { 'simulation_number': int,
        'region_results': dict,
        'region_champions': dict,
        'champion': dict,
        'all_games': list of game logs }
    """
    data = prepare_tournament_data(df_main)
    if not data:
        sim_logger.error("Failed to prepare bracket data.")
        return []
    # Here data is expected to provide region teams and names.
    # (Assuming elsewhere in your app you create data['region_teams'] and data['region_names'] as needed.)
    region_teams = {}  # using our bracket data directly
    region_names = []
    # Build bracket from available regions (only use main four)
    main_regions = ["East", "West", "South", "Midwest"]
    for reg in main_regions:
        if reg in data and len(data[reg]) == 16:
            region_teams[reg] = data[reg]
    all_sim_results = []
    for sim_num in range(simulations):
        region_results   = {}
        region_champions = {}
        all_games        = []
        for reg in main_regions:
            if reg in region_teams:
                bracket_rounds, bracket_games = simulate_region_bracket(region_teams[reg], reg, use_analytics)
                region_results[reg] = bracket_rounds
                if 5 in bracket_rounds and len(bracket_rounds[5]) > 0:
                    region_champions[reg] = bracket_rounds[5][0]
                all_games.extend(bracket_games)
        champion = None
        final_games = []
        if len(region_champions) == 4:
            champion, final_games = simulate_final_four_and_championship(region_champions, use_analytics=use_analytics)
            all_games.extend(final_games)
        sim_result = {
            'simulation_number': sim_num + 1,
            'region_results': region_results,
            'region_champions': region_champions,
            'champion': champion,
            'all_games': all_games
        }
        all_sim_results.append(sim_result)
    return all_sim_results

def run_tournament_simulation(bracket, num_sims=100):
    if not bracket:
        return {}
    aggregated_results = simulate_tournament(bracket, num_simulations=num_sims)
    return aggregated_results


def get_bracket_matchups():
    """
    Returns the bracket structure for each round by region,
    plus final four and championship pairings.
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
    final_four   = [(0,1), (2,3)]
    championship = [(0,1)]
    return round_64, round_32, sweet_16, elite_8, final_four, championship

# def run_simulation_once(df):
#     """
#     Run a single tournament simulation (one complete bracket) and return a list of detailed game logs.
#     """
#     r64, r32, s16, e8, f4, champ = get_bracket_matchups()
#     bracket = prepare_tournament_data(df)
#     #apply_completed_results(bracket, completed_results_2025)
#     if not bracket:
#         return []
#     current = {r: [copy.deepcopy(t) for t in bracket[r]] for r in bracket}
#     #current = copy.deepcopy(bracket)  # use bracket directly, not df_main
#     game_logs = []

def run_simulation_once(bracket):
    """
    Run a single tournament simulation (one complete bracket) and return a list of detailed game logs.
    """
    r64, r32, s16, e8, f4, champ = get_bracket_matchups()
    if not bracket:
        return []
    current = copy.deepcopy(bracket)
    #apply_completed_results(bracket, completed_results_2025)
    game_logs = []

    def record_game(rnd_name, region, tA, tB, w):
        upset = "UPSET" if w['seed'] > min(tA['seed'], tB['seed']) else ""
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
        if region not in current:
            continue
        winners = []
        for (s1, s2) in matchups:
            tA = next((x for x in current[region] if x['seed'] == s1), None)  # Use next() with None default
            tB = next((x for x in current[region] if x['seed'] == s2), None)
            if not tA or not tB:
                continue
            w = simulate_game(tA, tB)
            winners.append(w)
            game_logs.append(record_game("Round of 64", region, tA, tB, w))
        r64_winners[region] = winners
    # Round of 32
    r32_winners = {}
    for region, pairs in r32.items():
        if region not in r64_winners:
            continue
        winners = []
        region_list = r64_winners[region]
        for (i, j) in pairs:
            if i < len(region_list) and j < len(region_list):
                w = simulate_game(region_list[i], region_list[j])
                winners.append(w)
                game_logs.append(record_game("Round of 32", region, region_list[i], region_list[j], w))
        r32_winners[region] = winners
    # Sweet 16
    s16_winners = {}
    for region, pairs in s16.items():
        if region not in r32_winners:
            continue
        winners = []
        region_list = r32_winners[region]
        for (i, j) in pairs:
            if i < len(region_list) and j < len(region_list):
                w = simulate_game(region_list[i], region_list[j])
                winners.append(w)
                game_logs.append(record_game("Sweet 16", region, region_list[i], region_list[j], w))
        s16_winners[region] = winners
    # Elite 8
    e8_finalists = []
    for region, pairs in e8.items():
        if region not in s16_winners:
            continue
        region_list = s16_winners[region]
        for (i, j) in pairs:
            if i < len(region_list) and j < len(region_list):
                w = simulate_game(region_list[i], region_list[j])
                e8_finalists.append((region, w))
                game_logs.append(record_game("Elite 8", region, region_list[i], region_list[j], w))
    # Region champions
    region_champs = {}
    region_order = ['West', 'East', 'South', 'Midwest']
    for (reg, champ_dict) in e8_finalists:
        region_champs[reg] = champ_dict
    # Final Four
    ff_winners = []
    for (idxA, idxB) in f4:
        rA = region_order[idxA]
        rB = region_order[idxB]
        if rA not in region_champs or rB not in region_champs:
            continue
        w = simulate_game(region_champs[rA], region_champs[rB])
        ff_winners.append(w)
        game_logs.append(record_game("Final Four", "National Semifinal", region_champs[rA], region_champs[rB], w))
    # Championship
    champion = None
    for (i, j) in champ:
        if i < len(ff_winners) and j < len(ff_winners):
            champion = simulate_game(ff_winners[i], ff_winners[j])
            game_logs.append(record_game("Championship", "National Final", ff_winners[i], ff_winners[j], champion))
    return game_logs

def visualize_aggregated_results(aggregated_analysis):
    """
    Create Plotly charts summarizing:
      1) Championship win probabilities (horizontal bar chart).
      2) Upset percentages by round (vertical bar chart).
    Returns a tuple (fig_champ, fig_upsets).
    """
    # Championship Win Probabilities
    champ_df = aggregated_analysis.get('Champion')
    fig_champ = None
    if champ_df is not None and champ_df:
        champ_data = sorted(champ_df.items(), key=lambda x: x[1], reverse=True)[:10]
        teams = [x[0] for x in champ_data]
        probs = [x[1] for x in champ_data]
        fig_champ = px.bar(
            x=probs,
            y=teams,
            orientation='h',
            labels={'x': 'Win Probability (%)', 'y': 'Team'},
            title="Championship Win Probability (Top 10 Teams)",
            color=probs,
            color_continuous_scale='RdYlGn',
            template='plotly_dark'
        )
        fig_champ.update_yaxes(autorange="reversed")
        fig_champ.update_traces(text=[f"{p:.1f}%" for p in probs], textposition='outside')
    # Upset Percentage by Round
    upset_pct = aggregated_analysis.get('upset_pct_aggregated')
    fig_upsets = None
    if upset_pct is not None and not upset_pct.empty:
        df_upsets = pd.DataFrame({
            'Round': upset_pct.index,
            'Upset_PCT': upset_pct.values
        })
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
            template='plotly_dark'
        )
        fig_upsets.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_upsets.update_layout(yaxis_range=[0, df_upsets['Upset_PCT'].max()*1.2])
    return fig_champ, fig_upsets

def create_regional_prob_chart(region_df):
    """
    Plotly bar chart visualizing regional championship probabilities
    """
    if region_df is None or region_df.empty:
        return None
    top_teams_by_region = []
    for region in region_df['Region'].unique():
        region_data = region_df[region_df['Region'] == region].sort_values('Probability', ascending=False).head(8)
        top_teams_by_region.append(region_data)
    filtered_df = pd.concat(top_teams_by_region)
    fig = px.bar(filtered_df, x='Team', y='Probability', #color='Region',
                 barmode='group', facet_col='Region', facet_col_wrap=2,
                 labels={'Probability': 'WIN PROBABILITY', 'Team': 'TEAM'},
                 title='REGIONAL WIN PROBABILITY  (Top 8 per Region)',
                 color_discrete_sequence=px.colors.qualitative.G10,
                 template='plotly_dark')
    fig.update_layout(legend_title_text='Region', height=600,
                      margin=dict(t=80, l=50, r=50, b=100),
                      title_font=dict(size=18), xaxis_tickangle=-45)
    fig.update_yaxes(tickformat='.0%')
    for data in fig.data:
        fig.add_trace(go.Scatter(x=data.x, y=data.y,
                                 text=[f"{y:.1%}" for y in data.y],
                                 mode="text"))
    return fig

def display_simulation_results(single_run_logs):
    """
    Render a single-run game log with advanced styling.
    Expects a list of dictionaries with keys: round, region, matchup, winner, is_upset.
    """
    if not single_run_logs:
        st.warning("No single-run logs to display.")
        return
    rounds = [
        ("Round of 64", "⭐"),
        ("Round of 32", "⭐⭐"),
        ("Sweet 16", "⭐⭐⭐"),
        ("Elite 8", "⭐⭐⭐⭐"),
        ("Final Four", "⭐⭐⭐⭐⭐"),
        ("Championship", "🏆"),
    ]
    round_order = [r[0] for r in rounds]
    df = pd.DataFrame(single_run_logs)
    df["Round_idx"] = df["round"].apply(lambda r: round_order.index(r) if r in round_order else 999)
    df.sort_values(["Round_idx", "region"], inplace=True)
    df.drop(columns=["Round_idx"], inplace=True)
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
            color: #FF4500;
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
        """, unsafe_allow_html=True)
    st.markdown('<div class="single-sim-container">', unsafe_allow_html=True)
    for round_name, round_icon in rounds:
        subset = df[df["round"] == round_name]
        if subset.empty:
            continue
        st.markdown(f'<div class="round-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="round-title">{round_icon} {round_name} {round_icon}</div>',
            unsafe_allow_html=True
        )
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
                """, unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- App Header & Tabs ---
st.title(":primary[🏀 MARCH MADNESS 2025 -- NCAAM BASKETBALL 🏀]")

# if FinalFour25_logo:
#     st.image(FinalFour25_logo, use_container_width=True) #width=750

#st.subheader(":primary[2025 MARCH MADNESS -- NCAAM BASKETBALL -- RESEARCH HUB]")
st.subheader(":blue[_Cure your 🧠 BRACKET BRAIN 🧠 and propel yourself up the leaderboards_]")

tab_home, tab_H2H, tab_pred, tab_regions, tab_conf, tab_team, tab_radar = st.tabs(["🏀 HOME",  #🌐
                                                                          "📋 H2H MATCHUPS", 
                                                                          "🔮 PREDICTIONS",
                                                                          "🔥 REGIONAL HEATMAPS", #🌡️📍
                                                                          "🏆 CONFERENCE STATS", #🏅                                                                          
                                                                          "📊 TEAM METRICS", #📈📋📜📰📅
                                                                          "🕸️ RADAR CHARTS", #📡🧭
                                                                          ]) #🎱❓✅❌ ⚙️

# --- Home Tab ---
with tab_home:
    st.caption(":green[_DATA AS OF: 4/7/2025_]")
    # --- Top Upset Candidates Table for Sweet 16 ---
    st.markdown("### :primary[🏀 2025 CHAMPIONSHIP ODDS 🏀]")

    # Initialize bracket exactly once globally (if not yet initialized)
    if 'bracket' not in st.session_state:
        bracket = prepare_tournament_data(df_main)
        if bracket is not None:  # Only apply if bracket data is valid
            apply_completed_results(bracket, completed_results_2025)
            st.session_state['bracket'] = bracket
        else:
            st.error("Failed to prepare bracket data. Simulation cannot run.")
            st.stop()  # Stop further execution if bracket prep fails
    else:
        bracket = st.session_state['bracket']


    # 2025 GAME RESULTS -- Sweet 16 / Elite 8 / Final Four
    sweet_16_matchups = [
        ("Alabama", 2, "BYU", 6),
        ("Florida", 1, "Maryland", 4),
        ("Duke", 1, "Arizona", 4),
        ("Texas Tech", 3, "Arkansas", 10),
        ("Michigan St.", 2, "Mississippi", 6),
        ("Tennessee", 2, "Kentucky", 3),
        ("Auburn", 1, "Michigan", 5),
        ("Houston", 1, "Purdue", 4)
    ]

    elite_8_matchups = [
        ("Florida", 1, "Texas Tech", 3),
        ("Duke", 1, "Alabama", 2),
        ("Auburn", 1, "Michigan St.", 2),
        ("Houston", 1, "Tennessee", 2)
    ]

    final_four_matchups = [
        ("Duke", 1, "Houston", 1),
        ("Florida", 1, "Auburn", 1),
    ]

    if bracket is not None:
        upset_candidates = []

        for fav_team, fav_seed, dog_team, dog_seed in final_four_matchups: ## UPDATE FOR CURRENT ROUND ##
            # Fetch teams from the bracket data
            fav_data = next((team for region in bracket.values() for team in region if team['team'] == fav_team and team['seed'] == fav_seed), None)
            dog_data = next((team for region in bracket.values() for team in region if team['team'] == dog_team and team['seed'] == dog_seed), None)

            if fav_data and dog_data:
                # Modified logic for equal seeds: favor the team with higher AdjEM
                if fav_seed == dog_seed:
                    if fav_data.get('KP_AdjEM', -float('inf')) > dog_data.get('KP_AdjEM', -float('inf')):
                        stronger_team = fav_data
                        weaker_team = dog_data
                    elif fav_data.get('KP_AdjEM', -float('inf')) < dog_data.get('KP_AdjEM', -float('inf')):
                        stronger_team = dog_data
                        weaker_team = fav_data
                    else:  # If AdjEM is also equal, default to seed (shouldn't happen often)
                        stronger_team = fav_data if fav_seed < dog_seed else dog_data
                        weaker_team = dog_data if fav_seed < dog_seed else fav_data
                    upset_prob = calculate_win_probability(weaker_team, stronger_team)
                    if stronger_team is fav_data:
                        fav = fav_data['team']
                        dog = dog_data['team']
                    else:
                        fav = dog_data['team']
                        dog = fav_data['team']
                else:  # Standard seed comparison
                    upset_prob = calculate_win_probability(dog_data, fav_data)
                    fav = fav_data['team']
                    dog = dog_data['team']

                upset_candidates.append({
                    "MATCHUP": f"{fav_team} ({fav_seed}) vs {dog_team} ({dog_seed})",
                    "FAV": fav,
                    "FAV SEED": fav_seed,
                    "DOG": dog,
                    "DOG SEED": dog_seed,
                    "UPSET PROB (%)": round(upset_prob * 100, 1)
                })

        if upset_candidates:
            df_upsets = pd.DataFrame(upset_candidates)
            #df_upsets['FAV'] = df_upsets['FAV'].apply(get_team_logo_html)
            #df_upsets['DOG'] = df_upsets['DOG'].apply(get_team_logo_html)

            df_upsets = df_upsets.sort_values("UPSET PROB (%)", ascending=False).reset_index(drop=True)
            upset_styler = df_upsets.style.format({"UPSET PROB (%)": "{:.1f}"})\
                .background_gradient(subset=["UPSET PROB (%)"], cmap="RdYlGn")\
                .set_table_styles(advanced_table_styles + [index_style, cell_style, header])\
                .set_properties(**{"text-align": "center"})

            st.markdown(upset_styler.to_html(escape=False), unsafe_allow_html=True)
        else:
            st.info("No upset candidates.")
    else:
        st.error("Bracket data not available.")
    
    # # --- Top Upset Candidates Table in Round of 64 ---
    # st.markdown("### :primary[TOP UPSET CANDIDATES -- ROUND OF 64]")
    # bracket = prepare_tournament_data(df_main)
    # if bracket is not None:
    #     # Define the standard Round of 64 pairings (using seeding conventions)
    #     round_64_pairings = [(1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)]
    #     upset_candidates = []
    #     for region, teams in bracket.items():
    #         # Create a mapping from seed to team for the region
    #         team_by_seed = {team['seed']: team for team in teams}
    #         for pairing in round_64_pairings:
    #             seed_a, seed_b = pairing
    #             if seed_a in team_by_seed and seed_b in team_by_seed:
    #                 team_a = team_by_seed[seed_a]
    #                 team_b = team_by_seed[seed_b]
    #                 # The favorite is the team with the lower seed number
    #                 if seed_a < seed_b:
    #                     favorite = team_a
    #                     underdog = team_b
    #                 else:
    #                     favorite = team_b
    #                     underdog = team_a
    #                 # Calculate the upset probability: chance that the underdog beats the favorite
    #                 upset_prob = calculate_win_probability(underdog, favorite)
    #                 upset_candidates.append({
    #                     "MATCHUP": f"{team_a['team']} ({seed_a}) vs {team_b['team']} ({seed_b})",
    #                     "REGION": region,
    #                     "FAV": favorite['team'],
    #                     "FAV SEED": favorite['seed'],
    #                     "DOG": underdog['team'],
    #                     "DOG SEED": underdog['seed'],
    #                     "UPSET PROB (%)": round(upset_prob * 100, 1)
    #                 })
    #     if upset_candidates:
    #         df_upsets = pd.DataFrame(upset_candidates)
    #         df_upsets = df_upsets.sort_values("UPSET PROB (%)", ascending=False).reset_index(drop=True)
    #         upset_styler = df_upsets.style.format({"UPSET PROB (%)": "{:.1f}"})\
    #             .background_gradient(subset=["UPSET PROB (%)"], cmap="RdYlGn")\
    #             .set_table_styles(detailed_table_styles)\
    #             .set_properties(**{"text-align": "center"})
    #         st.markdown(upset_styler.to_html(escape=False), unsafe_allow_html=True)
    #     else:
    #         st.info("No upset candidates found for Round of 64.")
    # else:
    #     st.error("Bracket data not available.")



#     selected_team = st.selectbox(
#         ":green[_SELECT A TEAM:_]",
#         options=[""] + sorted(df_main["TM_KP"].dropna().unique().tolist()),
#         index=0,
#         key="select_team_home"
#     )


#     if selected_team:
#         team_data = df_main[df_main["TM_KP"] == selected_team].copy()
#         if not team_data.empty:
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown(f"### {selected_team}")
#                 conf = team_data["CONFERENCE"].iloc[0] if "CONFERENCE" in team_data.columns else "N/A"
#                 record = f"{int(team_data['WIN_25'].iloc[0])}-{int(team_data['LOSS_25'].iloc[0])}" if "WIN_25" in team_data.columns and "LOSS_25" in team_data.columns else "N/A"
#                 seed_info = f"Seed: {int(team_data['SEED_25'].iloc[0])}" if "SEED_25" in team_data.columns and not pd.isna(team_data['SEED_25'].iloc[0]) else ""
#                 kp_rank = f"KenPom Rank: {int(team_data['KP_Rank'].iloc[0])}" if "KP_Rank" in team_data.columns else ""

#                 st.markdown(f"""
#                 **Conference:** {conf}  
#                 **Record:** {record}  
#                 {seed_info}  
#                 {kp_rank}
#                 """)

#                 # Overall performance badge (existing logic)
#                 if all(m in team_data.columns for m in get_default_metrics()):
#                     t_avgs, t_stdevs = compute_tournament_stats(df_main)
#                     perf_data = compute_performance_text(team_data.iloc[0], t_avgs, t_stdevs)
#                     st.markdown(f"""
#                     <div style='text-align: center; margin: 20px 0;'>
#                         <span class='{perf_data["class"]}' style='font-size: 18px; padding: 8px 16px;'>
#                             Overall Rating: {perf_data["text"]}
#                         </span>
#                     </div>
#                     """, unsafe_allow_html=True)

#                 # "Interpretive Insights" block - provides a short textual breakdown of how team compares relative to NCAA
#                 radar_metrics = get_default_metrics()  # e.g. ['AVG MARGIN','KP_AdjEM','OFF EFF','DEF EFF','AST/TO%','STOCKS-TOV/GM']
#                 existing_metrics = [m for m in radar_metrics if m in team_data.columns]
#                 if existing_metrics:
#                     with st.expander("📊 Interpretive Insights"):
#                         insights = []
#                         for metric in get_default_metrics():
#                             if metric in team_data.columns:
#                                 mean_val = t_avgs[metric]
#                                 std_val = max(t_stdevs[metric], 1e-6)
#                                 team_val = team_data.iloc[0][metric]
#                                 z = (team_val - mean_val) / std_val
#                                 if metric in ["DEF EFF", "TO/GM"]:
#                                     z = -z
#                                 if abs(z) < 0.3:
#                                     insights.append(f"**{metric}** | Near NCAA average.")
#                                 elif z >= 1.0:
#                                     insights.append(f"**{metric}** | Clear strength.")
#                                 elif 0.3 <= z < 1.0:
#                                     insights.append(f"**{metric}** | Above NCAA average.")
#                                 elif -1.0 < z <= -0.3:
#                                     insights.append(f"**{metric}** | Below NCAA average.")
#                                 else:
#                                     insights.append(f"**{metric}** | Notable weakness.")

#                         # Display insights as bullet points
#                         st.markdown("**Team Metric Highlights:**")
#                         for line in insights:
#                             st.write(f"- {line}")
                        
#             with col2:
#                 # Radar chart of the selected team
#                 key_metrics = ["KP_AdjEM", "OFF EFF", "DEF EFF", "TS%", "OPP TS%", "AST/TO%", "STOCKS/GM", "AVG MARGIN"]
#                 available_metrics = [m for m in key_metrics if m in team_data.columns]
#                 # Reuse your existing create_radar_chart function if desired:
#                 radar_fig = create_radar_chart([selected_team], df_main)
#                 if radar_fig:
#                     st.plotly_chart(radar_fig, use_container_width=True)
            
#             with st.expander("View All Team Metrics"):
#                 detailed_metrics = [
#                     "KP_Rank", "KP_AdjEM", "KP_SOS_AdjEM", 
#                     "OFF EFF", "DEF EFF", "WIN% ALL GM", "WIN% CLOSE GM",
#                     "PTS/GM", "OPP PTS/GM", "AVG MARGIN",
#                     "eFG%", "OPP eFG%", "TS%", "OPP TS%", 
#                     "AST/GM", "TO/GM", "AST/TO%", 
#                     "OFF REB/GM", "DEF REB/GM", "BLKS/GM", "STL/GM", 
#                     "STOCKS/GM", "STOCKS-TOV/GM"
#                 ]
#                 available_detailed = [m for m in detailed_metrics if m in team_data.columns]
#                 detail_df = team_data[available_detailed].T.reset_index()
#                 detail_df.columns = ["Metric", "Value"]

#                 # Format numeric columns
#                 def _fmt(v):
#                     if isinstance(v, float):
#                         return f"{v:.2f}"
#                     else:
#                         return str(v)

#                 detail_df["Value"] = detail_df["Value"].apply(_fmt)

#                 # Convert to a Styler for advanced CSS
#                 detail_styler = (
#                     detail_df.style
#                     .set_properties(**{"text-align": "center"})
#                     .set_table_styles([
#                         {
#                             "selector": "th",
#                             "props": [
#                                 ("background-color", "#0360CE"),
#                                 ("color", "white"),
#                                 ("font-weight", "bold"),
#                                 ("text-align", "center"),
#                                 ("padding", "6px 12px"),
#                                 ("border", "1px solid #222")
#                             ]
#                         },
#                         {
#                             "selector": "td",
#                             "props": [
#                                 ("text-align", "center"),
#                                 ("border", "1px solid #ddd"),
#                                 ("padding", "5px 10px")
#                             ]
#                         },
#                         {
#                             "selector": "table",
#                             "props": [
#                                 ("border-collapse", "collapse"),
#                                 ("border", "2px solid #222"),
#                                 ("border-radius", "8px"),
#                                 ("overflow", "hidden"),
#                                 ("box-shadow", "0 4px 12px rgba(0, 0, 0, 0.1)")
#                             ]
#                         },
#                     ])
#                 )

#                 st.markdown(detail_styler.to_html(), unsafe_allow_html=True)

# with tab_H2H:
#     st.header(":primary[TEAM REPORTS]")
#     st.caption(":green[_DATA AS OF: 4/7/2025_]")
#     # Allow team selection – similar to the Home tab approach
#     selected_team = st.selectbox(
#         ":green[_SELECT A TEAM:_]",
#         options=[""] + sorted(df_main["TM_KP"].dropna().unique().tolist()),
#         index=0,
#         key="select_team_reports"  # unique key for this selectbox
#     )
#     if selected_team:
#         team_data = df_main[df_main["TM_KP"] == selected_team].copy()
#         if not team_data.empty:
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown(f"### {selected_team}")
#                 conf = team_data["CONFERENCE"].iloc[0] if "CONFERENCE" in team_data.columns else "N/A"
#                 record = f"{int(team_data['WIN_25'].iloc[0])}-{int(team_data['LOSS_25'].iloc[0])}" if "WIN_25" in team_data.columns and "LOSS_25" in team_data.columns else "N/A"
#                 seed_info = f"Seed: {int(team_data['SEED_25'].iloc[0])}" if "SEED_25" in team_data.columns and not pd.isna(team_data['SEED_25'].iloc[0]) else ""
#                 kp_rank = f"KenPom Rank: {int(team_data['KP_Rank'].iloc[0])}" if "KP_Rank" in team_data.columns else ""
#                 st.markdown(f"""
#                 **Conference:** {conf}  
#                 **Record:** {record}  
#                 {seed_info}  
#                 {kp_rank}
#                 """)
#                 if all(m in team_data.columns for m in get_default_metrics()):
#                     t_avgs, t_stdevs = compute_tournament_stats(df_main)
#                     perf_data = compute_performance_text(team_data.iloc[0], t_avgs, t_stdevs)
#                     st.markdown(f"""
#                     <div style='text-align: center; margin: 20px 0;'>
#                         <span class='{perf_data["class"]}' style='font-size: 18px; padding: 8px 16px;'>
#                             Overall Rating: {perf_data["text"]}
#                         </span>
#                     </div>
#                     """, unsafe_allow_html=True)
#                 with st.expander("📊 Interpretive Insights"):
#                     insights = []
#                     for metric in get_default_metrics():
#                         if metric in team_data.columns:
#                             mean_val = t_avgs[metric]
#                             std_val = max(t_stdevs[metric], 1e-6)
#                             team_val = team_data.iloc[0][metric]
#                             z = (team_val - mean_val) / std_val
#                             if metric in ["DEF EFF", "TO/GM"]:
#                                 z = -z
#                             if abs(z) < 0.3:
#                                 insights.append(f"**{metric}** | Near NCAA average.")
#                             elif z >= 1.0:
#                                 insights.append(f"**{metric}** | Clear strength.")
#                             elif 0.3 <= z < 1.0:
#                                 insights.append(f"**{metric}** | Above NCAA average.")
#                             elif -1.0 < z <= -0.3:
#                                 insights.append(f"**{metric}** | Below NCAA average.")
#                             else:
#                                 insights.append(f"**{metric}** | Notable weakness.")
#                     st.markdown("**Team Metric Highlights:**")
#                     for line in insights:
#                         st.write(f"- {line}")
#             with col2:
#                 radar_fig = create_radar_chart([selected_team], df_main)
#                 if radar_fig:
#                     st.plotly_chart(radar_fig, use_container_width=True)
#             with st.expander("View All Team Metrics"):
#                 detailed_metrics = [
#                     "KP_Rank", "KP_AdjEM", "KP_SOS_AdjEM", 
#                     "OFF EFF", "DEF EFF", "WIN% ALL GM", "WIN% CLOSE GM",
#                     "PTS/GM", "OPP PTS/GM", "AVG MARGIN",
#                     "eFG%", "OPP eFG%", "TS%", "OPP TS%", 
#                     "AST/GM", "TO/GM", "AST/TO%", 
#                     "OFF REB/GM", "DEF REB/GM", "BLKS/GM", "STL/GM", 
#                     "STOCKS/GM", "STOCKS-TOV/GM"
#                 ]
#                 available_detailed = [m for m in detailed_metrics if m in team_data.columns]
#                 detail_df = team_data[available_detailed].T.reset_index()
#                 detail_df.columns = ["Metric", "Value"]
#                 detail_df["Value"] = detail_df.apply(
#                     lambda x: f"{x['Value']:.1f}" if isinstance(x['Value'], float) else x['Value'],
#                     axis=1
#                 )
#                 st.table(detail_df)
#         else:
#             st.warning("No data available for the selected team.")


#######################################
# -- TEAM REPORTS TAB (HEAD-TO-HEAD) --
#######################################

# with tab_H2H:
#     # Advanced CSS styling for the tab and table elements
#     st.markdown("""
#     <style>
#     /* Main container styling */
#     .team-report-container {
#         background: linear-gradient(135deg, #f8f9fa, #e9ecef);
#         border-radius: 12px;
#         padding: 24px;
#         box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
#         margin-bottom: 25px;
#         border: 1px solid rgba(0, 60, 200, 0.1);
#     }

#     /* Header styling with animated gradient line */
#     .header-with-line {
#         position: relative;
#         padding-bottom: 12px;
#         margin-bottom: 22px;
#         font-weight: 600;
#     }
#     .header-with-line:after {
#         content: "";
#         position: absolute;
#         bottom: 0;
#         left: 0;
#         width: 120px;
#         height: 4px;
#         background: linear-gradient(90deg, #0039A6, #87CEEB);
#         border-radius: 2px;
#         animation: gradient-flow 3s ease infinite;
#         background-size: 200% 200%;
#     }
#     @keyframes gradient-flow {
#         0% {background-position: 0% 50%;}
#         50% {background-position: 100% 50%;}
#         100% {background-position: 0% 50%;}
#     }

#     /* Team cards styling */
#     .team-card {
#         background: white;
#         border-radius: 10px;
#         padding: 16px;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
#         transition: transform 0.3s ease, box-shadow 0.3s ease;
#         height: 100%;
#     }
#     .team-card:hover {
#         transform: translateY(-5px);
#         box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
#     }

#     /* Team info section */
#     .team-info {
#         border-left: 4px solid #0039A6;
#         padding-left: 15px;
#         margin: 15px 0;
#     }

#     /* Performance badges with improved visuals */
#     .badge-elite { 
#         background: linear-gradient(135deg, #FFD700, #FFA500);
#         color: #000; 
#         border-radius: 20px; 
#         font-weight: bold; 
#         padding: 5px 14px; 
#         box-shadow: 0 3px 6px rgba(0,0,0,0.1);
#         text-shadow: 0px 1px 1px rgba(0,0,0,0.4);
#     }
#     .badge-solid { 
#         background: linear-gradient(135deg, #4CAF50, #388E3C); 
#         color: white; 
#         border-radius: 20px; 
#         font-weight: bold; 
#         padding: 5px 14px; 
#         box-shadow: 0 3px 6px rgba(0,0,0,0.1);
#     }
#     .badge-mid { 
#         background: linear-gradient(135deg, #2196F3, #1976D2); 
#         color: white; 
#         border-radius: 20px; 
#         font-weight: bold; 
#         padding: 5px 14px; 
#         box-shadow: 0 3px 6px rgba(0,0,0,0.1);
#     }
#     .badge-subpar { 
#         background: linear-gradient(135deg, #FF9800, #F57C00); 
#         color: white; 
#         border-radius: 20px; 
#         font-weight: bold; 
#         padding: 5px 14px; 
#         box-shadow: 0 3px 6px rgba(0,0,0,0.1);
#     }
#     .badge-weak { 
#         background: linear-gradient(135deg, #F44336, #D32F2F); 
#         color: white; 
#         border-radius: 20px; 
#         font-weight: bold; 
#         padding: 5px 14px; 
#         box-shadow: 0 3px 6px rgba(0,0,0,0.1);
#     }

#     /* Head-to-head comparison container */
#     .h2h-container {
#         background: linear-gradient(135deg, #f0f4f8, #e6eef5);
#         border-radius: 10px;
#         padding: 20px;
#         margin-top: 25px;
#         margin-bottom: 25px;
#         box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
#         border: 1px solid rgba(0, 60, 200, 0.08);
#     }

#     /* Stats table styling */
#     .stats-table {
#         width: 100%;
#         border-collapse: separate;
#         border-spacing: 0;
#         border-radius: 8px;
#         overflow: hidden;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
#     }
#     .stats-table thead th {
#         background-color: #0039A6;
#         color: white;
#         padding: 12px;
#         font-weight: 600;
#         position: sticky;
#         top: 0;
#         z-index: 10;
#     }
#     .stats-table tbody tr:nth-child(even) {
#         background-color: rgba(0, 0, 0, 0.02);
#     }
#     .stats-table tbody tr:hover {
#         background-color: rgba(33, 150, 243, 0.08);
#     }
#     .stats-table td {
#         padding: 10px 12px;
#         border-bottom: 1px solid #eaeaea;
#     }

#     /* Insights section */
#     .insights-container {
#         background-color: #f8f9fa;
#         border-radius: 8px;
#         padding: 18px;
#         margin-top: 20px;
#         border-left: 4px solid #0039A6;
#     }
#     .insights-list li {
#         margin-bottom: 8px;
#         padding-left: 10px;
#         position: relative;
#     }
#     .insights-list li:before {
#         content: "•";
#         color: #0039A6;
#         font-weight: bold;
#         position: absolute;
#         left: -10px;
#     }

#     /* Win probability indicator */
#     .win-prob-container {
#         margin: 20px 0;
#         padding: 15px;
#         border-radius: 8px;
#         background-color: #f8f9fa;
#         border: 1px solid #dee2e6;
#     }
#     .prob-meter {
#         height: 24px;
#         background: linear-gradient(to right, #F44336, #FFEB3B, #4CAF50);
#         border-radius: 12px;
#         position: relative;
#         overflow: hidden;
#         margin: 10px 0;
#     }
#     .prob-indicator {
#         position: absolute;
#         top: 0;
#         width: 5px;
#         height: 100%;
#         background-color: black;
#         z-index: 1;
#     }
#     .prob-text {
#         text-align: center;
#         font-weight: bold;
#         font-size: 16px;
#         margin-top: 5px;
#     }

#     /* Metric comparison indicators */
#     .metric-advantage {
#         font-weight: bold;
#     }
#     .team1-advantage {
#         color: #0039A6;
#     }
#     .team2-advantage {
#         color: #D32F2F;
#     }
#     .no-advantage {
#         color: #757575;
#     }

#     /* Responsive adjustments */
#     @media screen and (max-width: 768px) {
#         .team-card {
#             margin-bottom: 15px;
#         }
#         .stats-table {
#             font-size: 14px;
#         }
#         .header-with-line:after {
#             width: 80px;
#         }
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     st.header(":blue[TEAM REPORTS]")
#     st.caption(":green[_DATA AS OF: 4/7/2025_]")

#######################################
# -- TEAM REPORTS TAB (HEAD-TO-HEAD) --
#######################################
with tab_H2H:
    # Advanced CSS styling for  tab and table elements
    st.markdown("""
                <style>
                /* Main container styling */
                .team-report-container {
                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
                margin-bottom: 25px;
                border: 1px solid rgba(0, 60, 200, 0.1);
            }

            /* Header styling with animated gradient line */
            .header-with-line {
                position: relative;
                padding-bottom: 12px;
                margin-bottom: 22px;
                font-weight: 600;
            }
            .header-with-line:after {
                content: "";
                position: absolute;
                bottom: 0;
                left: 0;
                width: 120px;
                height: 4px;
                background: linear-gradient(90deg, #0039A6, #87CEEB);
                border-radius: 2px;
                animation: gradient-flow 3s ease infinite;
                background-size: 200% 200%;
            }
            @keyframes gradient-flow {
                0% {background-position: 0% 50%;}
                50% {background-position: 100% 50%;}
                100% {background-position: 0% 50%;}
            }

            /* Team cards styling */
            .team-card {
                background: white;
                border-radius: 10px;
                padding: 16px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                height: 100%;
            }
            .team-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
            }

            /* Team info section */
            .team-info {
                border-left: 4px solid #0039A6;
                padding-left: 15px;
                margin: 15px 0;
            }

            /* Performance badges with improved visuals */
            .badge-elite { 
                background: linear-gradient(135deg, #FFD700, #FFA500);
                color: #000; 
                border-radius: 20px; 
                font-weight: bold; 
                padding: 5px 14px; 
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
                text-shadow: 0px 1px 1px rgba(0,0,0,0.4);
            }
            .badge-solid { 
                background: linear-gradient(135deg, #4CAF50, #388E3C); 
                color: white; 
                border-radius: 20px; 
                font-weight: bold; 
                padding: 5px 14px; 
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            }
            .badge-mid { 
                background: linear-gradient(135deg, #2196F3, #1976D2); 
                color: white; 
                border-radius: 20px; 
                font-weight: bold; 
                padding: 5px 14px; 
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            }
            .badge-subpar { 
                background: linear-gradient(135deg, #FF9800, #F57C00); 
                color: white; 
                border-radius: 20px; 
                font-weight: bold; 
                padding: 5px 14px; 
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            }
            .badge-weak { 
                background: linear-gradient(135deg, #F44336, #D32F2F); 
                color: white; 
                border-radius: 20px; 
                font-weight: bold; 
                padding: 5px 14px; 
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            }

            /* Head-to-head comparison container */
            .h2h-container {
                background: linear-gradient(135deg, #f0f4f8, #e6eef5);
                border-radius: 10px;
                padding: 20px;
                margin-top: 25px;
                margin-bottom: 25px;
                box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
                border: 1px solid rgba(0, 60, 200, 0.08);
            }

            /* Stats table styling */
            .stats-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            }
            .stats-table thead th {
                background-color: #0039A6;
                color: white;
                padding: 12px;
                font-weight: 600;
                position: sticky;
                top: 0;
                z-index: 10;
            }
            .stats-table tbody tr:nth-child(even) {
                background-color: rgba(0, 0, 0, 0.02);
            }
            .stats-table tbody tr:hover {
                background-color: rgba(33, 150, 243, 0.08);
            }
            .stats-table td {
                padding: 10px 12px;
                border-bottom: 1px solid #eaeaea;
            }

            /* Main Insights Styling */
            .insights-container {
                background: linear-gradient(to bottom, #ffffff, #f8f9fa);
                border-radius: 12px;
                padding: 18px 20px;
                margin-top: 15px;
                border: 1px solid #e0e0e0;
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
                transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
                position: relative;
                overflow: hidden;
            }
            
            .insights-container:hover {
                box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
                transform: translateY(-3px);
            }
            
            .insights-container:before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 4px;
            }
            
            .team-insights:before {
                background: linear-gradient(90deg, #0039A6, #4B7BEC);
            }
            
            .opponent-insights:before {
                background: linear-gradient(90deg, #B91C1C, #F87171);
            }
            
            .insights-header {
                font-size: 1.3rem;
                font-weight: 600;
                color: #1E3A8A;
                padding-bottom: 8px;
                margin-bottom: 15px;
                border-bottom: 2px solid #eaecef;
            }
            
            .insights-list {
                list-style-type: none !important;
                padding-left: 0 !important;
                margin-bottom: 0 !important;
            }
            
            /* Style for each metric item */
            .metric-item {
                display: flex;
                align-items: flex-start;
                background-color: rgba(240, 247, 255, 0.5);
                margin-bottom: 10px;
                padding: 12px 14px;
                border-radius: 8px;
                transition: all 0.2s ease;
                border-left: 3px solid #0039A6;
                position: relative;
                overflow: hidden;
            }
            
            .metric-item:hover {
                background-color: rgba(225, 239, 254, 0.8);
                transform: translateX(4px);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            }
            
            .metric-icon {
                flex-shrink: 0;
                width: 28px;
                height: 28px;
                background-color: #0039A6;
                color: white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 12px;
                font-size: 0.85rem;
                font-weight: bold;
            }
            
            .opponent-insights .metric-icon {
                background-color: #B91C1C;
            }
            
            .metric-name {
                font-weight: 600;
                color: #1E293B;
                margin-bottom: 3px;
                font-size: 0.95rem;
                display: block;
            }
            
            .metric-comment {
                color: #475569;
                font-size: 0.88rem;
                line-height: 1.4;
            }
            
            /* Different status colors */
            .metric-strong {
                border-left-color: #059669;
            }
            
            .metric-strong .metric-icon {
                background-color: #059669;
            }
            
            .metric-average {
                border-left-color: #D97706;
            }
            
            .metric-average .metric-icon {
                background-color: #D97706;
            }
            
            .metric-weak {
                border-left-color: #DC2626;
            }
            
            .metric-weak .metric-icon {
                background-color: #DC2626;
            }
            
            /* Empty state styling */
            .insights-empty {
                display: block;
                padding: 16px;
                text-align: center;
                color: #6B7280;
                border: 1px dashed #d1d5db;
                border-radius: 8px;
                margin-top: 12px;
                background-color: rgba(249, 250, 251, 0.8);
            }
            
            /* Animation */
            @keyframes slideInRight {
                from { opacity: 0; transform: translateX(-15px); }
                to { opacity: 1; transform: translateX(0); }
            }
            
            .metric-item {
                animation: slideInRight 0.35s ease forwards;
                opacity: 0;
            }
            
            /* Staggered animations */
            .metric-item:nth-child(1) { animation-delay: 0.05s; }
            .metric-item:nth-child(2) { animation-delay: 0.1s; }
            .metric-item:nth-child(3) { animation-delay: 0.15s; }
            .metric-item:nth-child(4) { animation-delay: 0.2s; }
            .metric-item:nth-child(5) { animation-delay: 0.25s; }
            .metric-item:nth-child(6) { animation-delay: 0.3s; }
            .metric-item:nth-child(7) { animation-delay: 0.35s; }
            .metric-item:nth-child(8) { animation-delay: 0.4s; }
            
            /* Responsive */
            @media (max-width: 768px) {
                .insights-container {
                    padding: 14px;
                }
                .metric-item {
                    padding: 10px;
                }
            }

            /* Win probability indicator */
            .win-prob-container {
                margin: 20px 0;
                padding: 15px;
                border-radius: 8px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
            }
            .prob-meter {
                height: 24px;
                background: linear-gradient(to right, #F44336, #FFEB3B, #4CAF50);
                border-radius: 12px;
                position: relative;
                overflow: hidden;
                margin: 10px 0;
            }
            .prob-indicator {
                position: absolute;
                top: 0;
                width: 5px;
                height: 100%;
                background-color: black;
                z-index: 1;
            }
            .prob-text {
                text-align: center;
                font-weight: bold;
                font-size: 16px;
                margin-top: 5px;
            }

            /* Metric comparison indicators */
            .metric-advantage {
                font-weight: bold;
            }
            .team1-advantage {
                color: #0039A6;
            }
            .team2-advantage {
                color: #D32F2F;
            }
            .no-advantage {
                color: #757575;
            }

            /* Responsive adjustments */
            @media screen and (max-width: 768px) {
                .team-card {
                    margin-bottom: 15px;
                }
                .stats-table {
                    font-size: 14px;
                }
                .header-with-line:after {
                    width: 80px;
                }
            }
            </style>
            """, unsafe_allow_html=True)
    
    st.header(":primary[TEAM REPORTS]")
    st.caption(":green[_DATA AS OF: 4/7/2025_]")

    # -- TEAM & OPPONENT SELECTION --
    H2H_options = [""] + sorted(df_main["TM_KP"].dropna().unique().tolist())
    selected_team = st.selectbox(
        ":blue[_SELECT A TEAM:_]",
        options=H2H_options,
        index=H2H_options.index("Florida"), #Duke
        key="select_team_reports"
    )
    selected_opponent = st.selectbox(
        ":red[_COMPARE vs. OPPONENT:_]",
        options=H2H_options,
        index=H2H_options.index("Houston"),
        key="select_opponent_reports"
    )

    # ------------------------------------------------------
    # TEAM / OPPONENT OVERVIEWS & INSIGHTS
    # ------------------------------------------------------
    if selected_team:
        team_data = df_main[df_main["TM_KP"] == selected_team].copy()
        #team_data["TM_KP"] = team_data["TM_KP"].apply(get_team_logo_html)
        if team_data.empty:
            st.warning("No data found for selected team.")
        else:
            st.markdown(f"## :blue[_HEAD-TO-HEAD:_ {selected_team} vs. {selected_opponent}]")

            # Extract basic team info
            conf = team_data["CONFERENCE"].iloc[0] if "CONFERENCE" in team_data.columns else "N/A" #.apply(get_conf_logo_html)

            record = "N/A"
            if "WIN_25" in team_data.columns and "LOSS_25" in team_data.columns:
                w = int(team_data["WIN_25"].iloc[0])
                l = int(team_data["LOSS_25"].iloc[0])
                record = f"{w}-{l}"

            # record_info = ""
            # if "WIN% ALL GM" in team_data.columns and not pd.isna(team_data["WIN% ALL GM"].iloc[0]):
            #     win_pct_all_gm = int(team_data["WIN% ALL GM"].iloc[0])
            #     win_pct_info = f"RECORD {win_pct_all_gm}"

            seed_info = ""
            if "SEED_25" in team_data.columns and not pd.isna(team_data["SEED_25"].iloc[0]):
                seed_num = int(team_data["SEED_25"].iloc[0])
                seed_info = f"SEED #{seed_num}"
            # Rankings
            rankings = []
            if "KP_Rank" in team_data.columns and not pd.isna(team_data["KP_Rank"].iloc[0]):
                kp_rank = int(team_data["KP_Rank"].iloc[0])
                rankings.append(f"KenPom: #{kp_rank}")
            if "BPI_Rk_25" in team_data.columns and not pd.isna(team_data["BPI_Rk_25"].iloc[0]):
                bpi_rank = int(team_data["BPI_Rk_25"].iloc[0])
                rankings.append(f"BPI Rank: #{bpi_rank}")            
            if "NET_25" in team_data.columns and not pd.isna(team_data["NET_25"].iloc[0]):
                net_rank = int(team_data["NET_25"].iloc[0])
                rankings.append(f"NET: #{net_rank}")
            rankings_html = " | ".join(rankings) if rankings else "N/A"

            # Build list of key stats to display as bubbles
            key_stats = []
            # if "WIN% ALL GM" in team_data.columns:
            #     val = round(team_data["WIN% ALL GM"].iloc[0], 1)
            #     key_stats.append(("WIN% ALL GM", f"{val*100:.0f}%", "#333333"))
            if "AVG MARGIN" in team_data.columns:
                val = round(team_data["AVG MARGIN"].iloc[0], 1)
                key_stats.append(("AVG MARGIN", f"{val:.1f}", "#333333"))                
            if "KP_AdjEM" in team_data.columns:
                val = round(team_data["KP_AdjEM"].iloc[0], 1)
                key_stats.append(("KenPom AdjEM", val, "#2E8B57"))
            if "BPI_25" in team_data.columns:
                val = round(team_data["BPI_25"].iloc[0], 1)
                key_stats.append(("ESPN BPI", val, "#6A5ACD"))
            if "KP_AdjO" in team_data.columns:
                val = round(team_data["KP_AdjO"].iloc[0], 1)
                key_stats.append(("KenPom AdjO", val, "#1E90FF"))
            if "KP_AdjD" in team_data.columns:
                val = round(team_data["KP_AdjD"].iloc[0], 1)
                key_stats.append(("KenPom AdjD", val, "#DC143C"))
            if "OFF EFF" in team_data.columns:
                val = round(team_data["OFF EFF"].iloc[0], 2)
                key_stats.append(("TeamRankings OEff", val, "#008B8B"))
            if "DEF EFF" in team_data.columns:
                val = round(team_data["DEF EFF"].iloc[0], 2)
                key_stats.append(("TeamRankings DEff", val, "#B22222"))

            # OPPONENT STATS
            opp_data = df_main[df_main["TM_KP"] == selected_opponent].copy()
            #opp_data["TM_KP"] = opp_data["TM_KP"].apply(get_team_logo_html)

            opp_conf = opp_data["CONFERENCE"].iloc[0] if "CONFERENCE" in opp_data.columns else "N/A" #.apply(get_conf_logo_html)

            #opp_record = "N/A"
            if "WIN_25" in opp_data.columns and "LOSS_25" in opp_data.columns:
                w = int(opp_data["WIN_25"].iloc[0])
                l = int(opp_data["LOSS_25"].iloc[0])
            opp_record = f"{w}-{l}"
            opp_seed_info = ""
            if "SEED_25" in opp_data.columns and not pd.isna(opp_data["SEED_25"].iloc[0]):
                opp_seed_num = int(opp_data["SEED_25"].iloc[0])
                opp_seed_info = f"Seed {opp_seed_num}"
            # Rankings
            opp_rankings = []
            if "KP_Rank" in opp_data.columns and not pd.isna(opp_data["KP_Rank"].iloc[0]):
                opp_kp_rank = int(opp_data["KP_Rank"].iloc[0])
                opp_rankings.append(f"KenPom Rank: #{opp_kp_rank}")
            if "BPI_Rk_25" in opp_data.columns and not pd.isna(opp_data["BPI_Rk_25"].iloc[0]):
                opp_bpi_rank = int(opp_data["BPI_Rk_25"].iloc[0])
                opp_rankings.append(f"BPI Rank: #{opp_bpi_rank}")
            if "NET_25" in opp_data.columns and not pd.isna(opp_data["NET_25"].iloc[0]):
                opp_net_rank = int(opp_data["NET_25"].iloc[0])
                opp_rankings.append(f"NET: #{opp_net_rank}")
            opp_rankings_html = " | ".join(opp_rankings) if opp_rankings else "N/A"            


            opp_key_stats = []
            # if "WIN% ALL GM" in opp_data.columns:
            #     val = round(opp_data["WIN% ALL GM"].iloc[0], 1)
            #     opp_key_stats.append(("WIN% ALL GM", f"{val*100:.0f}%", "#333333"))
            if "AVG MARGIN" in opp_data.columns:
                val = round(opp_data["AVG MARGIN"].iloc[0], 1)
                opp_key_stats.append(("AVG MARGIN", f"{val:.1f}", "#333333"))                                              
            if "KP_AdjEM" in opp_data.columns:
                val = round(opp_data["KP_AdjEM"].iloc[0], 1)
                opp_key_stats.append(("KenPom AdjEM", val, "#2E8B57"))
            if "BPI_25" in opp_data.columns:
                val = round(opp_data["BPI_25"].iloc[0], 1)
                opp_key_stats.append(("ESPN BPI", val, "#6A5ACD"))
            if "KP_AdjO" in opp_data.columns:
                val = round(opp_data["KP_AdjO"].iloc[0], 1)
                opp_key_stats.append(("KenPom AdjO", val, "#1E90FF"))
            if "KP_AdjD" in opp_data.columns:
                val = round(opp_data["KP_AdjD"].iloc[0], 1)
                opp_key_stats.append(("KenPom AdjD", val, "#DC143C"))
            if "OFF EFF" in opp_data.columns:
                val = round(opp_data["OFF EFF"].iloc[0], 2)
                opp_key_stats.append(("TeamRankings OEff", val, "#008B8B"))
            if "DEF EFF" in opp_data.columns:
                val = round(opp_data["DEF EFF"].iloc[0], 2)
                opp_key_stats.append(("TeamRankings DEff", val, "#B22222"))
            


            # Lay out: left column for team info, right column for radar
            colA, colB = st.columns(2)
            with colA:
                # Card‐like container with corrected "team-info" div
                st.markdown(f"""
                <div class="team-info" style="border:1px solid #ccc; border-radius:6px; padding:12px; margin-bottom:12px;">
                  <h3 style="margin-bottom:5px;background: linear-gradient(135deg, #00539B, #FFFFFF); color: white;">{selected_team}</h3>
                  <p style="margin:2px 0;"><strong>Conference:</strong> {conf}</p>
                  <p style="margin:2px 0;"><strong>Record:</strong> {record} | {seed_info}</p>
                  <p style="margin:2px 0;"><strong>Rankings:</strong> {rankings_html}</p>

                  <!-- 3x3 bubble grid -->
                  <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; margin-top:12px;">
                """, unsafe_allow_html=True)

#                <h5 style="margin-top:15px; border-bottom:1px solid #eee; padding-bottom:5px;">KEY STATS</h5>

                # Render each stat bubble in a 3x3 grid
                for stat_name, stat_value, color in key_stats:
                    st.markdown(f"""

                                <div style="text-align:center;">
                                <div style="
                                    margin:auto; 
                                    background-color:{color}; 
                                    border-radius:50%; 
                                    width:55px; height:55px; 
                                    display:flex; 
                                    align-items:center; 
                                    justify-content:center;
                                    margin-bottom:5px;">
                                <span style="font-weight:bold; color:#fff;">{stat_value}</span>
                                </div>
                                <p style="font-size:0.85rem;">{stat_name}</p>
                                </div>
                                """, unsafe_allow_html=True)

                # Close the grid and team-info
                st.markdown("""
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Compute and show team performance badge
                if all(m in team_data.columns for m in get_default_metrics()):
                    badge = compute_performance_badge(team_data.iloc[0], df_main)
                    st.markdown(f"""
                                <div style='text-align: center; margin: 20px 0;'>
                                <span class='{badge["class"]}' style='font-size: 18px; padding: 8px 16px;'>
                                OVERALL RATING: {badge["text"]}
                                </span>
                                </div>                                                    
                                """, unsafe_allow_html=True)


# <div style="margin-top:10px;">
#                       <strong>OVERALL RATING:</strong> <span>{badge["text"]}</span>
#                       background: linear-gradient(135deg, #FFD700, #FFA500);
#                       border-radius: 20px; 
#                       font-weight: bold; 
#                       padding: 5px 14px; 
#                       box-shadow: 0 3px 6px rgba(0,0,0,0.1);
#                       text-shadow: 0px 1px 1px rgba(0,0,0,0.4);
#                     </div>

            # ------------------------------------------------------
            # HEAD-TO-HEAD COMPARISON (2-TEAM SECTION)
            # ------------------------------------------------------
            with colB:
                if selected_opponent and selected_opponent != selected_team:
                    if opp_data.empty:
                        st.warning("No data available for selected opponent.")
                    else:
                        st.markdown("---")
                                            
                    #st.markdown(f"#### {selected_opponent}")
                    st.markdown(f"""
                                <div class="team-info" style="border:1px solid #ccc; border-radius:6px; padding:12px; margin-bottom:12px;">
                                <h3 style="margin-bottom:5px;background: linear-gradient(135deg, #00539B, #FFFFFF); color: white;">{selected_opponent}</h3>
                                <p style="margin:2px 0;"><strong>Conference:</strong> {opp_conf}</p>
                                <p style="margin:2px 0;"><strong>Record:</strong> {opp_record} | {opp_seed_info}</p>
                                <p style="margin:2px 0;"><strong>Rankings:</strong> {opp_rankings_html}</p>

                                <!-- 3x3 bubble grid -->
                                <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; margin-top:12px;">
                    """, unsafe_allow_html=True)

                    # <h5 style="margin-top:15px; border-bottom:1px solid #eee; padding-bottom:5px;">KEY STATS</h5>
                    for stat_name, stat_value, color in opp_key_stats:
                        st.markdown(f"""

                                    <div style="text-align:center;">
                                    <div style="
                                        margin:auto; 
                                        background-color:{color}; 
                                        border-radius:50%; 
                                        width:60px; height:60px; 
                                        display:flex; 
                                        align-items:center; 
                                        justify-content:center;
                                        margin-bottom:5px;">
                                        <span style="font-weight:bold; color:#fff;">{stat_value}</span>
                                    </div>
                                    <p style="font-size:0.85rem;">{stat_name}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                #st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Close the grid and team-info
                    st.markdown("""
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Compute and show performance badge
                    if all(m in opp_data.columns for m in get_default_metrics()):
                        opp_badge = compute_performance_badge(opp_data.iloc[0], df_main)
                        st.markdown(f"""
                                    <div style='text-align: center; margin: 20px 0;'>
                                    <span class='{opp_badge["class"]}' style='font-size: 18px; padding: 8px 16px;'>
                                    OVERALL RATING: {opp_badge["text"]}
                                    </span>
                                    </div>                                                    
                            """, unsafe_allow_html=True)

        # --- Head-to-Head Stats Table ---
        with st.expander("H2H STATISTICAL COMPARISON"):
            h2h_metrics = [
                #"SEED_25",
                "BPI_25", "KP_AdjEM", #"KP_Rank", "KP_SOS_AdjEM",
                "KP_AdjO", "KP_AdjD",
                "OFF EFF", "DEF EFF",
                #"WIN% ALL GM",
                "AVG MARGIN",
                "PTS/GM", "OPP PTS/GM",
                "FT%", "3PT%", "3PTA/GM", #"3PTM/GM", 
                "NET_eFG%", #"eFG%", "OPP eFG%",
                "AST/TO%", "STOCKS-TOV/GM",
            ]
            #opp_conf['CONFERENCE'] = opp_conf['CONFERENCE'].apply(get_conf_logo_html)
            row_team = team_data.iloc[0]
            row_opp = opp_data.iloc[0]
            valid_df = df_main.dropna(subset=h2h_metrics, how="all")
            ncaa_avg = valid_df[h2h_metrics].mean(numeric_only=True)
            tourney_df = valid_df[valid_df["SEED_25"].notna()]
            if not tourney_df.empty:
                tourney_avg = tourney_df[h2h_metrics].mean(numeric_only=True)
            else:
                tourney_avg = pd.Series([np.nan]*len(h2h_metrics), index=h2h_metrics)

            final_df = pd.DataFrame({"METRIC": h2h_metrics})
            final_df[selected_team] = [row_team[m] if m in row_team else np.nan for m in h2h_metrics]
            final_df[selected_opponent] = [row_opp[m] if m in row_opp else np.nan for m in h2h_metrics]
            final_df["TOURNEY AVG"] = [tourney_avg[m] for m in h2h_metrics]
            final_df["NCAA AVG"] = [ncaa_avg[m] for m in h2h_metrics]
            

            lower_is_better = {
                "KP_Rank": True,
                "KP_AdjD": True,
                "KP_SOS_AdjEM": True,
                "DEF EFF": True,
                "OPP PTS/GM": True,
                "OPP eFG%": True,
                "OPP TS%": True,
                "TO/GM": True,
            }

            advantage_list = []
            for idx, row_ in final_df.iterrows():
                metric = row_["METRIC"]
                valA = row_[selected_team]
                valB = row_[selected_opponent]
                if pd.isna(valA) or pd.isna(valB):
                    advantage_list.append("N/A")
                    continue
                invert = lower_is_better.get(metric, False)
                if invert:
                    if valA < valB:
                        advantage_list.append(selected_team)
                    elif valB < valA:
                        advantage_list.append(selected_opponent)
                    else:
                        advantage_list.append("TIE")
                else:
                    if valA > valB:
                        advantage_list.append(selected_team)
                    elif valB > valA:
                        advantage_list.append(selected_opponent)
                    else:
                        advantage_list.append("TIE")
            final_df["ADVANTAGE"] = advantage_list

            adv_team = sum(1 for x in advantage_list if x == selected_team)
            adv_opp = sum(1 for x in advantage_list if x == selected_opponent)

            numeric_cols = [selected_team, selected_opponent, "TOURNEY AVG", "NCAA AVG"]
            for col in numeric_cols:
                final_df[col] = final_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x))

            def colorize_row(row_):
                metric = row_["METRIC"]
                invert = lower_is_better.get(metric, False)
                styles = []
                for c in final_df.columns:
                    if c in ["METRIC", "ADVANTAGE"]:
                        styles.append("")
                        continue
                    cell_val_str = row_[c]
                    try:
                        cell_val = float(cell_val_str)
                    except:
                        styles.append("")
                        continue
                    row_vals = []
                    for nc in numeric_cols:
                        try:
                            v = float(row_[nc])
                        except:
                            v = np.nan
                        if invert and not np.isnan(v):
                            v = -v
                        row_vals.append(v)
                    valid_vals = [v for v in row_vals if not np.isnan(v)]
                    if not valid_vals:
                        styles.append("")
                        continue
                    vmin, vmax = min(valid_vals), max(valid_vals)
                    if invert:
                        cell_val = -cell_val
                    ratio = 0.5 if vmax == vmin else (cell_val - vmin) / (vmax - vmin)
                    cmap = plt.cm.RdYlGn
                    rgba = cmap(ratio)
                    color_hex = mcolors.to_hex(rgba)
                    styles.append(f"background-color: {color_hex}; text-align: center;")
                return styles

            styled_h2h = final_df.style.apply(colorize_row, axis=1)
            styled_h2h = styled_h2h.set_properties(**{"text-align": "center"})
            #styled_h2h = styled_h2h.set_table_styles(detailed_table_styles)
            styled_h2h = styled_h2h.set_table_styles(advanced_table_styles + [index_style, cell_style, header])
            st.markdown(styled_h2h.to_html(), unsafe_allow_html=True)

            # --- Single-game Win Probability ---
            team_dict = {
                "team": selected_team,
                "seed": row_team.get("SEED_25", 99),
                "KP_AdjEM": row_team.get("KP_AdjEM", 0),
                "BPI_25": row_team.get("BPI_25", 0),
                "OFF EFF": row_team.get("OFF EFF", 1.00),
                "DEF EFF": row_team.get("DEF EFF", 1.00),
                "WIN% ALL GM": row_team.get("WIN% ALL GM", 0.5),
                "WIN% CLOSE GM": row_team.get("WIN% CLOSE GM", 0.5),
                "AVG MARGIN": row_team.get("AVG MARGIN", 0),
                "KP_SOS_AdjEM": row_team.get("KP_SOS_AdjEM", 0),
                "KP_AdjO": row_team.get("KP_AdjO", 0),
                "KP_AdjD": row_team.get("KP_AdjD", 0),
            }
            opp_dict = {
                "team": selected_opponent,
                "seed": row_opp.get("SEED_25", 99),
                "KP_AdjEM": row_opp.get("KP_AdjEM", 0),
                "BPI_25": row_opp.get("BPI_25", 0),
                "OFF EFF": row_opp.get("OFF EFF", 1.00),
                "DEF EFF": row_opp.get("DEF EFF", 1.00),
                "WIN% ALL GM": row_opp.get("WIN% ALL GM", 0.5),
                "WIN% CLOSE GM": row_opp.get("WIN% CLOSE GM", 0.5),
                "AVG MARGIN": row_opp.get("AVG MARGIN", 0),
                "KP_SOS_AdjEM": row_opp.get("KP_SOS_AdjEM", 0),
                "KP_AdjO": row_opp.get("KP_AdjO", 0),
                "KP_AdjD": row_opp.get("KP_AdjD", 0),
            }
            pA = calculate_win_probability(team_dict, opp_dict)

            if adv_team > adv_opp:
                summary_text = (
                    f"{selected_team} leads in {adv_team} metrics while "
                    f"{selected_opponent} leads in {adv_opp}. "
                    f"{selected_team} appears favored overall."
                )
            elif adv_opp > adv_team:
                summary_text = (
                    f"{selected_opponent} leads in {adv_opp} metrics while "
                    f"{selected_team} leads in {adv_team}. "
                    f"{selected_opponent} appears favored overall."
                )
            else:
                summary_text = (
                    f"Both teams appear evenly matched with {adv_team} metrics each. "
                    f"Looks to be an evenly-matched affair!"
                )

            st.markdown(f"""
            <p><strong>Win Probability:</strong> {selected_team} has a {pA*100:.1f}% chance to beat {selected_opponent}.</p>
            <p><strong>Summary:</strong> {summary_text}</p>
            """, unsafe_allow_html=True)

            # --- Opponent Basic Info & Stat Bubbles ---
            colH2H1, colH2H2 = st.columns(2)
            with colH2H1:
                # TEAM Radar Chart
                single_radar_fig = create_radar_chart([selected_team], df_main)
                if single_radar_fig:
                    st.plotly_chart(single_radar_fig, use_container_width=True)
            # --- Opponent Radar Chart ---
            with colH2H2:
                # OPPONENT Radar Chart
                compare_radar_fig = create_radar_chart([selected_opponent], df_main)
                if compare_radar_fig:
                    st.plotly_chart(compare_radar_fig, use_container_width=True)

            # --- Compute Opponent Interpretive Insights ---
            # Display BOTH teams' interpretive insights side-by-side
            colI1, colI2 = st.columns(2)

            # Helper function to determine metric status
            def get_metric_status(comment):
                comment = comment.lower()
                if any(term in comment for term in ["clear strength", "above ncaa", "excellent", "strong"]):
                    return "strong"
                elif any(term in comment for term in ["near ncaa", "average", "moderate"]):
                    return "average"
                else:
                    return "weak"

            # Helper function to get icon for metric
            def get_metric_icon(metric_name):
                icons = {
                    "AVG MARGIN": "🔒", #📌📊💯🏆🧱📈🔒⚡🛡️🔥
                    "KP_AdjEM": "🔒",
                    "BPI_25": "🔒",
                    "KP_AdjO": "🔒",
                    "KP_AdjD": "🔒",
                    "OFF EFF": "🔒",
                    "DEF EFF": "🔒",
                    "AST/TO%": "🔒",
                    "STOCKS-TOV/GM": "🔒"
                }
                
                for key in icons:
                    if key.lower() in metric_name.lower():
                        return icons[key]
                return "⚡" # Default icon

            with colI1:
                st.markdown(f"<div class='insights-header'>{selected_team} Insights</div>", unsafe_allow_html=True)
                team_insights_str = get_interpretive_insights(row_team, df_main)
                if team_insights_str:
                    st.markdown("<div class='insights-container team-insights'>", unsafe_allow_html=True)
                    for i, ins in enumerate(team_insights_str):
                        metric, comment = ins.split(" | ")
                        status = get_metric_status(comment)
                        icon = get_metric_icon(metric)
                        
                        st.markdown(
                            f"""<div class="metric-item metric-{status}">
                                <div class="metric-icon">{icon}</div>
                                <div class="metric-content">
                                    <span class="metric-name">{metric}</span>
                                    <span class="metric-comment">{comment}</span>
                                </div>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div class='insights-empty'>No interpretive insights available for this team.</div>", 
                        unsafe_allow_html=True
                    )

            with colI2:
                st.markdown(f"<div class='insights-header'>{selected_opponent} Insights</div>", unsafe_allow_html=True)
                opp_insights = get_interpretive_insights_opp(opp_data.iloc[0], df_main)
                if opp_insights:
                    st.markdown("<div class='insights-container opponent-insights'>", unsafe_allow_html=True)
                    for i, ins in enumerate(opp_insights):
                        metric, comment = ins.split(" | ")
                        status = get_metric_status(comment)
                        icon = get_metric_icon(metric)
                        
                        st.markdown(
                            f"""<div class="metric-item metric-{status}">
                                <div class="metric-icon">{icon}</div>
                                <div class="metric-content">
                                    <span class="metric-name">{metric}</span>
                                    <span class="metric-comment">{comment}</span>
                                </div>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div class='insights-empty'>No interpretive insights available for this opponent.</div>", 
                        unsafe_allow_html=True
                    )

            # Only show the single team's interpretive insights here if NO opponent is selected
            # (Prevents duplication once we show a 2-team comparison below.)
            if not selected_opponent or selected_opponent == selected_team:
                st.markdown(f"<div class='insights-header'>{selected_team} Insights</div>", unsafe_allow_html=True)
                team_insights_str = get_interpretive_insights(row_team, df_main)
                if team_insights_str:
                    st.markdown("<div class='insights-container team-insights'>", unsafe_allow_html=True)
                    st.markdown("<ul class='insights-list'>", unsafe_allow_html=True)
                    for i, ins in enumerate(team_insights_str):
                        metric, comment = ins.split(" | ")
                        st.markdown(
                            f"<li style='animation-delay: {0.1 * (i+1)}s'><strong>{metric}</strong> <span class='insights-comment'>{comment}</span></li>", 
                            unsafe_allow_html=True
                        )
                    st.markdown("</ul>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div class='insights-empty'><i class='fas fa-info-circle'></i> No interpretive insights available for this team.</div>", 
                        unsafe_allow_html=True
                    )

with tab_pred:
    st.header(":primary[BRACKET SIMULATION]")
    st.caption(":green[_DATA AS OF: 4/7/2025_]")

    show_logs = st.checkbox(":blue[_DISPLAY SINGLE-SIM LOGS?_]", value=True)

    if st.button(":green[RUN BRACKET SIMULATION]", icon="🏀"):
        with st.spinner(":green[RUNNING SIMULATIONS ...]"):

            # --- Bracket initialization and application of completed results ---
            if 'bracket' not in st.session_state:
                bracket = prepare_tournament_data(df_main)
                if bracket is not None:  # Only apply if bracket data is valid
                    #apply_completed_results(bracket, completed_results_2025)
                    st.session_state['bracket'] = bracket
                else:
                    st.error("Failed to prepare bracket data. Simulation cannot run.")
                    st.stop()  # Stop further execution if bracket prep fails
            else:
                bracket = st.session_state['bracket']

            # # Ensure your functions now directly use this bracket:
            # aggregated = run_tournament_simulation(bracket, num_sims=1000)
            # #single_run = run_simulation_once(bracket)
            # single_run = run_simulation_once(bracket) # st.session_state['bracket'])

                        # --- Run Simulations ---
            num_sims_aggregate = 1000 # Number of simulations for probability aggregation
            aggregated = simulate_tournament(bracket, num_simulations=num_sims_aggregate)

            # Run one simulation separately *only if* logs are requested
            single_run = None
            if show_logs:
                single_run = run_simulation_once(bracket)

            
            st.success("SIMULATIONS COMPLETE! A VICTOR HAS BEEN ANNOUNCED")


        # 1) If requested, show single-run logs first
        if show_logs and single_run:
            st.subheader(":blue[_Round-by-Round (Single Simulation)_]")
            display_simulation_results(single_run)
        else:
            st.info(":blue[_Single-run logs hidden. Check box above to display them._]")

        # 2) Now show aggregated results
        if not aggregated:
            st.error(":yellow[Aggregated simulation results failed to compile.]")
            st.stop()

        st.subheader(":primary[AGGREGATED SIMULATION RESULTS]")

        # A) Champion probabilities turned into a styled DataFrame
        champ_probs = aggregated.get("Champion", {})
        if not champ_probs:
            st.warning("Championship probabilities failed to calculate.")
            st.stop()

        # Build a table with columns: [Team,CHAMP%,Seed,Region,Conference,KP_AdjEM,NET_25]
        champion_items = sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)
        data_rows = []
        for team, pct in champion_items:
            row = {"TEAM": team, "CHAMP%": pct}
            subset = df_main[df_main["TM_KP"] == team]
            if not subset.empty:
                row["CONFERENCE"] = subset["CONFERENCE"].iloc[0] if "CONFERENCE" in subset.columns else ""
                row["REGION"] = subset["REGION_25"].iloc[0] if "REGION_25" in subset.columns else ""
                row["SEED"] = int(subset["SEED_25"].iloc[0]) if ("SEED_25" in subset.columns and not pd.isna(subset["SEED_25"].iloc[0])) else ""
                row["KP_AdjEM"] = subset["KP_AdjEM"].iloc[0] if "KP_AdjEM" in subset.columns else None
                row["BPI_25"]   = subset["BPI_25"].iloc[0]   if "BPI_25" in subset.columns else None
                row["NET_25"]   = subset["NET_25"].iloc[0]   if "NET_25" in subset.columns else None
                row["AVG MARGIN"]   = subset["AVG MARGIN"].iloc[0]   if "AVG MARGIN" in subset.columns else None
                row["AST/TO%"]   = subset["AST/TO%"].iloc[0]   if "AST/TO%" in subset.columns else None
                row["STOCKS-TOV/GM"]   = subset["STOCKS-TOV/GM"].iloc[0]   if "STOCKS-TOV/GM" in subset.columns else None
            data_rows.append(row)

        champion_df = pd.DataFrame(data_rows)
        champion_df["CHAMP%"] = champion_df["CHAMP%"].round(1)
        champion_df.rename(columns={"CHAMP%": "CHAMP PROBABILITY (%)"}, inplace=True)

        # Reorder columns
        champion_df = champion_df[["TEAM", "CHAMP PROBABILITY (%)", "SEED", "REGION", "CONFERENCE", "KP_AdjEM", "NET_25", "BPI_25"]]
        champion_df["CONFERENCE"] = champion_df["CONFERENCE"].apply(get_conf_logo_html)

        # Create a Styler
        champion_styler = (
            champion_df.style
            .format({
                "CHAMP PROBABILITY (%)": "{:.2f}",
                "KP_AdjEM": "{:.2f}",
                "NET_25": "{:.0f}",
                "BPI_25": "{:.1f}",
            })
            .background_gradient(
                cmap="RdYlGn", 
                subset=["CHAMP PROBABILITY (%)", "KP_AdjEM", "BPI_25"]
            ).background_gradient(
                cmap="RdYlGn_r", 
                subset=["SEED", "NET_25"]
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

        st.markdown(":blue[_HIGHEST CHAMPIONSHIP PROBABILTIES_]")
        st.markdown(champion_styler.to_html(), unsafe_allow_html=True)

        # Optional --  raw text summary
        # st.write("**Raw Summary**:")
        # for row in data_rows[:15]:
        #     st.write(f"{row['TEAM']}: {row['CHAMP%']:.1f}%")


        # --- REGIONAL SUBPLOT (2×2) --- #
        region_probs = aggregated.get("Region", None)
        if not region_probs:
            st.warning("No region champion data found.")
        else:
            st.markdown(f"### :primary[REGIONAL CHAMPION / FINAL FOUR PROBABILITY (%)]")
            fig_region = make_subplots(
                rows=2, cols=2,
                subplot_titles=["West", "East", "South", "Midwest"]
            )
            row_col_map = {"West": (1,1), "East": (1,2), "South": (2,1), "Midwest": (2,2)}
            for region_name in ["West", "East", "South", "Midwest"]:
                if region_name not in region_probs:
                    continue
                # region_probs[region_name] is a dict: { "TEAM": <float prob> }
                items = sorted(region_probs[region_name].items(),
                            key=lambda x: x[1], reverse=True)[:8]
                x_vals = [itm[0] for itm in items]
                y_vals = [itm[1] for itm in items]
                (r, c) = row_col_map[region_name]
                fig_region.add_trace(
                    go.Bar(
                        x=x_vals, y=y_vals,
                        name=region_name,
                        text=[f"{v:.1f}%" for v in y_vals],
                        textposition="outside",
                        #color='CONFERENCE',
                        #marker_color='CONFERENCE',
                        #marker_color="steelblue",

                    ),
                    row=r, col=c
                )
                fig_region.update_xaxes(tickangle=-45, row=r, col=c)
                if y_vals:
                    fig_region.update_yaxes(range=[0, max(y_vals)*1.2], row=r, col=c)

            fig_region.update_layout(
                template="plotly_dark",
                height=600,
                #title="REGIONAL CHAMPION / FINAL FOUR ODDS",
                showlegend=False,
                margin=dict(l=50, r=50, t=60, b=60),
            )
            st.plotly_chart(fig_region, use_container_width=True)

        # --- 1×1 CHAMPIONSHIP BAR CHART --- #
        st.markdown(f"### :primary[CHAMPIONSHIP PROBABILITY (%)]")
        top_champs = sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)[:12]
        fig_champ = go.Figure()
        fig_champ.add_trace(
            go.Bar(
                x=[tc[0] for tc in top_champs],
                y=[tc[1] for tc in top_champs],
                text=[f"{tc[1]:.1f}%" for tc in top_champs],
                textposition="outside",
                color='CONFERENCE',
                marker_color='CONFERENCE',                
                #marker_color="tomato",
            )
        )
        fig_champ.update_layout(
            template="plotly_dark",
            #title="CHAMPIONSHIP PROBABILITY (TOP 12 TEAMS)",
            xaxis=dict(tickangle=-45),
            yaxis=dict(range=[0, max([tc[1] for tc in top_champs])*1.15]),
            showlegend=False,
            margin=dict(l=20, r=20, t=80, b=60),
            height=450
        )
        st.plotly_chart(fig_champ, use_container_width=True)
    else:
        st.info("Run simulation to view results.")

# --- Radar Charts Tab ---
with tab_radar:
    st.header(":primary[REGIONAL RADAR CHARTS]")
    st.caption(":green[_DATA AS OF: 4/7/2025_]")
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
    st.caption(":green[_DATA AS OF: 4/7/2025_]")
    df_heat = df_main.copy()
    numeric_cols_heat = df_heat.select_dtypes(include=np.number).columns
    mean_series = df_heat.mean(numeric_only=True)
    mean_series = mean_series.reindex(df_heat.columns, fill_value=np.nan)
    df_heat.loc["TOURNEY AVG"] = mean_series
    df_heat['CONFERENCE'] = df_heat["CONFERENCE"].apply(get_conf_logo_html)
    df_heat_T = df_heat.T
    #df_heat_T = df_heat_T[core_cols]

    east_teams_2025 = [
    "Duke", "Alabama", "Wisconsin", "Arizona",
    "Oregon", "BYU", "Saint Mary's", "Mississippi St",
    "Baylor", "Vanderbilt", "VCU", "Liberty",
    "Akron", "Montana", "Robert Morris", "American",
    "TOURNEY AVG",
    ]
    west_teams_2025 = [
    "Florida", "St John's", "Texas Tech", "Maryland",
    "Memphis", "Missouri", "Kansas", "UConn",
    "Oklahoma", "Arkansas", "Drake", "Colorado St",
    "Grand Canyon", "NC Wilmington", "Omaha", "Norfolk St",
    "TOURNEY AVG",
    ]
    south_teams_2025 = [
    "Auburn", "Michigan St.", "Iowa St", "Texas A&M",
    "Michigan", "Mississippi", "Marquette", "Louisville",
    "Creighton", "New Mexico", "North Carolina", "UCSD",
    "Yale", "Lipscomb", "Bryant", "Alabama St.",
    "TOURNEY AVG",
    ]
    midwest_teams_2025 = [
    "Houston", "Tennessee", "Kentucky", "Purdue",
    "Clemson", "Illinois", "UCLA", "Gonzaga",
    "Georgia", "Utah St", "Texas", "McNeese",
    "High Point", "Troy", "Wofford", "SIUE",
    "TOURNEY AVG",
    ]
    regions = {
        "EAST REGION": east_teams_2025,
        "MIDWEST REGION": midwest_teams_2025,
        "SOUTH REGION": south_teams_2025,
        "WEST REGION": west_teams_2025
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
        "BPI_25": "RdYlGn",
        "KP_SOS_AdjEM": "RdYlGn_r",
        "OFF EFF": "RdYlGn",
        "DEF EFF": "RdYlGn_r",
        "AVG MARGIN": "RdYlGn",
        "eFG%": "RdYlGn",
        "TS%": "RdYlGn",
        "OPP TS%": "RdYlGn_r",
        "OPP eFG%": "RdYlGn_r",
        "AST/TO%": "RdYlGn",
        "STOCKS/GM": "RdYlGn",
        "STOCKS-TOV/GM": "RdYlGn",
        "WIN% ALL GM": "RdYlGn",
        "WIN% CLOSE GM": "RdYlGn",
        "NET_25": "RdYlGn_r",
        "SEED_25": "RdYlGn_r",
        "KP_AdjO": "RdYlGn",
        "KP_AdjD": "RdYlGn_r",
        "PTS/GM": "RdYlGn",
        "OPP PTS/GM": "RdYlGn",
        "OFF REB/GM": "RdYlGn",
        "DEF REB/GM": "RdYlGn",
        "STL/GM": "RdYlGn",
        "AST/GM": "RdYlGn",
        "TO/GM": "RdYlGn_r",

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
    st.caption(":green[_DATA AS OF: 4/7/2025_]")

    st.subheader(":primary[🏀 NCAAM BASKETBALL CONFERENCE TREEMAP 🏀]", divider='grey')
    treemap = create_treemap(df_main_notnull)
    if treemap is not None:
        st.plotly_chart(treemap, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

    numeric_cols = [c for c in core_cols if c in df_main.columns and pd.api.types.is_numeric_dtype(df_main[c])]
    ### CONFERENCE POWER RANKINGS ###    
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
            conf_stats = df_main.groupby("CONFERENCE").agg(
                {
                    "KP_AdjEM": ["count", "max", "mean", "min"],
                    "SEED_25": ["count", "mean"],
                    "NET_25": "mean",
                    "BPI_25": "mean", #"BPI_Rank": "mean",
                    #"TR_OEff_25": "mean", #"TR_DEff_25": "mean",
                    "WIN% ALL GM": "mean", #"WIN% CLOSE GM": "mean",
                    "AVG MARGIN": "mean",
                    "eFG%": "mean", #"TS%": "mean",
                    "AST/TO%": "mean", #"NET AST/TOV RATIO": "mean",
                    "STOCKS/GM": "mean", "STOCKS-TOV/GM": "mean",
                }
            ).reset_index()

            # Flatten multi-level column index
            conf_stats.columns = [
                "CONFERENCE",
                "# TEAMS", "MAX AdjEM", "MEAN AdjEM", "MIN AdjEM",
                "# BIDS", "MEAN SEED_25", "MEAN NET_25", "MEAN BPI_25",
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
                    - **MEAN/MAX/MIN AdjEM**: Average/Range of KenPom Adjusted Efficiency Margin (higher is better)
                    - **MEAN SEED_25**: Average tournament seed (lower is better)
                    - **MEAN NET_25**: Average NCAA NET ranking (lower is better)
                    - **MEAN BPI_25**: Average ESPN BPI rating (lower is better)
                    - **MEAN AST/TO%**: Average assist-to-turnover ratio (lower is better)
                    - **MEAN WIN %**: Average win percentage (higher is better)
                    - **MEAN AVG MARGIN**: Average average scoring margin (higher is better)
                    - **MEAN eFG%**: Average effective field goal percentage (higher is better)
                    - **MEAN STOCKS/GM**: Average stocks per game (higher is better)
                    - **MEAN STOCKS-TOV/GM**: Average stocks-to-turnover per game (higher is better)
                    """)

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
                "MEAN BPI_25": "{:.1f}",

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
                 "MEAN BPI_25",
                #"AVG TR_OEff_25",
                "MEAN WIN %", "MEAN AVG MARGIN",
                "MEAN eFG%",
                "MEAN AST/TO%",
                "MEAN STOCKS/GM", "MEAN STOCKS-TOV/GM",
                
            ])
            .background_gradient(cmap="RdYlGn_r", subset=["MEAN SEED_25", "MEAN NET_25",
                                                          #"AVG TR_DEff_25",
                                                          ])
            #.set_table_styles(detailed_table_styles)
            .set_table_styles(advanced_table_styles + [index_style, cell_style, header])
        )

        st.markdown(styled_conf_stats.to_html(escape=False), unsafe_allow_html=True)

    # if "CONFERENCE" in df_main.columns:
    #     conf_metric = st.selectbox(
    #         "Select Metric for Conference Comparison", numeric_cols,
    #         index=numeric_cols.index("KP_AdjEM") if "KP_AdjEM" in numeric_cols else 0
    #     )
    #     conf_group = df_main.groupby("CONFERENCE")[conf_metric].mean().dropna().sort_values(ascending=False)
    #     fig_conf = px.bar(
    #         conf_group, y=conf_group.index, x=conf_group.values, orientation='h',
    #         title=f"Average {conf_metric} by Conference",
    #         labels={"y": "Conference", "x": conf_metric},
    #         color=conf_group.values, color_continuous_scale="Viridis",
    #         template="plotly_dark"
    #     )
    #     for conf_val in conf_group.index:
    #         teams = df_main[df_main["CONFERENCE"] == conf_val]
    #         fig_conf.add_trace(go.Scatter(
    #             x=teams[conf_metric], y=[conf_val] * len(teams),
    #             mode="markers", marker=dict(color="white", size=6, opacity=0.7),
    #             name=f"{conf_val} Teams"
    #         ))
    #     fig_conf.update_layout(showlegend=False)
    #     st.plotly_chart(fig_conf, use_container_width=True)
    # else:
    #     st.error("Conference data is not available.")
    # with st.expander("*About Conference Comparisons:*"):
    #     st.markdown("""
    #     **Conference Comparison Glossary:**
    #     - **Avg AdjEM**: Average Adjusted Efficiency Margin.
    #     - **Conference**: Grouping of teams by their athletic conferences.
    #     - Metrics intended to afford insight into relative conference strength.
    #     """)

# --- Team Metrics Comparison Tab ---
with tab_team:
    st.header(":primary[TEAM METRICS COMPARISON]")
    st.caption(":green[_DATA AS OF: 4/7/2025_]")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if "TM_KP" in df_main.columns:
        all_teams = sorted(df_main["TM_KP"].dropna().unique().tolist())
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_teams = st.multiselect(
                "👉 SELECT TEAMS TO COMPARE:",
                options=all_teams,
                default=['Duke', 'Houston', 'Florida', 'Auburn', 
                          # 'Arizona', 'Maryland', 'Texas Tech', 'Alabama', #'Arkansas', 'BYU',
                          # 'Michigan',  'Purdue', 'Tennessee', 'Kentucky',
                          # 'Michigan St.', 'Mississippi', 'Kansas',  'Iowa St.',
                         ])
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
                    "SEED_25",
                    "WIN% ALL GM", #"WIN% CLOSE GM",
                    "AVG MARGIN",
                    "KP_Rank", "BPI_Rk_25", "NET_25", 
                    #"WIN_25", "LOSS_25",
                    "KP_AdjEM", "BPI_25", #"KP_SOS_AdjEM",
                    "KP_AdjO", "KP_AdjD", 
                    "OFF EFF", "DEF EFF", 
                    #"TS%", "OPP TS%",
                    "eFG%", "OPP eFG%",
                    "3PT%", "3PTA/GM", #"3PTM/GM",
                    "AST/TO%", "STOCKS/GM", 
                ]
                metrics_to_display = [m for m in metrics_to_display if m in selected_df.columns]
                display_df = selected_df[metrics_to_display].copy()
                ncaa_avg = df_main[metrics_to_display].mean().to_frame().T
                ncaa_avg.index = ["NCAA AVERAGE"]
                display_df = pd.concat([display_df, ncaa_avg])
                format_dict = {
                    "KP_Rank": "{:.0f}",
                    #"WIN_25": "{:.0f}", 
                    #"LOSS_25": "{:.0f}",
                    "WIN% ALL GM": "{:.1%}",
                    #"WIN% CLOSE GM": "{:.1%}",
                    "KP_AdjEM": "{:.1f}",
                    "KP_AdjO": "{:.1f}",
                    "KP_AdjD": "{:.1f}",
                    "BPI_25": "{:.1f}",
                    "NET_25": "{:.0f}",
                    "SEED_25": "{:.0f}",
                    #"KP_SOS_AdjEM": "{:.1f}",
                    "OFF EFF": "{:.2f}", 
                    "DEF EFF": "{:.2f}",
                    #"TS%": "{:.1f}%", 
                    #"OPP TS%": "{:.1f}%",
                    "eFG%": "{:.1f}%", 
                    "OPP eFG%": "{:.1f}%",
                    "AST/TO%": "{:.2f}",
                    "STOCKS/GM": "{:.1f}",
                    "STOCKS-TOV/GM": "{:.2f}",
                    "AVG MARGIN": "{:.1f}",
                }
                color_scales = {
                    "KP_Rank": "RdYlGn_r",
                    "WIN_25": "RdYlGn", 
                    "LOSS_25": "RdYlGn_r",
                    "AVG MARGIN": "RdYlGn",
                    "WIN% ALL GM": "RdYlGn",
                    "WIN% CLOSE GM": "RdYlGn",
                    "KP_AdjEM": "RdYlGn",
                    "KP_AdjO": "RdYlGn",
                    "KP_AdjD": "RdYlGn_r",
                    "BPI_25": "RdYlGn",
                    "NET_25": "RdYlGn_r",
                    "SEED_25": "RdYlGn_r",
                    "KP_SOS_AdjEM": "RdYlGn_r",
                    "OFF EFF": "RdYlGn", 
                    "DEF EFF": "RdYlGn_r",
                    "TS%": "RdYlGn", 
                    "OPP TS%": "RdYlGn_r",
                    "eFG%": "RdYlGn", 
                    "OPP eFG%": "RdYlGn_r",
                    "AST/TO%": "RdYlGn",
                    "STOCKS/GM": "RdYlGn",
                    "STOCKS-TOV/GM": "RdYlGn",
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


# if Space_court1_banner:
#     st.image(Space_court1_banner, use_container_width=True) #width=750

if UAP_court2_banner:
    st.image(Space_court1_banner, use_container_width=True) #width=750


#if FinalFour25_logo:
    #st.image(FinalFour25_logo, width=750)

# if Banner_logo:
#     st.image(Banner_logo, width=750)

# if freshest_banner:
#     st.image(freshest_banner, width=750)


st.markdown("---")
st.caption(":blue[PYTHON CODE FRAMEWORK: [GitHub](https://github.com/nehat312/march-madness-2025)]")
st.caption(":blue[DATA SOURCES: [KenPom](https://kenpom.com/), [ESPN](https://espn.com/), [TeamRankings](https://www.teamrankings.com/ncaa-basketball/ranking/predictive-by-other/)]")
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