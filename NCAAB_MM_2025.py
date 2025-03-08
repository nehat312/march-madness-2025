#######################################################
## NCAAB MARCH MADNESS 2025 -- STREAMLIT MAIN SCRIPT ##
#######################################################

## LIBRARY IMPORTS ##
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

import plotly as ply
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from PIL import Image
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

## DIRECTORY CONFIGURATION ##
# Point to your GitHub RAW CSV path; do NOT force 'TEAM' as index_col since your CSV
# does not contain a 'TEAM' column name. Instead we read normally and let the
# row labels become the DataFrame’s index via index_col=0.
abs_path = r'https://raw.githubusercontent.com/nehat312/march-madness-2025/main'
mm_database_csv = abs_path + '/data/mm_2025_database.csv'

## DATA IMPORT ##
# Read the CSV, use the first column (which should be your school names) as the index:
mm_database_2025 = pd.read_csv(mm_database_csv, index_col=0)

# Make sure the index name is “TEAM” (helpful for labeling):
mm_database_2025.index.name = 'TEAM'

## PRE-PROCESSING ##
# Replace your 2023 column references with the 2025 columns that actually exist
# in the CSV. For instance:
#   - 'KP_Rank' instead of 'KenPom RANK'
#   - 'NET_24' instead of 'NET RANK'
#   - 'WIN_25' instead of 'WIN'
#   - 'LOSS_25' instead of 'LOSS'
#   - 'KP_AdjEM' instead of 'KenPom EM'
#   - 'KP_SOS_AdjEM' instead of 'KenPom SOS EM'
#   etc.
tourney_matchup_cols = [
    'KP_Rank',         # was "KenPom RANK" 
    'NET_24',          # was "NET RANK" 
    'WIN_25',          # was "WIN"
    'LOSS_25',         # was "LOSS"
    'WIN% ALL GM',
    'WIN% CLOSE GM',
    'KP_AdjEM',        # was "KenPom EM"
    'KP_SOS_AdjEM',    # was "KenPom SOS EM"
    'OFF EFF',
    'DEF EFF',
    'AVG MARGIN',
    'PTS/GM',
    'OPP PTS/GM',
    'eFG%',
    'OPP eFG%',
    'TS%',
    'OPP TS%',
    'AST/TO%',
    'STOCKS/GM',
    'STOCKS-TOV/GM',
]

# Some teams won’t have all columns completed yet (NaNs). Just filter to columns that exist:
available_cols = [c for c in tourney_matchup_cols if c in mm_database_2025.columns]
mm_database_2025 = mm_database_2025[available_cols].copy()

# Example filter: top 100 by KenPom rank
if 'KP_Rank' in mm_database_2025.columns:
    top_KP100 = mm_database_2025[ mm_database_2025['KP_Rank'] <= 100 ].copy()
else:
    # Fallback if KP_Rank not found
    top_KP100 = mm_database_2025.copy()

##########################################
## VISUALIZATION: TREEMAP EXAMPLE (2025) ##
##########################################
# We use 'TM_KP' for hover labels (since your CSV has 'TM_KP'),
# and 'CONFERENCE' for the group hierarchy. 
# For the treemap color & values, we use 'KP_AdjEM' (KenPom EM) if present.
has_adj_em = 'KP_AdjEM' in top_KP100.columns
treemap_value_col = 'KP_AdjEM' if has_adj_em else available_cols[0]  # fallback

# For coloring, if we do not have 'KP_AdjEM', pick something valid from the data
treemap_color_col = 'KP_AdjEM' if has_adj_em else available_cols[0]

# The path uses 'CONFERENCE' and 'TM_KP' if they exist, else fallback
path_list = []
if 'CONFERENCE' in top_KP100.columns:
    path_list.append('CONFERENCE')
# 'TM_KP' is from your code snippet, so if it exists:
if 'TM_KP' in mm_database_2025.columns:
    path_list.append('TM_KP')
elif 'TEAM' in top_KP100.index.name:
    # If no 'TM_KP', but we want to group by index:
    # we can do something like path=['TEAM'] if we rename index to a column
    pass

# Build the treemap only if we actually have something to group by:
if len(path_list) >= 1:
    treemap = px.treemap(
        data_frame=top_KP100.reset_index(),  # reset_index so 'TEAM' is a column
        path=path_list,
        values=treemap_value_col,
        color=treemap_color_col,
        color_discrete_sequence=px.colors.diverging.RdYlGn,
        color_continuous_scale=px.colors.diverging.RdYlGn,
        hover_name='TEAM',  # or 'TM_KP' if you prefer
        template='plotly_dark',
    )

    # Layout & styling
    viz_margin_dict = dict(l=20, r=20, t=50, b=20)
    viz_bg_color = '#0360CE'
    viz_font_dict = dict(size=12, color='#FFFFFF')

    treemap.update_layout(
        margin=viz_margin_dict,
        paper_bgcolor=viz_bg_color,
        font=viz_font_dict,
        title='2025 MARCH MADNESS LANDSCAPE',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

########################
## STREAMLIT APP SETUP ##
########################
st.set_page_config(page_title='NCAA BASKETBALL -- MARCH MADNESS 2025', 
                   layout='wide',
                   initial_sidebar_state='auto')

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

###################################
## LOGO IMAGES (IF LOCALLY SAVED) ##
###################################
# If these images are local, ensure they exist in the 'images' folder
# Or use URLs. Example:
NCAA_logo = Image.open('images/NCAA_logo1.png')

##################################
## PAGE HEADER & BASIC DISPLAYS ##
##################################
st.title('NCAA BASKETBALL -- MARCH MADNESS 2025')
st.write('*2025 MARCH MADNESS RESEARCH HUB*')
st.image(NCAA_logo, caption='NCAAB', width=200)

###############################
## SHOW THE TREEMAP (IF BUILT) ##
###############################
if len(path_list) < 1:
    st.warning("Not enough columns in CSV to build the Treemap. Check 'TM_KP' and 'CONFERENCE'.")
else:
    st.plotly_chart(treemap, use_container_width=True)

###########################################
## EXAMPLE: CREATING FAKE "REGIONAL" DATA ##
###########################################
# Adjust team names or region groupings once brackets / seeds are announced

heatmap_2025 = mm_database_2025.copy()
heatmap_2025.loc['TOURNEY AVG'] = heatmap_2025.mean(numeric_only=True)

heatmap_2025_T = heatmap_2025.T

# Example: East region teams. You must supply real 2025 region sets or remove these for now.
east_teams_2025 = [
    "Alabama","Auburn","UAB",  # <--- placeholder
    "TOURNEY AVG",
]
# Filter to those columns if they exist
East_region_2025 = heatmap_2025_T.loc[:, [t for t in east_teams_2025 if t in heatmap_2025_T.columns]]

# Same for West/South/Midwest
# (You can remove these if you do not yet have real bracket assignments.)
west_teams_2025 = [
    "Gonzaga","UCLA","Southern Cal",
    "TOURNEY AVG",
]
West_region_2025 = heatmap_2025_T.loc[:, [t for t in west_teams_2025 if t in heatmap_2025_T.columns]]

# Example aggregator columns (region means)
if not East_region_2025.empty:
    East_region_2025['EAST AVG'] = East_region_2025.mean(axis=1, numeric_only=True)
if not West_region_2025.empty:
    West_region_2025['WEST AVG'] = West_region_2025.mean(axis=1, numeric_only=True)

#########################
## PANDAS STYLER DEMOS  ##
#########################
# Example usage. You can expand it to match your original 2023 styling.
styler_dict = {
    "KP_Rank": "Spectral_r", 
    "NET_24": "Spectral_r",
    "WIN_25": "Spectral",
    "LOSS_25": "Spectral_r",
    "WIN% ALL GM": "Spectral",
    "WIN% CLOSE GM": "Spectral",
    "KP_AdjEM": "Spectral",
    "KP_SOS_AdjEM": "Spectral",
    "OFF EFF": "Spectral",
    "DEF EFF": "Spectral_r",
    "AVG MARGIN": "Spectral",
    "PTS/GM": "Spectral",
    "OPP PTS/GM": "Spectral_r",
    "eFG%": "Spectral",
    "OPP eFG%": "Spectral_r",
    "TS%": "Spectral",
    "OPP TS%": "Spectral_r",
    "AST/TO%": "Spectral",
    "STOCKS/GM": "Spectral",
    "STOCKS-TOV/GM": "Spectral",
}

############
## TABS UI ##
############
tab_0, tab_1, tab_2, tab_3, tab_4 = st.tabs(['2025','TBU','TBU','TBU','TBU'])

with tab_0:
    st.subheader("EAST REGION (DEMO)")
    if not East_region_2025.empty:
        # Build a Styler
        East_region_styler = East_region_2025.style
        for col, cmap in styler_dict.items():
            if col in East_region_2025.index:
                East_region_styler = East_region_styler.background_gradient(
                    cmap=cmap,
                    subset=pd.IndexSlice[col, :],
                    axis=1
                )
        st.markdown(East_region_styler.to_html(), unsafe_allow_html=True)
    else:
        st.write("No EAST region data yet.")

    st.subheader("WEST REGION (DEMO)")
    if not West_region_2025.empty:
        West_region_styler = West_region_2025.style
        for col, cmap in styler_dict.items():
            if col in West_region_2025.index:
                West_region_styler = West_region_styler.background_gradient(
                    cmap=cmap,
                    subset=pd.IndexSlice[col, :],
                    axis=1
                )
        st.markdown(West_region_styler.to_html(), unsafe_allow_html=True)
    else:
        st.write("No WEST region data yet.")

with tab_1:
    st.subheader("Placeholder for future bracket or summary data")
    st.image(NCAA_logo, width=180)

with tab_2:
    st.subheader("Placeholder for future bracket or summary data")
    st.image(NCAA_logo, width=180)

with tab_3:
    st.subheader("Placeholder for future bracket or summary data")
    st.image(NCAA_logo, width=180)

with tab_4:
    st.subheader("Placeholder for future bracket or summary data")
    st.image(NCAA_logo, width=180)

#########################
## LINK OUT / FOOTER   ##
#########################
github_link1 = '[GITHUB: 2025 Repo](https://github.com/nehat312/march-madness-2025)'
kenpom_site_link = '[KENPOM](https://kenpom.com/)'
st.write(github_link1 + " | " + kenpom_site_link)

## SCRIPT TERMINATION ##
st.stop()
