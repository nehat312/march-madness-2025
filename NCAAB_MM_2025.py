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
import os

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

## DIRECTORY CONFIGURATION ##
abs_path = r'https://raw.githubusercontent.com/nehat312/march-madness-2025/main'
mm_database_csv = abs_path + '/data/mm_2025_database.csv'

################################
## READ THE 2025 CSV DATAFRAME ##
################################
# The CSV’s first column should be Team Names, so index_col=0 ensures that column is used as the index.
mm_database_2025 = pd.read_csv(mm_database_csv, index_col=0)
mm_database_2025.index.name = 'TEAM'  # rename the DF index to "TEAM" for clarity

##################################################
## SAMPLE PRE-PROCESSING & COLUMN SELECTION/RENAMES
##################################################
# Adjust column references to match the actual 2025 CSV columns
# Many columns from the snippet below reference names we know exist in 2025 data

tourney_matchup_cols = [
    'KP_Rank',         # KenPom ranking
    'NET_24',          # 2024 NET rank, but presumably still relevant in 2025 CSV
    'WIN_25',          
    'LOSS_25',
    'WIN% ALL GM',
    'WIN% CLOSE GM',
    'KP_AdjEM',        # KenPom adjusted efficiency margin
    'KP_SOS_AdjEM',    # KenPom SOS adjusted EM
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

# Only keep columns that actually exist in the CSV
available_cols = [c for c in tourney_matchup_cols if c in mm_database_2025.columns]
df_main = mm_database_2025[available_cols].copy()

#####################
## SIMPLE FILTERING ##
#####################
# Example: top 100 teams by KenPom rank
if 'KP_Rank' in df_main.columns:
    top_KP100 = df_main[df_main['KP_Rank'] <= 100].copy()
else:
    top_KP100 = df_main.copy()

######################################
## TREEMAP (CONFERENCE vs. KP_AdjEM) ##
######################################
# If your CSV includes 'CONFERENCE' and 'TM_KP' or 'TM_MODEL', we can build a treemap.
# Adjust to whichever naming your CSV includes for team name or model name columns.
# Let’s check if they exist:
path_list = []
if 'CONFERENCE' in mm_database_2025.columns:
    path_list.append('CONFERENCE')
if 'TM_KP' in mm_database_2025.columns:
    path_list.append('TM_KP')

# For values & color, we’ll use 'KP_AdjEM' if present:
treemap_value_col = 'KP_AdjEM' if 'KP_AdjEM' in top_KP100.columns else None

if len(path_list) > 0 and treemap_value_col is not None:
    # We must .reset_index() so that the 'TEAM' index becomes a column for Plotly
    treemap_data = top_KP100.reset_index()

    treemap = px.treemap(
        data_frame=treemap_data,
        path=path_list,
        values=treemap_value_col,
        color=treemap_value_col,
        color_continuous_scale=px.colors.diverging.RdYlGn,
        color_discrete_sequence=px.colors.diverging.RdYlGn,
        hover_name='TEAM',   # or 'TM_KP' if you want
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
        title='2025 MARCH MADNESS LANDSCAPE: (KenPom AdjEM by Conference)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
else:
    treemap = None  # fallback if columns not present

##################################
## LOGO IMAGE (CHECK FILE FIRST) ##
##################################
logo_path = 'images/NCAA_logo1.png'
NCAA_logo = None
if os.path.exists(logo_path):
    NCAA_logo = Image.open(logo_path)
else:
    # Optionally, load from a URL or skip
    # For example:
    # NCAA_logo = Image.open(requests.get('https://upload.wikimedia.org/...', stream=True).raw)
    pass

##################################
## STREAMLIT APP CONFIG & LAYOUT ##
##################################
st.set_page_config(
    page_title='NCAA BASKETBALL -- MARCH MADNESS 2025',
    layout='wide',
    initial_sidebar_state='auto'
)

hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title('NCAA BASKETBALL -- MARCH MADNESS 2025')
st.write('*2025 MARCH MADNESS RESEARCH HUB*')
if NCAA_logo:
    st.image(NCAA_logo, caption='NCAAB', width=200)

#####################
## DEFINE TABS LAYOUT
#####################
tab_0, tab_1, tab_2, tab_3, tab_4 = st.tabs(['Home', 'EDA & Plots', 'Regional Heatmaps', 'TBU', 'TBU'])

####################
## TAB 0: HOME PAGE
####################
with tab_0:
    st.subheader('Overall Landscape')
    if treemap is not None:
        st.plotly_chart(treemap, use_container_width=True)
    else:
        st.warning("Treemap not available (missing 'CONFERENCE' or 'KP_AdjEM').")

    st.write("Use the tabs above to explore additional EDA, region-based tables, and other 2025 data.")

########################
## TAB 1: EDA & PLOTS
########################
with tab_1:
    st.header("Exploratory Data Analysis")
    st.markdown("""
    Below are some quick exploratory plots of the 2025 data. 
    Adjust columns, bins, or filters as needed.
    """)

    ## Example 1: Distribution (Histogram) of KP_AdjEM
    if 'KP_AdjEM' in df_main.columns:
        st.subheader("Histogram of KenPom AdjEM (all teams)")
        fig_hist = px.histogram(
            df_main, 
            x='KP_AdjEM', 
            nbins=20, 
            title="Distribution of KenPom AdjEM"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.write("`KP_AdjEM` not found in columns for histogram plot.")

    ## Example 2: Correlation Heatmap
    # Choose a subset of numeric columns
    numeric_cols = [
        'KP_Rank','NET_24','WIN_25','LOSS_25','KP_AdjEM',
        'KP_SOS_AdjEM','OFF EFF','DEF EFF','AVG MARGIN','PTS/GM','OPP PTS/GM'
    ]
    numeric_cols = [c for c in numeric_cols if c in df_main.columns]

    if len(numeric_cols) >= 2:
        st.subheader("Correlation Heatmap")
        temp_df = df_main[numeric_cols].dropna()
        corr_matrix = temp_df.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Correlation Matrix (selected columns)'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns to produce a correlation heatmap.")

    ## Example 3: Conference-level Average of KP_AdjEM
    if 'CONFERENCE' in mm_database_2025.columns and 'KP_AdjEM' in df_main.columns:
        st.subheader("Avg KenPom AdjEM by Conference (Bar Chart)")
        conf_group = mm_database_2025.groupby('CONFERENCE')['KP_AdjEM'].mean(numeric_only=True).dropna()
        conf_group = conf_group.sort_values(ascending=False)

        fig_conf = px.bar(
            conf_group,
            x=conf_group.index,
            y=conf_group.values,
            title="Average KenPom AdjEM by Conference",
            labels={'x': 'Conference', 'y': 'KP_AdjEM'},
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.write("No 'CONFERENCE' or 'KP_AdjEM' column found for conference bar chart.")


############################################
## TAB 2: REGIONAL HEATMAPS (SEED PLACEHOLDERS)
############################################
with tab_2:
    st.header("Regional Heatmaps / Bracket Data (2025)")

    # The bracket is not set yet, but we can demonstrate placeholders using 
    # the 2023 code approach if you have team lists. For example:
    # Example region teams (fake placeholders) - update once seeds are known
    east_teams_2025 = [
        "Alabama", "Auburn", "Tennessee", "TOURNEY AVG"
    ]
    # Build transposed DF with average row
    df_heat = df_main.copy()
    df_heat.loc['TOURNEY AVG'] = df_heat.mean(numeric_only=True)
    df_heat_T = df_heat.T

    East_region_2025 = df_heat_T.loc[:, [t for t in east_teams_2025 if t in df_heat_T.columns]]

    # Simple styling logic for demonstration
    styler_dict = {
        "KP_Rank": "Spectral_r",
        "NET_24": "Spectral_r",
        "WIN_25": "Spectral",
        "LOSS_25": "Spectral_r",
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

    st.subheader('EAST REGION (DEMO PLACEHOLDER)')
    if East_region_2025.empty:
        st.write("No example East region teams found in the dataset yet.")
    else:
        # Build a Styler
        East_region_styler = East_region_2025.style
        for row_label, cmap in styler_dict.items():
            if row_label in East_region_2025.index:
                East_region_styler = East_region_styler.background_gradient(
                    cmap=cmap,
                    subset=pd.IndexSlice[row_label, :],
                    axis=1
                )
        # Convert to HTML and display
        st.markdown(East_region_styler.to_html(), unsafe_allow_html=True)

    st.write("Add your real regions once seeds are determined, or load them from a bracket file.")


#############################
## TAB 3 & TAB 4: PLACEHOLDERS
#############################
with tab_3:
    st.subheader('TBU (Future Content)')
    if NCAA_logo:
        st.image(NCAA_logo, width=200)
    st.write("Use this tab for future bracket analysis, matchup predictions, etc.")

with tab_4:
    st.subheader('TBU (Future Content)')
    if NCAA_logo:
        st.image(NCAA_logo, width=200)
    st.write("Use this tab for additional EDA, historical comparisons, or advanced stats.")


################################
## FOOTER LINKS & SCRIPT ENDING
################################
st.write("---")
github_link1 = '[GITHUB: 2025 Repo](https://github.com/nehat312/march-madness-2025)'
kenpom_site_link = '[KENPOM](https://kenpom.com/)'
st.write(f"{github_link1} | {kenpom_site_link}")

st.stop()
