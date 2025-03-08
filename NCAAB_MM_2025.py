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

# from streamlit_aggrid import AgGrid

# import matplotlib.pyplot as plt
# import seaborn as sns
# import dash as dash
# from dash import dash_table
# from dash import dcc
# from dash import html
# from dash.dependencies import Input, Output
# from dash.exceptions import PreventUpdate
# import dash_bootstrap_components as dbc

# import scipy.stats as stats
# import statistics

## DIRECTORY CONFIGURATION ##
abs_path = r'https://raw.githubusercontent.com/nehat312/march-madness-2025/main'
#mm_database_xlsx = abs_path + '/data/mm_2023_database.xlsx'
mm_database_csv = abs_path + '/data/mm_2025_database.csv'

## DATA IMPORT ##
# Only using the 2025 CSV database for TR_df (the same file is read below)
# mm_database_2023 = pd.read_csv(mm_database_csv, index_col='TEAM')
mm_database_2025 = pd.read_csv(mm_database_csv, index_col='TEAM')

## PRE-PROCESSING ##
# For the pre-tournament overall NCAA men's basketball field, we work with the 2025 dataset.
# Note: the "SEED" column is retained in tourney_matchup_cols even though selection brackets aren’t available yet.
tourney_matchup_cols = ['KenPom RANK', 'NET RANK',
            'WIN', 'LOSS',
            'WIN% ALL GM', 'WIN% CLOSE GM',
            'KenPom EM', #'AdjO', 'AdjD', #'KP_Rank', #'AdjT', 'Luck',
            'KenPom SOS EM', #'SOS OppO', 'SOS OppD', 'NCSOS Adj EM'
            'OFF EFF', 'DEF EFF',
            'AVG MARGIN', #'OPP AVG MARGIN':"Spectral_r",
            'PTS/GM', 'OPP PTS/GM',
            'eFG%', 'OPP eFG%', 'TS%', 'OPP TS%',
            'AST/TO%', #'NET AST/TOV RATIO',
            'STOCKS/GM', 'STOCKS-TOV/GM',
            ]

# Subset the 2025 database to the tourney matchup columns
mm_database_2025 = mm_database_2025[tourney_matchup_cols]

# Filter top 100 teams by KP_Rank (make sure your CSV has this column)
top_KP100 = mm_database_2025[mm_database_2025['KenPom RANK'] <= 100]

## VISUALIZATION: TREEMAP ##
# Create a treemap by CONFERENCE and Team Key Performance (TM_KP is assumed to be in your dataset)
# (Adjust the column names if needed; here we use 'CONFERENCE' and 'TM_KP' as provided.)
treemap = px.treemap(
    data_frame=top_KP100,  # using top_KP100 from the 2025 csv
    path=['CONFERENCE', 'TM_KP'],
    values='KenPom EM',  # using KenPom EM as the value (adjust if needed)
    color='KenPom EM',
    color_discrete_sequence=px.colors.diverging.RdYlGn,
    color_continuous_scale=px.colors.diverging.RdYlGn,
    hover_name='TM_KP',
    template='plotly_dark',
)

# Update layout and display the treemap (use st.plotly_chart for Streamlit)
viz_margin_dict = dict(l=20, r=20, t=50, b=20)
viz_bg_color = '#0360CE'
viz_font_dict = dict(size=12, color='#FFFFFF')

treemap.update_layout(
    margin=viz_margin_dict,
    paper_bgcolor=viz_bg_color,
    font=viz_font_dict,
    title='2025 MARCH MADNESS LANDSCAPE (KenPom AdjEM by CONFERENCE)',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
)
st.plotly_chart(treemap, use_container_width=True)

## IMAGE IMPORT ##
NCAA_logo = Image.open('images/NCAA_logo1.png')
# Uncomment and add additional logos when available
# FF_logo = Image.open('images/FinalFour_2023.png')
# FAU_logo = Image.open('images/FAU_Owls.png')
# Miami_logo = Image.open('images/Miami_Canes.png')
# UConn_logo = Image.open('images/UConn_Huskies.png')
# SDSU_logo = Image.open('images/SDSU_Aztecs.png')

## FORMAT / STYLE ##
# Chart background style definitions
viz_margin_dict = dict(l=20, r=20, t=50, b=20)
viz_bg_color = '#0360CE'
viz_font_dict = dict(size=12, color='#FFFFFF')

## COLOR SCALES ##
Tropic = px.colors.diverging.Tropic
Blackbody = px.colors.sequential.Blackbody
BlueRed = px.colors.sequential.Bluered

Sunsetdark = px.colors.sequential.Sunsetdark
Sunset = px.colors.sequential.Sunset
Temps = px.colors.diverging.Temps
Tealrose = px.colors.diverging.Tealrose

Ice = px.colors.sequential.ice
Ice_r = px.colors.sequential.ice_r
Dense = px.colors.sequential.dense
Deep = px.colors.sequential.deep
PuOr = px.colors.diverging.PuOr
Speed = px.colors.sequential.speed

## VISUALIZATION LABELS ##
chart_labels = {
    'Team': 'TEAM', 'TEAM_KP': 'TEAM', 'TEAM_TR': 'TEAM',
    'Conference': 'CONFERENCE', 'Seed': 'SEED',
    'Win': 'WIN', 'Loss': 'LOSS',
    'win-pct-all-games': 'WIN%', 'win-pct-close-games': 'WIN%_CLOSE',
    'effective-possession-ratio': 'POSS%',
    'three-point-pct': '3P%', 'two-point-pct': '2P%', 'free-throw-pct': 'FT%',
    'field-goals-made-per-game': 'FGM/GM', 'field-goals-attempted-per-game': 'FGA/GM',
    'three-pointers-made-per-game': '3PM/GM', 'three-pointers-attempted-per-game': '3PA/GM',
    'offensive-efficiency': 'O_EFF', 'defensive-efficiency': 'D_EFF',
    'total-rebounds-per-game': 'TRB/GM', 'offensive-rebounds-per-game': 'ORB/GM',
    'defensive-rebounds-per-game': 'DRB/GM',
    'offensive-rebounding-pct': 'ORB%', 'defensive-rebounding-pct': 'DRB%',
    'total-rebounding-percentage': 'TRB%',
    'blocks-per-game': 'B/GM', 'steals-per-game': 'S/GM', 'assists-per-game': 'AST/GM',
    'turnovers-per-game': 'TO/GM',
    'possessions-per-game': 'POSS/GM',
    'personal-fouls-per-game': 'PF/GM',
    'opponent-three-point-pct': 'OPP_3P%', 'opponent-two-point-pct': 'OPP_2P%',
    'opponent-free-throw-pct': 'OPP_FT%', 'opponent-shooting-pct': 'OPP_FG%',
    'opponent-assists-per-game': 'OPP_AST/GM', 'opponent-turnovers-per-game': 'OPP_TO/GM',
    'opponent-assist--per--turnover-ratio': 'OPP_AST/TO',
    'opponent-offensive-rebounds-per-game': 'OPP_OREB/GM',
    'opponent-defensive-rebounds-per-game': 'OPP_DREB/GM',
    'opponent-total-rebounds-per-game': 'OPP_TREB/GM',
    'opponent-offensive-rebounding-pct': 'OPP_OREB%', 'opponent-defensive-rebounding-pct': 'OPP_DREB%',
    'opponent-blocks-per-game': 'OPP_BLK/GM', 'opponent-steals-per-game': 'OPP_STL/GM',
    'opponent-effective-possession-ratio': 'OPP_POSS%',
    'net-avg-scoring-margin': 'NET_AVG_MARGIN', 'net-points-per-game': 'NET_PTS/GM',
    'net-adj-efficiency': 'NET_ADJ_EFF',
    'net-effective-field-goal-pct': 'NET_EFG%', 'net-true-shooting-percentage': 'NET_TS%',
    'stocks-per-game': 'STOCKS/GM', 'total-turnovers-per-game': 'TTL_TO/GM',
    'net-assist--per--turnover-ratio': 'NET_AST/TO',
    'net-total-rebounds-per-game': 'NET_TREB/GM', 'net-off-rebound-pct': 'NET_OREB%',
    'net-def-rebound-pct': 'NET_DREB%',
    'Adj EM': 'KenPom ADJ EM', 'SOS Adj EM': 'KenPom SOS ADJ EM',
    'points-per-game': 'PTS/GM', 'opponent-points-per-game': 'OPP PTS/GM',
    'average-scoring-margin': 'AVG SCORE MARGIN', 'opponent-average-scoring-margin': 'OPP AVG SCORE MARGIN',
    'effective-field-goal-pct': 'eFG%', 'true-shooting-percentage': 'TS%',
    'opponent-effective-field-goal-pct': 'OPP eFG%', 'opponent-true-shooting-percentage': 'OPP TS%',
    'assist--per--turnover-ratio': 'AST_TOV_RATIO', 'NET AST/TOV RATIO': 'NET AST/TOV%',
    'STOCKS-per-game': 'STOCKS/GM', 'STOCKS-TOV-per-game': 'STOCKS-TOV/GM',
    'MMS STOCKS-TOV-per-game': 'STOCKS-TOV/GM', 'MMS Adj EM': 'KenPom ADJ EM',
}

## PANDAS STYLER CONFIGURATION ##
# (The following style dicts are preserved from your original code.)
header = {'selector': 'th',
          'props': [('background-color', '#0360CE'), ('color', 'white'),
                    ('text-align', 'center'), ('vertical-align', 'center'),
                    ('font-weight', 'bold'),
                    ('border-bottom', '2px solid #000000'),
                    ]}

header_level0 = {'selector': 'th.col_heading.level0',
                 'props': [('font-size', '12px'),]}

index = {'selector': 'th.row_heading',
         'props': [('background-color', '#000000'), ('color', 'white'),
                   ('text-align', 'center'), ('vertical-align', 'center'),
                   ('font-weight', 'bold'), ('font-size', '12px'),
                   ]}

numbers = {'selector': 'td.data',
           'props': [('text-align', 'center'), ('vertical-align', 'center'),
                     ('font-weight', 'bold')]}
                   
borders_right = {'selector': '.row_heading.level1',
                 'props': [('border-right', '1px solid #FFFFFF')]}

top_row = {'selector': 'td.data.row0',
           'props': [('border-bottom', '2px dashed #000000'),
                     ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')], }

# (The remainder of the row and column style definitions are preserved below.)
table_row1 = {'selector': '.row1',
              'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row2 = {'selector': '.row2',
              'props': [('border-bottom', '2px dashed #000000'),
                        ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row3 = {'selector': '.row3',
              'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row4 = {'selector': '.row4',
              'props': [('border-bottom', '2px dashed #000000'),
                        ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row5 = {'selector': '.row5',
              'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row6 = {'selector': '.row6',
              'props': [('border-bottom', '2px dashed #000000'),
                        ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row7 = {'selector': '.row7',
              'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row8 = {'selector': '.row8',
              'props': [('border-bottom', '2px dashed #000000'),
                        ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row9 = {'selector': '.row9',
              'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row10 = {'selector': '.row10',
               'props': [('border-bottom', '2px dashed #000000'),
                         ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row11 = {'selector': '.row11',
               'props': [('border-bottom', '2px dashed #000000'),
                         ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row12 = {'selector': '.row12',
               'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row13 = {'selector': '.row13',
               'props': [('border-bottom', '2px dashed #000000'),
                         ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row14 = {'selector': '.row14',
               'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row15 = {'selector': '.row15',
               'props': [('border-bottom', '2px dashed #000000'),
                         ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row16 = {'selector': '.row16',
               'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row17 = {'selector': '.row17',
               'props': [('border-bottom', '2px dashed #000000'),
                         ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row18 = {'selector': '.row18',
               'props': [('border-bottom', '2px dashed #000000'),
                         ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row19 = {'selector': '.row19',
               'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row20 = {'selector': '.row20',
               'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}
table_row21 = {'selector': '.row21',
               'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}

table_col1 = {'selector': '.col1',
              'props': [('border-left', '2px dashed #000000')]}
table_col2 = {'selector': '.col2',
              'props': [('border-left', '2px dashed #000000')]}
table_col3 = {'selector': '.col3',
              'props': [('border-left', '2px dashed #000000')]}
table_col4 = {'selector': '.col4',
              'props': [('border-left', '2px dashed #000000')]}
table_col5 = {'selector': '.col5',
              'props': [('border-left', '2px dashed #000000')]}
table_col6 = {'selector': '.col6',
              'props': [('border-left', '2px dashed #000000')]}
table_col7 = {'selector': '.col7',
              'props': [('border-left', '2px dashed #000000')]}
table_col8 = {'selector': '.col8',
              'props': [('border-left', '2px dashed #000000')]}
table_col9 = {'selector': '.col9',
              'props': [('border-left', '2px dashed #000000')]}
table_col10 = {'selector': '.col10',
               'props': [('border-left', '2px dashed #000000')]}
table_col11 = {'selector': '.col11',
               'props': [('border-left', '2px dashed #000000')]}
table_col12 = {'selector': '.col12',
               'props': [('border-left', '2px dashed #000000')]}
table_col13 = {'selector': '.col13',
               'props': [('border-left', '2px dashed #000000')]}
table_col14 = {'selector': '.col14',
               'props': [('border-left', '2px dashed #000000')]}
table_col15 = {'selector': '.col15',
               'props': [('border-left', '2px dashed #000000')]}
table_col16 = {'selector': '.col16',
               'props': [('border-left', '3px solid #000000')]}
table_col17 = {'selector': '.col17',
               'props': [('border-left', '2px dashed #000000')]}
table_col18 = {'selector': '.col18',
               'props': [('border-left', '2px dashed #000000')]}
table_col19 = {'selector': '.col19',
               'props': [('border-left', '2px dashed #000000')]}
table_col20 = {'selector': '.col20',
               'props': [('border-left', '2px dashed #000000')]}
table_col21 = {'selector': '.col21',
               'props': [('border-left', '2px dashed #000000')]}

## HEATMAP DATAFRAMES ##
# Update variable names from 2023 to 2025
heatmap_tourney_2025 = mm_database_2025.copy()
heatmap_tourney_2025.loc['TOURNEY AVG'] = heatmap_tourney_2025.mean()

# Create the transposed version for heatmapping
heatmap_tourney_2025_T = pd.DataFrame(heatmap_tourney_2025.T)

# Define regional groups (update team names as needed for 2025)
East_region_2025 = heatmap_tourney_2025_T[['Purdue', 'Marquette', 'Kansas St', 'Tennessee',
                                           'Duke', 'Kentucky', 'Michigan St', 'Memphis',
                                           'Fla Atlantic', 'USC', 'Providence', 'Oral Roberts',
                                           'Lafayette', 'Montana St', 'Vermont', 'F Dickinson',
                                           'TOURNEY AVG',]]
                                           
West_region_2025 = heatmap_tourney_2025_T[['Kansas', 'UCLA', 'Gonzaga', 'Connecticut',
                                           'St Marys', 'TX Christian', 'Northwestern', 'Arkansas',
                                           'Illinois', 'Boise State', 'Arizona St', 'VCU',
                                           'Iona', 'Grd Canyon', 'NC-Asheville', 'Howard',
                                           'TOURNEY AVG',]]
                                           
South_region_2025 = heatmap_tourney_2025_T[['Alabama', 'Arizona', 'Baylor', 'Virginia',
                                           'San Diego St', 'Creighton', 'Missouri', 'Maryland',
                                           'W Virginia', 'Utah State', 'NC State', 'Col Charlestn',
                                           'Furman', 'UCSB', 'Princeton', 'TX A&M-CC',
                                           'TOURNEY AVG',]]
                                           
Midwest_region_2025 = heatmap_tourney_2025_T[['Houston', 'Texas', 'Xavier', 'Indiana',
                                           'Miami (FL)', 'Iowa State', 'Texas A&M', 'Iowa',
                                           'Auburn', 'Penn State', 'Pittsburgh', 'Drake',
                                           'Kent State', 'Kennesaw St', 'Colgate', 'N Kentucky',
                                           'TOURNEY AVG',]]

# Compute region averages
East_region_2025['EAST AVG'] = East_region_2025.mean(numeric_only=True, axis=1)
East_region_2025['WEST AVG'] = West_region_2025.mean(numeric_only=True, axis=1)
East_region_2025['SOUTH AVG'] = South_region_2025.mean(numeric_only=True, axis=1)
East_region_2025['MIDWEST AVG'] = Midwest_region_2025.mean(numeric_only=True, axis=1)

West_region_2025['WEST AVG'] = West_region_2025.mean(numeric_only=True, axis=1)
West_region_2025['EAST AVG'] = East_region_2025.mean(numeric_only=True, axis=1)
West_region_2025['SOUTH AVG'] = South_region_2025.mean(numeric_only=True, axis=1)
West_region_2025['MIDWEST AVG'] = Midwest_region_2025.mean(numeric_only=True, axis=1)

South_region_2025['SOUTH AVG'] = South_region_2025.mean(numeric_only=True, axis=1)
South_region_2025['EAST AVG'] = East_region_2025.mean(numeric_only=True, axis=1)
South_region_2025['WEST AVG'] = West_region_2025.mean(numeric_only=True, axis=1)
South_region_2025['MIDWEST AVG'] = Midwest_region_2025.mean(numeric_only=True, axis=1)

Midwest_region_2025['MIDWEST AVG'] = Midwest_region_2025.mean(numeric_only=True, axis=1)
Midwest_region_2025['EAST AVG'] = East_region_2025.mean(numeric_only=True, axis=1)
Midwest_region_2025['SOUTH AVG'] = South_region_2025.mean(numeric_only=True, axis=1)
Midwest_region_2025['WEST AVG'] = West_region_2025.mean(numeric_only=True, axis=1)

# Styler color gradients for key columns
styler_dict = {"KenPom RANK": "Spectral_r", 'NET RANK': "Spectral_r",
               "WIN": "Spectral", "LOSS": "Spectral_r",
               'WIN% ALL GM': "Spectral", 'WIN% CLOSE GM': "Spectral",
               'KenPom EM':"Spectral", 'KenPom SOS EM':"Spectral",
               'OFF EFF':"Spectral", 'DEF EFF':"Spectral_r",
               'AVG MARGIN':"Spectral",
               'PTS/GM':"Spectral", 'OPP PTS/GM':"Spectral_r",
               'eFG%':"Spectral", 'OPP eFG%':"Spectral_r",
               'TS%':"Spectral", 'OPP TS%':"Spectral_r",
               'AST/TO%':"Spectral",
               'STOCKS/GM':"Spectral", 'STOCKS-TOV/GM':"Spectral",
               }

East_region_styler = East_region_2025.style
West_region_styler = West_region_2025.style
South_region_styler = South_region_2025.style
Midwest_region_styler = Midwest_region_2025.style
for idx, cmap in styler_dict.items():
    East_region_styler = East_region_styler.background_gradient(cmap=cmap, subset=pd.IndexSlice[idx, :], axis=1)
    West_region_styler = West_region_styler.background_gradient(cmap=cmap, subset=pd.IndexSlice[idx, :], axis=1)
    South_region_styler = South_region_styler.background_gradient(cmap=cmap, subset=pd.IndexSlice[idx, :], axis=1)
    Midwest_region_styler = Midwest_region_styler.background_gradient(cmap=cmap, subset=pd.IndexSlice[idx, :], axis=1)

## STREAMLIT APP CONFIGURATION ##
st.set_page_config(page_title='NCAA BASKETBALL -- MARCH MADNESS 2025', layout='wide', initial_sidebar_state='auto')

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

## CSS LAYOUT CUSTOMIZATION ##
th_props = [('font-size', '12px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('color', '#EBEDE9'),
            ('background-color', '#29609C')]
td_props = [('font-size', '12px')]
df_styles = [dict(selector="th", props=th_props),
             dict(selector="td", props=td_props)]

col_format_dict = {
    # Define column format if necessary
}

## SIDEBAR (Optional) ##
# You can uncomment and set additional sidebar filters if needed
# st.sidebar.subheader('DIRECTORY:')

## HEADER ##
st.container()
st.title('NCAA BASKETBALL -- MARCH MADNESS 2025')
st.write('*2025 MARCH MADNESS RESEARCH HUB*')

## LOGOS ##
# Example: display NCAA logo (add more logos as they become available)
st.image(NCAA_logo, caption='NCAAB', width=200)

# Define tabs for different sections
tab_0, tab_1, tab_2, tab_3, tab_4 = st.tabs(['2025', 'TBU', 'TBU', 'TBU', 'TBU'])

with tab_0:
    ## REGIONAL HEATMAPS ##
    st.subheader('EAST REGION')
    st.markdown(
        East_region_styler.format('{:.2f}', na_rep='NA')
        .set_table_styles([header, header_level0, index, top_row, numbers, borders_right,
                           table_row1, table_row2, table_row3, table_row4, table_row5,
                           table_row6, table_row7, table_row8, table_row9, table_row10,
                           table_row11, table_row12, table_row13, table_row14, table_row15,
                           table_row16, table_row17, table_row18, table_row19, table_row20, table_row21,
                           table_col1, table_col2, table_col3, table_col4, table_col5, table_col6,
                           table_col7, table_col8, table_col9, table_col10, table_col11, table_col12,
                           table_col13, table_col14, table_col15, table_col16, table_col17, table_col18,
                           table_col19, table_col20, table_col21,
                           ])
        .set_properties(**{'min-width': '55px', 'max-width': '55px', 'column-width': '55px', 'width': '55px'})
        .to_html(table_uuid='east_region'),
        unsafe_allow_html=True)

    st.subheader('WEST REGION')
    st.markdown(
        West_region_styler.format('{:.2f}', na_rep='NA')
        .set_table_styles([header, header_level0, index, top_row, numbers, borders_right,
                           table_row1, table_row2, table_row3, table_row4, table_row5,
                           table_row6, table_row7, table_row8, table_row9, table_row10,
                           table_row11, table_row12, table_row13, table_row14, table_row15,
                           table_row16, table_row17, table_row18, table_row19, table_row20, table_row21,
                           table_col1, table_col2, table_col3, table_col4, table_col5, table_col6,
                           table_col7, table_col8, table_col9, table_col10, table_col11, table_col12,
                           table_col13, table_col14, table_col15, table_col16, table_col17, table_col18,
                           table_col19, table_col20, table_col21,
                           ])
        .set_properties(**{'min-width': '55px', 'max-width': '55px', 'column-width': '55px', 'width': '55px'})
        .to_html(table_uuid='west_region'),
        unsafe_allow_html=True)

    st.subheader('SOUTH REGION')
    st.markdown(
        South_region_styler.format('{:.2f}', na_rep='NA')
        .set_table_styles([header, header_level0, index, top_row, numbers, borders_right,
                           table_row1, table_row2, table_row3, table_row4, table_row5,
                           table_row6, table_row7, table_row8, table_row9, table_row10,
                           table_row11, table_row12, table_row13, table_row14, table_row15,
                           table_row16, table_row17, table_row18, table_row19, table_row20, table_row21,
                           table_col1, table_col2, table_col3, table_col4, table_col5, table_col6,
                           table_col7, table_col8, table_col9, table_col10, table_col11, table_col12,
                           table_col13, table_col14, table_col15, table_col16, table_col17, table_col18,
                           table_col19, table_col20, table_col21,
                           ])
        .set_properties(**{'min-width': '55px', 'max-width': '55px', 'column-width': '55px', 'width': '55px'})
        .to_html(table_uuid='south_region'),
        unsafe_allow_html=True)

    st.subheader('MIDWEST REGION')
    st.markdown(
        Midwest_region_styler.format('{:.2f}', na_rep='NA')
        .set_table_styles([header, header_level0, index, top_row, numbers, borders_right,
                           table_row1, table_row2, table_row3, table_row4, table_row5,
                           table_row6, table_row7, table_row8, table_row9, table_row10,
                           table_row11, table_row12, table_row13, table_row14, table_row15,
                           table_row16, table_row17, table_row18, table_row19, table_row20, table_row21,
                           table_col1, table_col2, table_col3, table_col4, table_col5, table_col6,
                           table_col7, table_col8, table_col9, table_col10, table_col11, table_col12,
                           table_col13, table_col14, table_col15, table_col16, table_col17, table_col18,
                           table_col19, table_col20, table_col21,
                           ])
        .set_properties(**{'min-width': '55px', 'max-width': '55px', 'column-width': '55px', 'width': '55px'})
        .to_html(table_uuid='midwest_region'),
        unsafe_allow_html=True)

    # Additional visualizations (e.g., scatter plots) can be added here as needed.

    ## EXTERNAL LINKS ##
    github_link1 = '[GITHUB <> NE](https://github.com/nehat312/march-madness-2025/)'
    github_link2 = '[GITHUB <> TP](https://github.com/Tyler-Pickett/march-madness-2023/)'  # Update when available
    kenpom_site_link = '[KENPOM](https://kenpom.com/)'

    link_col_1, link_col_2, link_col_3 = st.columns(3)
    link_col_1.markdown(github_link1, unsafe_allow_html=True)
    link_col_2.markdown(github_link2, unsafe_allow_html=True)
    link_col_3.markdown(kenpom_site_link, unsafe_allow_html=True)

with tab_1:
    st.subheader('EXAMPLE SUBHEADER')
    st.image(NCAA_logo, width=200)

with tab_2:
    st.subheader('EXAMPLE SUBHEADER')
    st.image(NCAA_logo, width=200)
    # Example: st.dataframe(mil_bucks_2021.style.format(col_format_dict).set_table_styles(df_styles))

with tab_3:
    st.subheader('EXAMPLE SUBHEADER')
    st.image(NCAA_logo, width=200)
    # Example: st.dataframe(lal_lakers_2020.style.format(col_format_dict).set_table_styles(df_styles))

with tab_4:
    st.subheader('EXAMPLE SUBHEADER')
    st.image(NCAA_logo, width=200)
    # Example: st.dataframe(tor_raptors_2019.style.format(col_format_dict).set_table_styles(df_styles))

## SCRIPT TERMINATION ##
st.stop()
