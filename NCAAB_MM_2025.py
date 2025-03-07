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
# mm_database_2023 = pd.read_xlsx(mm_database_xlsx, index_col='TEAM', header=0, sheet_name='TR')
mm_database_2023 = pd.read_csv(mm_database_csv, index_col='TEAM')
mm_database_2025 = pd.read_csv(mm_database_csv, index_col='TEAM')

## PRE-PROCESSING ##
# players_path.sort_values(by='disc_year', inplace=True)

# ## TIME INTERVALS ##
# today = datetime.date.today()
# before = today - datetime.timedelta(days=1095) #700
# start_date = '2003-01-01'
# end_date = today


## IMAGE IMPORT ##

## FINAL FOUR LOGOS ##
NCAA_logo = Image.open('images/NCAA_logo1.png')
# FF_logo = Image.open('images/FinalFour_2023.png')
# FAU_logo = Image.open('images/FAU_Owls.png')
# Miami_logo = Image.open('images/Miami_Canes.png')
# UConn_logo = Image.open('images/UConn_Huskies.png')
# SDSU_logo = Image.open('images/SDSU_Aztecs.png')



## FORMAT / STYLE ##

## CHART BACKGROUND ##

viz_margin_dict = dict(l=20, r=20, t=50, b=20)
viz_bg_color = '#0360CE' #"LightSteelBlue"
viz_font_dict=dict(size=12, color='#FFFFFF') #  family="Courier New, monospace",
  #F5DFC9 - Court #0360CE - Blue #0360CE - "LightSteelBlue" #FFFFFF - White #000000 - Black


# viz_border_shape = go.layout.Shape(type='rect', xref='paper', yref='paper',  line={'width': 1, 'color': 'black'}) #x0=0, y0=-0.1, x1=1.01, y1=1.02,


## PANDAS STYLER ##

# pd.options.display.float_format = '{:,.2f}'.format
# pd.set_option('display.float_format', lambda x: '%.2f' % x)

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
# IceFire = px.colors.diverging.
# YlOrRd = px.colors.sequential.YlOrRd
# Mint = px.colors.sequential.Mint
# Electric = px.colors.sequential.Electric

# pd.options.display.float_format = '${:,.2f}'.format
# pd.set_option('display.max_colwidth', 200)

## VISUALIZATION LABELS ##

all_cols = [
            ]

live_cols = ['TEAM_KP', 'CONFERENCE', 'CONF_TYPE', 'WIN', 'LOSS', 'SEED', 'PTS/GM',
       'AVG MARGIN', 'FGM/GM', 'FGA/GM', 'OFF EFF', 'DEF EFF', 'eFG%', 'TS%',
       '3PT%', '2PT%', 'FT%', '3PTM/GM', '3PTA/GM', 'OFF REB/GM', 'DEF REB/GM',
       'TTL REB/GM', 'OFF REB%', 'DEF REB%', 'TTL REB%', 'BLKS/GM', 'STL/GM',
       'AST/GM', 'TO/GM', 'AST/TO%', 'WIN% ALL GM', 'WIN% CLOSE GM', 'POSS/GM',
       'PF/GM', 'OPP PTS/GM', 'OPP AVG MARGIN', 'OPP FG%', 'OPP eFG%',
       'OPP 3PT%', 'OPP 2PT%', 'OPP FT%', 'OPP TS%', 'OPP AST/GM', 'OPP TO/GM',
       'OPP AST/TO%', 'NET AST/TOV RATIO', 'STOCKS/GM', 'STOCKS-TOV/GM',
       'NET RANK', 'KenPom RANK', 'KenPom EM', 'KenPom AdjO', 'KenPom AdjD',
       'KenPom AdjT', 'KenPom Luck', 'KenPom SOS EM', 'KenPom SOS OppO',
       'KenPom SOS OppD', 'KenPom NCSOS Adj EM',
       ]

historical_cols = ['Season', 'SznDayNum', 'DayNumGm', 'Date', 'Month', 'WTeamID',
       'WTeam_TR', 'LTeamID', 'LTeam_TR', 'WConf', 'LConf', 'Champion',
       'Reg_MM', 'Net_WRatio', 'W_WRatio', 'L_WRatio', 'WGP', 'LGP', 'WWin',
       'WLoss', 'LWin', 'LLoss', 'NetMarginAvg', 'WMarginAvg', 'LMarginAvg',
       'WMargin', 'LMargin', 'WScore', 'LScore', 'Seed_2023', 'Seed_2022',
       'KP_Rank_2022', 'Seed', 'KP_Rank', 'KP_ADJ_EM', 'KP_ADJ_O', 'KP_ADJ_D',
       'City', 'State', 'Venue', 'Venue_Capacity', 'Mascot', 'Mascot2',
       'tax_family', 'tax_order', 'tax_class', 'tax_phylum', 'tax_kingdom',
       'tax_domain', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM',
       'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM',
       'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO',
       'LStl', 'LBlk', 'LPF', 'WEFG', 'LEFG', 'WTS', 'LTS', 'WStock', 'LStock',
       'WAst/TO', 'LAst/TO', 'WStock/TO', 'LStock/TO', 'WStock_TO',
       'LStock_TO', 'WAvgEFG', 'LAvgEFG', 'WAvgTS', 'LAvgTS', 'WAvgAst/TO',
       'LAvgAst/TO', 'WAvgStock/TO', 'LAvgStock/TO', 'WAvgStock_TO',
       'LAvgStock_TO', 'NetEFG', 'NetTS', 'NetAst/TO', 'NetStock/TO',
       'NetStock_TO', 'NetAvgEFG', 'NetAvgTS', 'NetAvgAst/TO',
       'NetAvgStock/TO', 'NetAvgStock_TO',
       ]

tourney_matchup_cols = ['SEED', 'KenPom RANK', 'NET RANK',
            'WIN', 'LOSS',
            'WIN% ALL GM', 'WIN% CLOSE GM',
            'KenPom EM', #'AdjO', 'AdjD', #'KP_Rank', #'AdjT', 'Luck',
            'KenPom SOS EM', #'SOS OppO', 'SOS OppD', 'NCSOS Adj EM'
            'OFF EFF', 'DEF EFF',
            'AVG MARGIN', #'opponent-average-scoring-margin',
            'PTS/GM', 'OPP PTS/GM',
            'eFG%', 'OPP eFG%', 'TS%', 'OPP TS%',
            'AST/TO%', #'NET AST/TOV RATIO',
            'STOCKS/GM', 'STOCKS-TOV/GM',
            ]

scale_cols = [
            ]

team_df_cols = [
                ]

chart_labels = {'Team': 'TEAM', 'TEAM_KP': 'TEAM', 'TEAM_TR': 'TEAM',
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
                # 'win-pct-all-games':'WIN%', 'win-pct-close-games':'CLOSE GM WIN%',
                # 'offensive-efficiency':'OFF EFF', 'defensive-efficiency':'DEF EFF',
                'Adj EM': 'KenPom ADJ EM', 'SOS Adj EM': 'KenPom SOS ADJ EM',
                'points-per-game': 'PTS/GM', 'opponent-points-per-game': 'OPP PTS/GM',
                'average-scoring-margin': 'AVG SCORE MARGIN', 'opponent-average-scoring-margin': 'OPP AVG SCORE MARGIN',
                'effective-field-goal-pct': 'eFG%', 'true-shooting-percentage': 'TS%',
                'opponent-effective-field-goal-pct': 'OPP eFG%', 'opponent-true-shooting-percentage': 'OPP TS%',
                'assist--per--turnover-ratio': 'AST_TOV_RATIO', 'NET AST/TOV RATIO': 'NET AST/TOV%',
                'STOCKS-per-game': 'STOCKS/GM', 'STOCKS-TOV-per-game': 'STOCKS-TOV/GM',
                'MMS STOCKS-TOV-per-game': 'STOCKS-TOV/GM', 'MMS Adj EM': 'KenPom ADJ EM',

                }

# team_logos_dict = {'FAU OWLS':FAU_logo,
#                    'MIAMI HURRICANES':Miami_logo,
#                    'UCONN HUSKIES':UConn_logo,
#                    'SDSU AZTECS':SDSU_logo,
#                    }

#https://nbacolors.com/team/washington-wizards-color
#https://teamcolorcodes.com/nba-team-color-codes/

# team_colors_dict = {'FAU OWLS':['#E03A3E','#C1D32F','#26282A'],
#                     'MIAMI HURRICANES':['#000000','#FFFFFF'],
#                     'UCONN HUSKIES':['#007A33','#007A33'],
#                     'SDSU AZTECS':['#000000','#FFFFFF'],
#                   }

## COURT BACKGROUND ##
# court_img_dict = dict(source=court_img_1, xref="paper", yref="paper", x=0.5, y=0.5, sizex=2.5, sizey=1,
#                       xanchor="center", yanchor="middle", #left right  #top bottom
#                       opacity=.35, visible=True, layer="below", # sizing="contain",
#                       )

## FEATURED VARIABLES ##

# team_list = list(champion_players['TEAM'].unique())

team_logos_list = []

eastconf_logos_list = []

westconf_logos_list = []


# champion_players['LOGO'] = champion_players.TEAM.map(team_logos_dict)


#%%
# # group_cols = ['']
# college_raptor = champion_players.groupby(['TEAM', 'COLLEGE']).mean()
# print(college_raptor)

# Keep? Functional?
# champion_players['LOGO'] = champion_players.TEAM.map(team_logos_dict)


## FILTER DATA ##
mm_database_2025 = mm_database_2025[tourney_matchup_cols]

top_KP100 = mm_database_2025[mm_database_2025['KP_Rank'] <= 100]
#tourney_2023 = mm_database_2024[mm_database_2024['NET_24'] > 0]
#tourney_2023 = tourney_2023[tourney_matchup_cols]

# tourney_2023 = tr_data_hub[tr_data_hub['Season'] > 2016]

# reg_szn_games = tourney_2023[tourney_2023['Reg_MM'] == 0]
# madness_games = tourney_2023[tourney_2023['Reg_MM'] == 1]


# DATAFRAMES

## COPY DATAFRAME ##
heatmap_tourney_2023 = mm_database_2025.copy()
heatmap_tourney_2023.loc['TOURNEY AVG'] = heatmap_tourney_2023.mean()


## HEATMAP DATAFRAMES ##
heatmap_tourney_2023_T = pd.DataFrame(heatmap_tourney_2023.T)
# heatmap_tourney_2023_T = pd.DataFrame(heatmap_tourney_2023_T)


East_region_2023 = heatmap_tourney_2023_T[['Purdue', 'Marquette', 'Kansas St', 'Tennessee',
                                           'Duke', 'Kentucky', 'Michigan St', 'Memphis',
                                           'Fla Atlantic', 'USC', 'Providence', 'Oral Roberts',
                                           'Lafayette', 'Montana St', 'Vermont', 'F Dickinson',
                                           'TOURNEY AVG',]]

West_region_2023 = heatmap_tourney_2023_T[['Kansas', 'UCLA', 'Gonzaga', 'Connecticut',
                                           'St Marys', 'TX Christian', 'Northwestern', 'Arkansas',
                                           'Illinois', 'Boise State', 'Arizona St', 'VCU',
                                           'Iona', 'Grd Canyon', 'NC-Asheville', 'Howard',
                                           'TOURNEY AVG',]]

South_region_2023 = heatmap_tourney_2023_T[['Alabama', 'Arizona', 'Baylor', 'Virginia',
                                           'San Diego St', 'Creighton', 'Missouri', 'Maryland',
                                           'W Virginia', 'Utah State', 'NC State', 'Col Charlestn',
                                           'Furman', 'UCSB', 'Princeton', 'TX A&M-CC',
                                           'TOURNEY AVG',]]

Midwest_region_2023 = heatmap_tourney_2023_T[['Houston', 'Texas', 'Xavier', 'Indiana',
                                           'Miami (FL)', 'Iowa State', 'Texas A&M', 'Iowa',
                                           'Auburn', 'Penn State', 'Pittsburgh', 'Drake',
                                           'Kent State', 'Kennesaw St', 'Colgate', 'N Kentucky',
                                           'TOURNEY AVG',]]

East_region = East_region_2023
West_region = West_region_2023
South_region = South_region_2023
Midwest_region = Midwest_region_2023

East_region['EAST AVG'] = East_region_2023.mean(numeric_only=True, axis=1)
East_region['WEST AVG'] = West_region_2023.mean(numeric_only=True, axis=1)
East_region['SOUTH AVG'] = South_region_2023.mean(numeric_only=True, axis=1)
East_region['MIDWEST AVG'] = Midwest_region_2023.mean(numeric_only=True, axis=1)

West_region['WEST AVG'] = West_region_2023.mean(numeric_only=True, axis=1)
West_region['EAST AVG'] = East_region_2023.mean(numeric_only=True, axis=1)
West_region['SOUTH AVG'] = South_region_2023.mean(numeric_only=True, axis=1)
West_region['MIDWEST AVG'] = Midwest_region_2023.mean(numeric_only=True, axis=1)

South_region['SOUTH AVG'] = South_region_2023.mean(numeric_only=True, axis=1)
South_region['EAST AVG'] = East_region_2023.mean(numeric_only=True, axis=1)
South_region['WEST AVG'] = West_region_2023.mean(numeric_only=True, axis=1)
South_region['MIDWEST AVG'] = Midwest_region_2023.mean(numeric_only=True, axis=1)

Midwest_region['MIDWEST AVG'] = Midwest_region_2023.mean(numeric_only=True, axis=1)
Midwest_region['EAST AVG'] = East_region_2023.mean(numeric_only=True, axis=1)
Midwest_region['SOUTH AVG'] = South_region_2023.mean(numeric_only=True, axis=1)
Midwest_region['WEST AVG'] = West_region_2023.mean(numeric_only=True, axis=1)

styler_dict = {"SEED": "Spectral_r", 'KenPom RANK': "Spectral_r", 'NET RANK': "Spectral_r",
               "WIN": "Spectral", "LOSS": "Spectral_r",
               'WIN% ALL GM': "Spectral", 'WIN% CLOSE GM': "Spectral",
               'KenPom EM':"Spectral", 'KenPom SOS EM':"Spectral",
               'OFF EFF':"Spectral", 'DEF EFF':"Spectral_r",
               'AVG MARGIN':"Spectral", #'OPP AVG MARGIN':"Spectral_r",
               'PTS/GM':"Spectral", 'OPP PTS/GM':"Spectral_r",
               'eFG%':"Spectral", 'OPP eFG%':"Spectral_r",
               'TS%':"Spectral", 'OPP TS%':"Spectral_r",
               'AST/TO%':"Spectral", #'NET AST/TOV RATIO',
               'STOCKS/GM':"Spectral", 'STOCKS-TOV/GM':"Spectral",
               }

# plot_color_gradients('Diverging',
#                      ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
#                       'Spectral', 'Spectral', 'coolwarm', 'bwr', 'seismic'])

East_region_styler = East_region.style
West_region_styler = West_region.style
South_region_styler = South_region.style
Midwest_region_styler = Midwest_region.style
for idx, cmap in styler_dict.items():
    East_region_styler = East_region_styler.background_gradient(cmap=cmap, subset=pd.IndexSlice[idx, :], axis=1)
    West_region_styler = West_region_styler.background_gradient(cmap=cmap, subset=pd.IndexSlice[idx, :], axis=1)
    South_region_styler = South_region_styler.background_gradient(cmap=cmap, subset=pd.IndexSlice[idx, :], axis=1)
    Midwest_region_styler = Midwest_region_styler.background_gradient(cmap=cmap, subset=pd.IndexSlice[idx, :], axis=1)

## HEADERS / INDEX ##

header = {'selector': 'th',
          'props': [('background-color', '#0360CE'), ('color', 'white'),  # 00008b ##5b9bd5
                    ('text-align', 'center'), ('vertical-align', 'center'),
                    ('font-weight', 'bold'),
                    ('border-bottom', '2px solid #000000'),
                    ]}

header_level0 = {'selector': 'th.col_heading.level0',
                 'props': [('font-size', '12px'),
                           # ('min-width:', '100px'), ('max-width:', '100px'), ('column-width:', '100px'),
                           ]}

index = {'selector': 'th.row_heading',
         'props': [('background-color', '#000000'), ('color', 'white'),
                   ('text-align', 'center'), ('vertical-align', 'center'),
                   ('font-weight', 'bold'), ('font-size', '12px'),
                   # ('min-width:', '100px'), ('max-width:', '100px'), ('column-width:', '100px'),

                   ]}

numbers = {'selector': 'td.data',
           'props': [('text-align', 'center'), ('vertical-align', 'center'),
                     ('font-weight', 'bold')]}  # ('background-color', '#0360CE'), ('color', 'white')

borders_right = {'selector': '.row_heading.level1',
                 'props': [('border-right', '1px solid #FFFFFF')]}

## ROWS ##

top_row = {'selector': 'td.data.row0',
           # [('background-color', '#000000'), ('color', 'white'),
           'props': [('border-bottom', '2px dashed #000000'),
                     ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')], }

table_row0 = {'selector': '.row0',
              'props': [('border-bottom', '2px dashed #000000'),
                        ('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}  # ('border-top', '1px solid #000000')

table_row1 = {'selector': '.row1',
              'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '12px')]}

table_row2 = {'selector': '.row2',  # 'selector': '.row_heading.level0.row1'
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

## COLUMNS ##

table_col0 = {'selector': '.row0',
              'props': [('border-left', '3px solid #000000'),
                        ('min-width:', '75px'), ('max-width:', '75px'), ('column-width:', '75px'),
                        ]}

table_col1 = {'selector': '.col1',
              'props': [('border-left', '2px dashed #000000')]}

table_col2 = {'selector': '.col2',  # 'selector': '.row_heading.level0.row1'
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

# apply styles
# table = table.style.set_table_styles([header, header_level0, index, top_row, borders_bottom1, borders_bottom2, borders_bottom3, borders_bottom4, borders_bottom5, borders_right])

## FILL BARS ##
# df.style.bar(subset = ["D"],
#              align = "mid",
#              color = ["salmon", "lightgreen"])\
#         .set_properties(**{'border': '0.5px solid black'})


## TO DO / NOTES
## TABLEAU ## NEO4J ## MONGODB ##
## PAGES OR TABS FOR EACH ROSTER (???) ##
## AGG GRID ##
    # grid = AgGrid(df)

## SCALING ##
    # ss = StandardScaler()
    # mms = MinMaxScaler()

    # scaled_MP = ss.fit_transform(champion_players['MP'])
    # scaled_MP = scaled_MP.reshape(-1,1)
    # champion_players['ss_MP'] = scaled_MP
    # print(champion_players['ss_MP'])


## GROUP BY CHAMPIONSHIP TEAM

# champion_teams = champion_players[team_df_cols]

# champ_df_list = [
#                  ]
#
# for df in champ_df_list:
#   df = df.drop(columns=['CHAMP', 'PLAYER'], inplace=True)





#%%
## VISUALIZATIONS ##





####################################################################################################################

## BOX PLOTS ##

####################################################################################################################

# ## BAR CHARTS ##
# bar_usg_salary = px.bar(data_frame=champion_players,
#                               x=champion_players['CHAMP'],
#                               y=champion_players['SALARY'],
#                               color=champion_players['USG%'],     # EXPERIENCE AGE MP APE
#                               color_continuous_scale=Tropic,
#                               color_discrete_sequence=Tropic,
#                               # color_continuous_midpoint=10,
#                               # color_discrete_map=team_logos_dict,
#                               hover_name=champion_players['PLAYER'],
#                               hover_data=champion_players[['CHAMP', 'SALARY', 'MP']], #'WS/$',
#                               barmode='group',
#                               title='USG% RELATIVE TO TEAM SALARY',
#                               labels=chart_labels,
#                               # template='simple_white+gridon',
#                               # range_x=[1991,2023],
#                               # range_y=[0,200000000],
#                               height=750,
#                             # category_orders={"InternetService": ["DSL", "Fiber optic", "No"],
# #                               "gender": ["Female", "Male"]})
#                               )
#
# bar_WS_salary = px.bar(data_frame=champion_players,
#                               x=champion_players['CHAMP'],
#                               y=champion_players['SALARY'],
#                               color=champion_players['WS'],     # EXPERIENCE AGE MP APE
#                               color_continuous_scale=Tropic,
#                               color_discrete_sequence=Tropic,
#                               color_continuous_midpoint=10,
#                               # color_discrete_map=team_logos_dict,
#                               hover_name=champion_players['PLAYER'],
#                               hover_data=champion_players[['CHAMP', 'SALARY', 'MP', 'WS']], #'WS/$',
#                               barmode='group',
#                               title='WIN SHARES (WS) METRIC RELATIVE TO CHAMPIONSHIP TEAM SALARY',
#                               labels=chart_labels,
#                               # template='simple_white+gridon',
#                               # range_x=[1991,2023],
#                               # range_y=[0,200000000],
#                               height=750,
#                               # width=1000,
#                               )
#
# bar_raptor_salary = px.bar(data_frame=champion_players,
#                               x=champion_players['CHAMP'],
#                               y=champion_players['SALARY'],
#                               color=champion_players['RAPTOR'],     # EXPERIENCE AGE MP APE
#                               color_continuous_scale=Tropic,
#                               color_discrete_sequence=Tropic,
#                               color_continuous_midpoint=0,
#                               # color_discrete_map=team_logos_dict,
#                               hover_name=champion_players['PLAYER'],
#                               hover_data=champion_players[['CHAMP', 'SALARY', 'MP', 'RAPTOR']],
#                               barmode='group',
#                               title='RAPTOR METRIC RELATIVE TO CHAMPIONSHIP TEAM SALARY',
#                               labels=chart_labels,
#                               # template='simple_white+gridon',
#                               # range_x=[1991,2022],
#                               # range_y=[0,200000000],
#                               height=750,
#                               # width=1000,
#                               )
#
# bar_lebron_salary = px.bar(data_frame=lebron_val_players,
#                               x=lebron_val_players['CHAMP'],
#                               y=lebron_val_players['SALARY'],
#                               color=lebron_val_players['LEBRON'],     # EXPERIENCE AGE MP APE
#                               color_continuous_scale=Tropic,
#                               color_discrete_sequence=Tropic,
#                               # color_continuous_midpoint=10,
#                               # color_discrete_map=team_logos_dict,
#                               hover_name=lebron_val_players['PLAYER'],
#                               hover_data=lebron_val_players[['CHAMP', 'SALARY', 'MP', 'LEBRON']],
#                               barmode='group',
#                               title='LEBRON METRIC RELATIVE TO CHAMPIONSHIP TEAM SALARY',
#                               labels=chart_labels,
#                               # template='simple_white+gridon',
#                               # range_x=[1991,2022],
#                               # range_y=[0,200000000],
#                               height=750,
#                               # width=1000,
#                               )
#
# bar_eWINS_WS = px.bar(data_frame=champion_players,
#                               x=champion_players['CHAMP'],
#                               y=champion_players['SALARY'],
#                               color=champion_players['$MM/PlrWS'],     # EXPERIENCE AGE MP APE
#                               color_continuous_scale=Tropic,
#                               color_discrete_sequence=Tropic,
#                               # color_continuous_midpoint=10,
#                               # color_discrete_map=team_logos_dict,
#                               hover_name=champion_players['PLAYER'],
#                               hover_data=champion_players[['CHAMP', 'SALARY', 'MP']], #'WS/$',
#                               barmode='group',
#                               title='$MM / PLAYER WS RELATIVE TO TEAM SALARY',
#                               labels=chart_labels,
#                               # template='simple_white+gridon',
#                               # range_x=[1991,2023],
#                               # range_y=[0,200000000],
#                               height=750,
#                             # category_orders={"InternetService": ["DSL", "Fiber optic", "No"],
# #                               "gender": ["Female", "Male"]})
#                               )
#
#
#
# ####################################################################################################################
#
# ## SCATTER MATRIX ##
# scatter_matrix_metrics = px.scatter_matrix(champion_players,
#                                          dimensions=['USG%', 'TS%', 'AST%/TO%', 'STOCK%', 'RAPTOR', 'LEBRON', 'WS', ],
#                                          color=champion_players['WTD POS'],
#                                          color_continuous_scale=Dense,
#                                          color_discrete_sequence=Dense,
#                                            # symbol=champion_players['TEAM'],
#                                            # # symbol_sequence=team_logos_dict,
#                                            # symbol_map=team_logos_dict,
#                                          # color_discrete_map=team_logos_dict,
#                                          hover_name=champion_players['PLAYER'],
#                                          hover_data=champion_players[['MP', 'CHAMP']],
#                                          title='PLAYER PERFORMANCE BY CHAMPIONSHIP TEAM',
#                                          labels=chart_labels,
#                                          # custom_data= [league_logo_list],
#                                          height=800,
#                                          # width=800,
#                                          )
#
# scatter_matrix_measurables = px.scatter_matrix(champion_players,
#                                          dimensions=['BMI', 'APE', 'WS', 'RAPTOR', 'LEBRON'],
#                                          color=champion_players['WTD POS'],
#                                          color_continuous_scale=Dense,
#                                          color_discrete_sequence=Dense,
#                                          # color_discrete_map=team_logos_dict,
#                                          hover_name=champion_players['PLAYER'],
#                                          hover_data=champion_players[['MP', 'CHAMP']],
#                                          title='PLAYER PERFORMANCE BY CHAMPIONSHIP TEAM',
#                                          labels=chart_labels,
#                                          # custom_data= [league_logo_list],
#                                          height=800,
#                                          # width=800,
#                                          )
#
#
# ####################################################################################################################
#
# ## SCATTER TERNARY ##
#
# scatter_ternary_stl_blk_ast_to = px.scatter_ternary(data_frame=champion_players,
#                                        a=champion_players['STL%'],
#                                        b=champion_players['BLK%'],
#                                        c=champion_players['AST%/TO%'],
#                                        color=champion_players['WTD POS'],
#                                         color_discrete_sequence=Dense,
#                                        color_continuous_scale=Dense,
#                                         color_continuous_midpoint=3,
#                                        # symbol=champion_players['TEAM'], #'RD POS'
#                                        #              symbol_map=team_logos_dict,
#                                        size=champion_players['TS%'],
#                                        size_max=20,
#                                        opacity=.8,
#                                        title='NBA CHAMPIONS -- STL% - BLK% - AST%/TO%',
#                                        hover_name=champion_players['PLAYER'],
#                                        hover_data=champion_players[['CHAMP', 'SALARY', 'MP',]],
#                                        labels=chart_labels,
#                                        height=900,
#                                        )
#
# scatter_ternary_ast_to_usg = px.scatter_ternary(data_frame=champion_players,
#                                        a=champion_players['AST%'],
#                                        b=champion_players['TO%'],
#                                        c=champion_players['USG%'],
#                                        color=champion_players['WTD POS'],
#                                                 color_discrete_sequence=Dense,
#                                                 color_continuous_scale=Dense,
#                                                 color_continuous_midpoint=3,
#                                                 symbol=champion_players['RD POS'],
#                                                 size=champion_players['TS%'],
#                                                 size_max=20,
#                                        opacity=.8,
#                                        title='NBA CHAMPIONS -- TS% - USG% - STOCK%',
#                                        hover_name=champion_players['PLAYER'],
#                                        hover_data=champion_players[['CHAMP', 'SALARY', 'MP', 'STL%', 'BLK%']],
#                                        labels=chart_labels,
#                                        height=900,
#                                        )
#
#
# ####################################################################################################################
#
# ## SCATTER 3D ##
#
# scatter_3D_to_ast_usg = px.scatter_3d(data_frame=champion_players,
#                                      x=champion_players['TO%'],
#                                      y=champion_players['AST%'],
#                                      z=champion_players['USG%'],
#                                      color=champion_players['WTD POS'],
#                                       symbol=champion_players['RD POS'],
#                                      color_discrete_sequence=Dense,
#                                      color_continuous_scale=Dense,
#                                      color_continuous_midpoint=3,
#                                      title='NBA CHAMPIONS -- TO% / AST% / USG% BY POSITION',
#                                      hover_name=champion_players['PLAYER'],
#                                      hover_data=champion_players[['CHAMP']], #'LOGO'
#                                      # 'HEIGHT (IN)' 'WEIGHT (LBS)' 'BMI' 'W-SPAN (IN)'
#                                      # custom_data=['LOGO'],
#                                      # size=champion_players['WS'],
#                                      # size_max=50,
#                                      labels=chart_labels,
#                                      # range_x=[0,360],
#                                      # range_y=[-50,50],
#                                      # range_z=[0,2500],
#                                      # range_color=Sunsetdark,
#                                      opacity=.8,
#                                      height=1000,
#                                      # width=1000,
#                                      )
#
# #%%
#
# ## HISTORICAL LINE CHARTS
# box_eWINS_WS = px.bar(data_frame=champion_players,
#                               x=champion_players['CHAMP'],
#                               y=champion_players['$MM/eWIN'], #, '$MM/TmWIN']
#                               color=champion_players['WTD POS'],     # EXPERIENCE AGE MP APE
#                               color_discrete_sequence=Tropic,
#                       color_continuous_scale=Dense,
#                               # color_continuous_midpoint=10,
#                               # color_discrete_map=team_logos_dict,
#                               hover_name=champion_players['PLAYER'],
#                               hover_data=champion_players[['CHAMP',]], #'WS/$',
#                               title='$MM / PLAYER WS RELATIVE TO TEAM SALARY',
#                               labels=chart_labels,
#                               # template='simple_white+gridon',
#                               # range_x=[1991,2023],
#                               # range_y=[0,200000000],
#                               height=750,
#                             # category_orders={"InternetService": ["DSL", "Fiber optic", "No"],
# #                               "gender": ["Female", "Male"]})
#                               )

#%%

####################################################################################################################
## TRACE LINES ##

# line_NBA_salary = go.Scatter(x=champion_players['CHAMP'], y=champion_players['NBA TM AVG SAL'],
#                              line_color='#000000', mode='lines')
#
# line_NBA_eWIN = go.Scatter(x=champion_players['CHAMP'], y=champion_players['$MM/eWIN'],
#                              line_color='#000000', mode='lines')

####################################################################################################################

## LOGO OVERLAY
# for x,y, png in zip(fig.data[0].x, fig.data[0].y, Path.cwd().joinpath("nfl-logos").glob("*.png")):
#     fig.add_layout_image(
#         x=x,
#         y=y,
#         source=Image.open(png),
#         xref="x",
#         yref="y",
#         sizex=2,
#         sizey=2,
#         xanchor="center",
#         yanchor="middle",
#     )

####################################################################################################################

## DENSITY MAP
##

####################################################################################################################
#%%

#####################
### STREAMLIT APP ###
#####################

## CONFIGURATION ##
st.set_page_config(page_title='NCAA BASKETBALL -- MARCH MADNESS 2023', layout='wide', initial_sidebar_state='auto') #, page_icon=":smirk:"

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
            ('color', '#EBEDE9'), #6d6d6d #29609C
            ('background-color', '#29609C') #f7f7f9
            ]

td_props = [('font-size', '12px'),
            # ('text-align', 'center'),
            # ('font-weight', 'bold'),
            # ('color', '#EBEDE9'), #6d6d6d #29609C
            # ('background-color', '#29609C') #f7f7f9
            ]

df_styles = [dict(selector="th", props=th_props),
             dict(selector="td", props=td_props)]


col_format_dict = {
    # 'WTD POS': "{:.2}",
    #                'BMI': "{:.1}",
    #                'W-SPAN (IN)': "{:.1}",
    #                'APE': "{:.1}",
    #                'USG%': "{:.1}",
    #                'TS%': "{:.1}",
    #                'AST%/TO%': "{:.1}",
    #                'STOCK%': "{:.1}",
    #                # 'USG%': "{:.1}",
    #
    #                'WS': "{:,}",

#                    'O-WS': "{:,}",
#                    'D-WS': "{:,}",

#                    'TM_WS': "{:,}",
                   # #: "{:.1%}", #:"{:.1}x", "${:.2}", #"${:,}"
                   }


## SIDEBAR ##
# sidebar_header = st.sidebar.subheader('DIRECTORY:')

# sector_sidebar_select = st.sidebar.selectbox('SECTOR', (sector_list_of_names), help='SELECT CRE SECTOR')
# ticker_sidebar_select = st.sidebar.selectbox('TICKER', (sector_dict['apartment'])) #sector_sidebar_select

# sidebar_start = st.sidebar.date_input('START DATE', before)
# sidebar_end = st.sidebar.date_input('END DATE', today)
# if sidebar_start < sidebar_end:
#     st.sidebar.success('START DATE: `%s`\n\nEND DATE: `%s`' % (sidebar_start, sidebar_end))
# else:
#     st.sidebar.error('ERROR: END DATE BEFORE START DATE')



## HEADER ##
st.container()

st.title('NCAA BASKETBALL -- MARCH MADNESS 2025')
st.write('*2025 MARCH MADNESS RESEARCH HUB*')

## LOGOS ##
#MM_col_0, MM_col_1, MM_col_2, MM_col_3, MM_col_4, = st.columns(5)

#MM_col_0.image(Miami_logo, caption='MIAMI HURRICANES', width=200)
#MM_col_1.image(FAU_logo, caption='FAU OWLS', width=200)
# MM_col_2.image(NCAA_logo, caption='NCAAB')
#MM_col_2.image(FF_logo, width=200)
#MM_col_3.image(UConn_logo, caption='UCONN HUSKIES', width=200)
#MM_col_4.image(SDSU_logo, caption='SDSU AZTECS', width=200)


tab_0, tab_1, tab_2, tab_3, tab_4 = st.tabs(['2025', 'TBU', 'TBU', 'TBU', 'TBU', ])
#tab_5, tab_6, tab_7, tab_8, tab_9, tab_10, \
#tab_11, tab_12, tab_13, tab_14, tab_15, tab_16, tab_17, tab_18, tab_19, tab_20, \
    # = st.tabs(['2025', 'TBU', 'TBU', 'TBU', 'TBU',
    #            # '2022', '2021', '2020', '2019', '2018',
    #            # '2017', '2016', '2015', '2014', '2013',
    #            # '2012', '2011', '2010', '2009', '2008',
    #            # '2007', '2006', '2005', '2004', '2003',
    #            # '2002', '2001', '2000', '1999', '1998',
    #            # '1997', '1996', '1995', '1994', '1993',
    #            # '1992', '1991',
    #            ])


with tab_0:

## REGIONAL HEATMAPS ##

    ## EAST REGION ##
    st.subheader('EAST REGION')

    st.markdown(
        East_region_styler.format('{:.2f}', na_rep='NA').set_table_styles([header, header_level0, index, top_row,
                                                                           numbers, borders_right,
                                                                           table_row1, table_row2, table_row3,
                                                                           table_row4, table_row5, table_row6,
                                                                           table_row7, table_row8, table_row9,
                                                                           table_row10, table_row11, table_row12,
                                                                           table_row13, table_row14, table_row15,
                                                                           table_row16, table_row17, table_row18,
                                                                           table_row19, table_row20, table_row21,
                                                                           table_col1, table_col2, table_col3,
                                                                           table_col4, table_col5, table_col6,
                                                                           table_col7, table_col8, table_col9,
                                                                           table_col10, table_col11, table_col12,
                                                                           table_col13, table_col14, table_col15,
                                                                           table_col16, table_col17, table_col18,
                                                                           table_col19, table_col20, table_col21,

                                                                           ]).set_properties(**{'min-width': '55px'},
                                                                                             **{'max-width': '55px'},
                                                                                             **{'column-width': '55px'},
                                                                                             **{'width': '55px'},
                                                                                             ).to_html(table_uuid='east_region'), unsafe_allow_html=True)

    ## WEST REGION ##
    st.subheader('WEST REGION')
    st.markdown(
        West_region_styler.format('{:.2f}', na_rep='NA').set_table_styles([header, header_level0, index, top_row,
                                                                       numbers, borders_right,
                                                                       table_row1, table_row2, table_row3,
                                                                       table_row4, table_row5, table_row6,
                                                                       table_row7, table_row8, table_row9,
                                                                       table_row10, table_row11, table_row12,
                                                                       table_row13, table_row14, table_row15,
                                                                       table_row16, table_row17, table_row18,
                                                                       table_row19, table_row20, table_row21,
                                                                       table_col1, table_col2, table_col3,
                                                                       table_col4, table_col5, table_col6,
                                                                       table_col7, table_col8, table_col9,
                                                                       table_col10, table_col11, table_col12,
                                                                       table_col13, table_col14, table_col15,
                                                                       table_col16, table_col17, table_col18,
                                                                       table_col19, table_col20, table_col21,

                                                                       ]).set_properties(**{'min-width': '55px'},
                                                                                         **{'max-width': '55px'},
                                                                                         **{'column-width': '55px'},
                                                                                         **{'width': '55px'},
                                                                                         ).to_html(table_uuid='east_region'), unsafe_allow_html=True)

    ## SOUTH REGION ##
    st.subheader('SOUTH REGION')
    st.markdown(
        South_region_styler.format('{:.2f}', na_rep='NA').set_table_styles([header, header_level0, index, top_row,
                                                                        numbers, borders_right,
                                                                        table_row1, table_row2, table_row3,
                                                                        table_row4, table_row5, table_row6,
                                                                        table_row7, table_row8, table_row9,
                                                                        table_row10, table_row11, table_row12,
                                                                        table_row13, table_row14, table_row15,
                                                                        table_row16, table_row17, table_row18,
                                                                        table_row19, table_row20, table_row21,
                                                                        table_col1, table_col2, table_col3,
                                                                        table_col4, table_col5, table_col6,
                                                                        table_col7, table_col8, table_col9,
                                                                        table_col10, table_col11, table_col12,
                                                                        table_col13, table_col14, table_col15,
                                                                        table_col16, table_col17, table_col18,
                                                                        table_col19, table_col20, table_col21,

                                                                        ]).set_properties(**{'min-width': '55px'},
                                                                                          **{'max-width': '55px'},
                                                                                          **{'column-width': '55px'},
                                                                                          **{'width': '55px'},
                                                                                          ).to_html(table_uuid='east_region'), unsafe_allow_html=True)

    ## MIDWEST REGION ##
    st.subheader('MIDWEST REGION')
    st.markdown(
        Midwest_region_styler.format('{:.2f}', na_rep='NA').set_table_styles([header, header_level0, index, top_row,
                                                                          numbers, borders_right,
                                                                          table_row1, table_row2, table_row3,
                                                                          table_row4, table_row5, table_row6,
                                                                          table_row7, table_row8, table_row9,
                                                                          table_row10, table_row11, table_row12,
                                                                          table_row13, table_row14, table_row15,
                                                                          table_row16, table_row17, table_row18,
                                                                          table_row19, table_row20, table_row21,
                                                                          table_col1, table_col2, table_col3,
                                                                          table_col4, table_col5, table_col6,
                                                                          table_col7, table_col8, table_col9,
                                                                          table_col10, table_col11, table_col12,
                                                                          table_col13, table_col14, table_col15,
                                                                          table_col16, table_col17, table_col18,
                                                                          table_col19, table_col20, table_col21,

                                                                          ]).set_properties(**{'min-width': '55px'},
                                                                                            **{'max-width': '55px'},
                                                                                            **{'column-width': '55px'},
                                                                                            **{'width': '55px'},
                                                                                            ).to_html(table_uuid='east_region'), unsafe_allow_html=True)

    # st.plotly_chart(scatter_matrix_metrics, use_container_width=True, sharing="streamlit")
    # st.image(capstone_court, width=1000, use_column_width=True)

    ## SUB-COLUMNS ##

    # left, right = st.columns(2)
    # with left:
    #     st.plotly_chart(scatter_ternary_ast_to_usg, use_container_width=True, sharing="streamlit") #.add_layout_image(court_img_dict)
    # with right:
    #     st.plotly_chart(scatter_ternary_stl_blk_ast_to, use_container_width=True, sharing="streamlit") #.add_layout_image(court_img_dict)



    ## SCATTER MATRIX ##
    # st.plotly_chart(scatter_matrix_teams, use_container_width=True, sharing="streamlit")

    # st.plotly_chart(scatter_matrix_measurables, use_container_width=True, sharing="streamlit")


    # st.plotly_chart(box_eWINS_WS.add_traces(line_NBA_eWIN).add_layout_image(court_img_dict), use_container_width=True, sharing="streamlit")

    ## 3D SCATTER ##
    # st.plotly_chart(scatter_3D_to_ast_usg.add_layout_image(court_img_dict_3D), use_container_width=True, sharing="streamlit")



    ## LEAGUE LOGOS ##
    # east_col_1, nba_col_2, west_col_3 = st.columns(3)
    # east_col_1.image(East_logo, width=250) # caption='WESTERN CONFERENCE'
    # nba_col_2.image(nba_logo_1, width=300) # caption='NATIONAL BASKETBALL ASSOCIATION'
    # west_col_3.image(West_logo, width=250) # caption='EASTERN CONFERENCE'



    ## FORM FUNCTIONS ##
    # @st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)


    ## DISCOVERY INFORMATION ##
    # st.plotly_chart(disc_info_1.update_yaxes(categoryorder='total ascending'), use_container_width=True, sharing="streamlit")


    ## EXTERNAL LINKS ##

    github_link1 = '[GITHUB <> NE](https://github.com/nehat312/march-madness-2025/)'
    github_link2 = '[GITHUB <> TP](https://github.com/Tyler-Pickett/march-madness-2023/)'
    kenpom_site_link = '[KENPOM](https://kenpom.com/)'

    link_col_1, link_col_2, link_col_3 = st.columns(3)
    ext_link_1 = link_col_1.markdown(github_link1, unsafe_allow_html=True)
    ext_link_2 = link_col_2.markdown(github_link2, unsafe_allow_html=True)
    ext_link_3 = link_col_3.markdown(kenpom_site_link, unsafe_allow_html=True)


with tab_1:
    st.subheader('EXAMPLE SUBHEADER')
    st.image(NCAA_logo, width=200)

    ## LEAGUE LOGOS ##
    # east_col_1, nba_col_2, west_col_3 = st.columns(3)
    # east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    # nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    # west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_2:
    st.subheader('EXAMPLE SUBHEADER')
    st.image(NCAA_logo, width=200)
    # st.dataframe(mil_bucks_2021.style.format(col_format_dict).set_table_styles(df_styles))

with tab_3:
    st.subheader('EXAMPLE SUBHEADER')
    st.image(NCAA_logo, width=200)
    # st.dataframe(lal_lakers_2020.style.format(col_format_dict).set_table_styles(df_styles))

with tab_4:
    st.subheader('EXAMPLE SUBHEADER')
    st.image(NCAA_logo, width=200)
    # st.dataframe(tor_raptors_2019.style.format(col_format_dict).set_table_styles(df_styles))


## SCRIPT TERMINATION ##
st.stop()




### INTERPRETATION ###




### SCRATCH NOTES ###


# top-level filters
# job_filter = st.selectbox (“Select the status”, pd.unique (df(“job”)))
# df = df [df [“status”] == status_filter]

## METRICS - TOP of each team?
# kpix.metric
# {
# label = “Fitness”,
# value = f”$ { round ( balance, 4) }”,
# delta = - round (balance / count_fitness) * 100,
# }


## PAGE BACKGROUND ##

# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("images/Court1.png");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )
#
# add_bg_from_url()


# CONFIG TEMPLATE
    # st.set_page_config(page_title="CSS hacks", page_icon=":smirk:")
    #
    # c1 = st.container()
    # st.markdown("---")
    # c2 = st.container()
    # with c1:
    #     st.markdown("Hello")
    #     st.slider("World", 0, 10, key="1")
    # with c2:
    #     st.markdown("Hello")
    #     st.slider("World", 0, 10, key="2")

# STYLE WITH CSS THROUGH MARKDOWN
    # st.markdown("""
    # <style>
    # div[data-testid="stBlock"] {
    #     padding: 1em 0;
    #     border: thick double #32a1ce;
    # }
    # </style>
    # """, unsafe_allow_html=True)


# STYLE WITH JS THROUGH HTML IFRAME
    # components.html("""
    # <script>
    # const elements = window.parent.document.querySelectorAll('div[data-testid="stBlock"]')
    # console.log(elements)
    # elements[0].style.backgroundColor = 'paleturquoise'
    # elements[1].style.backgroundColor = 'lightgreen'
    # </script>
    # """, height=0, width=0)


# st.markdown("""
#             <style>
#             div[data-testid="stBlock"] {padding: 1em 0; border: thick double #32a1ce; color: blue}
#             </style>
#             """,
#             unsafe_allow_html=True)

# style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)',
#                                            'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
#                                            'border': '4px solid black', 'font-family': 'Arial'}),

#pattern_shape = "nation", pattern_shape_sequence = [".", "x", "+"]

            # fig = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group", facet_row="time", facet_col="day",
            #        category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]})

            # fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"], color="species")

            # fig = px.parallel_categories(df, color="size", color_continuous_scale=px.colors.sequential.Inferno)

            # fig = px.parallel_coordinates(df, color="species_id", labels={"species_id": "Species",
            #                   "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
            #                   "petal_width": "Petal Width", "petal_length": "Petal Length", },
            #                     color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)

