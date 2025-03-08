#########################################
# NCAAB MARCH MADNESS 2025 - MAIN APP  #
#########################################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import os

# ---------------------------------------------------------------------------
# 1) PATH TO YOUR GITHUB-BASED CSV
# ---------------------------------------------------------------------------
abs_path = r'https://raw.githubusercontent.com/nehat312/march-madness-2025/main'
mm_database_csv = abs_path + '/data/mm_2025_database.csv'

# ---------------------------------------------------------------------------
# 2) READ THE DATA
#    We assume the CSV's first column is the school/team names,
#    so index_col=0 will set that column as the DataFrame index.
# ---------------------------------------------------------------------------
mm_database_2025 = pd.read_csv(mm_database_csv, index_col=0)
mm_database_2025.index.name = "TEAM"  # rename for clarity

# ---------------------------------------------------------------------------
# 3) SELECT RELEVANT COLUMNS
#    We'll gather the columns you want for the main stats, plus CONFERENCE
#    and TM_KP for the treemap. Adjust as needed if you have different names.
# ---------------------------------------------------------------------------
core_cols = [
    "KP_Rank",       # float
    "WIN_25",        # float
    "LOSS_25",       # float
    "WIN% ALL GM",   # float
    "WIN% CLOSE GM", # float
    "KP_AdjEM",      # float
    "KP_SOS_AdjEM",  # float
    "OFF EFF",       # float
    "DEF EFF",       # float
    "AVG MARGIN",    # float
    "PTS/GM",        # float
    "OPP PTS/GM",    # float
    "eFG%",          # float
    "OPP eFG%",      # float
    "TS%",           # float
    "OPP TS%",       # float
    "AST/TO%",       # float
    "STOCKS/GM",     # float
    "STOCKS-TOV/GM", # float
]

# Additional columns we need for the treemap
extra_cols_for_treemap = ["CONFERENCE", "TM_KP"]

# Combine sets and only keep those that actually exist in the CSV
all_desired_cols = core_cols + extra_cols_for_treemap
actual_cols = [c for c in all_desired_cols if c in mm_database_2025.columns]

# Subset the master DataFrame
df_main = mm_database_2025[actual_cols].copy()

# ---------------------------------------------------------------------------
# 4) OPTIONAL: LOGO LOADING (CHECK FILE FIRST)
#    If you have an images folder with NCAA_logo1.png locally.
# ---------------------------------------------------------------------------
logo_path = "images/NCAA_logo1.png"
NCAA_logo = None
if os.path.exists(logo_path):
    NCAA_logo = Image.open(logo_path)
# else:
#     # Optionally skip or load from URL

# ---------------------------------------------------------------------------
# 5) STREAMLIT PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NCAA BASKETBALL -- MARCH MADNESS 2025",
    layout="wide",
    initial_sidebar_state="auto"
)

# Hide the Streamlit menu / watermark
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title("NCAA BASKETBALL -- MARCH MADNESS 2025")
st.write("*2025 MARCH MADNESS RESEARCH HUB*")
if NCAA_logo:
    st.image(NCAA_logo, width=180)

# ---------------------------------------------------------------------------
# 6) BUILD A TREEMAP (IF POSSIBLE)
#    We check if "CONFERENCE" and "TM_KP" exist for the path,
#    and "KP_AdjEM" for the color/values. If missing, we skip or fallback.
# ---------------------------------------------------------------------------
df_main_notnull = df_main.dropna(subset=["KP_AdjEM"], how="any")  # to ensure no issues
treemap = None
if "CONFERENCE" in df_main_notnull.columns and "TM_KP" in df_main_notnull.columns:
    # Create a path for the treemap
    path_list = ["CONFERENCE", "TM_KP"]
    # The measure/values is "KP_AdjEM"
    treemap_data = df_main_notnull.reset_index()  # to get 'TEAM' as a column for hover

    treemap = px.treemap(
        data_frame=treemap_data,
        path=path_list,                   # must be columns in the DF
        values="KP_AdjEM",                # must be numeric
        color="KP_AdjEM",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        hover_data=["TEAM", "KP_Rank"],   # optional
        template="plotly_dark",
        title="2025 KenPom AdjEM by Conference"
    )
    # Some style updates
    treemap.update_layout(margin=dict(l=10, r=10, t=50, b=10))
else:
    st.warning("Treemap not displayed: missing 'CONFERENCE', 'TM_KP', or 'KP_AdjEM' in the data.")

# ---------------------------------------------------------------------------
# 7) TABS LAYOUT
# ---------------------------------------------------------------------------
tab_home, tab_eda, tab_regions, tab_future1, tab_future2 = st.tabs([
    "Home", "EDA & Plots", "Regional Heatmaps", "TBD", "TBD"
])

# ---------------------------------------------------------------------------
# 7A) HOME TAB
# ---------------------------------------------------------------------------
with tab_home:
    st.subheader("Overview / Treemap")
    if treemap is not None:
        st.plotly_chart(treemap, use_container_width=True)
    else:
        st.write("Treemap not available.")

    st.write("Use the tabs above to explore additional data, EDA, region placeholders, etc.")

# ---------------------------------------------------------------------------
# 7B) EDA & PLOTS TAB
# ---------------------------------------------------------------------------
with tab_eda:
    st.header("Basic Exploratory Data Analysis")
    numeric_cols = [
        "KP_Rank", "WIN_25", "LOSS_25", "KP_AdjEM", "KP_SOS_AdjEM",
        "OFF EFF", "DEF EFF", "AVG MARGIN", "PTS/GM", "OPP PTS/GM"
    ]
    numeric_cols = [c for c in numeric_cols if c in df_main.columns]

    # Example 1: Distribution of KP_AdjEM
    if "KP_AdjEM" in df_main.columns:
        st.subheader("Histogram of KenPom AdjEM")
        fig_hist = px.histogram(
            df_main,
            x="KP_AdjEM",
            nbins=20,
            title="Distribution of KP_AdjEM (All Teams)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Example 2: Correlation Heatmap
    if len(numeric_cols) >= 2:
        st.subheader("Correlation Heatmap (Selected Metrics)")
        df_for_corr = df_main[numeric_cols].dropna()
        corr_mat = df_for_corr.corr()
        fig_corr = px.imshow(
            corr_mat,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Example 3: Group by CONFERENCE -> Avg KP_AdjEM
    if "CONFERENCE" in mm_database_2025.columns and "KP_AdjEM" in mm_database_2025.columns:
        st.subheader("Avg KenPom AdjEM by Conference")
        # group & mean
        conf_group = mm_database_2025.groupby("CONFERENCE")["KP_AdjEM"].mean(numeric_only=True)
        conf_group = conf_group.dropna().sort_values(ascending=False)
        fig_conf = px.bar(
            conf_group,
            x=conf_group.index,
            y=conf_group.values,
            title="Average KP_AdjEM by Conference",
            labels={"x": "Conference", "y": "KP_AdjEM"}
        )
        st.plotly_chart(fig_conf, use_container_width=True)

# ---------------------------------------------------------------------------
# 7C) REGIONAL HEATMAPS TAB
#     Currently placeholder, since seeds not set. Demonstrates pivot & styling.
# ---------------------------------------------------------------------------
with tab_regions:
    st.header("Regional Heatmaps (Placeholder)")
    st.write("Will be updated with actual seeds/regions once available.")

    # Create a row for 'TOURNEY AVG' if desired
    df_heat = df_main.copy()
    df_heat.loc["TOURNEY AVG"] = df_heat.mean(numeric_only=True)
    df_heat_T = df_heat.T  # Transpose

    # Example: define a fake list of 'East region' teams, if present
    east_teams_2025 = [
        "Alabama", "Houston", "Duke", "TOURNEY AVG"
    ]
    # Filter columns to those teams that exist
    east_teams_found = [tm for tm in east_teams_2025 if tm in df_heat_T.columns]
    East_region_2025 = df_heat_T[east_teams_found]

    if East_region_2025.empty:
        st.info("No 'East region' placeholder teams found in the dataset.")
    else:
        st.subheader("EAST REGION - DEMO")
        # Simple styling via .style
        styler_dict = {
            "KP_Rank": "Spectral_r",
            "WIN_25": "YlGn",
            "LOSS_25": "YlOrRd",
            "KP_AdjEM": "Spectral",
            "KP_SOS_AdjEM": "Spectral",
            "OFF EFF": "GnBu",
            "DEF EFF": "OrRd",
            "AVG MARGIN": "RdBu",
        }
        east_styler = East_region_2025.style
        # We color each row label with a colormap:
        for row_label, cmap in styler_dict.items():
            if row_label in East_region_2025.index:
                east_styler = east_styler.background_gradient(
                    cmap=cmap,
                    subset=pd.IndexSlice[row_label, :],
                    axis=1
                )
        st.markdown(east_styler.to_html(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 7D) FUTURE TABS
# ---------------------------------------------------------------------------
with tab_future1:
    st.subheader("TBD Tab 1")
    st.write("Placeholder for future bracket or advanced analytics.")
    if NCAA_logo:
        st.image(NCAA_logo, width=140)

with tab_future2:
    st.subheader("TBD Tab 2")
    st.write("Placeholder for additional expansions or data merges.")
    if NCAA_logo:
        st.image(NCAA_logo, width=140)

# ---------------------------------------------------------------------------
# 8) FOOTER / STOP
# ---------------------------------------------------------------------------
st.markdown("---")
github_link = "[GitHub: 2025 Repo](https://github.com/nehat312/march-madness-2025)"
kenpom_link = "[KENPOM](https://kenpom.com/)"
st.write(github_link + " | " + kenpom_link)

st.stop()
