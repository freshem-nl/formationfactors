"""
Script to perform statistical analysis of the formation factor and surface conductivity, and to check for differences between lithoclasses, stratigraphy and stratlithoclasses.

project: FRESHEM (11210255-005)
author: Romee van Dam (Deltares)
date: 15-04-26
"""

#%% imports

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kruskal
from pathlib import Path
import os
import numpy as np
import scikit_posthocs as sp
import seaborn as sns

#%% paths and parameters
path_labresults = Path("data/3-input/lab_results")
fn_labresults = path_labresults / "20260304_tbl20_WPchloride_FFdata.xlsx"
fn_labresults_inc_grainsize = path_labresults / "20260126_tbl05_Measurementdata_Full.xlsx"
path_results = Path("data/4-output/ff_ecs_uncertainty")

os.chdir(r"c:\Users\dam_re\OneDrive - Stichting Deltares\Documents\Projecten\FRESHEM\scripts\formationfactors")
#%% create dataframe with ff and ECs

# read data
df_all = pd.read_excel(fn_labresults, 
    keep_default_na=False,
    na_values=["", " ", "NULL", "NaN"] # otherwise the formation of Naaldwijk (NA) is set as NaN
)
df = df_all.copy()

# short name for column names
ff_col = "SIP3_FormationFactor_F_3W_unitless"
surfcond_col = "SIP3_SurfCond_Sigmas_3W_S/m"

litho_col = "LITHOKLASSE_CD"
strat_col = "Stratigrafie"
stratlitho_col = "StratLithoklasse"


# clean up data

# get lithoclass from stratlithoclass if missing
if len(df.loc[~df[litho_col].notnull()])>0:
    print("Warning: some samples have missing lithoclass. The value will be taken from stratlithoclass if available.")
    for idx in df.loc[~df[litho_col].notnull()].index:
        df.loc[idx, litho_col] = df.loc[idx, stratlitho_col][-2:]

# drop rows with missing values in any of the relevant columns
if len(df.loc[~df[ff_col].notnull()])>0: #TODO: if SIP3 is missing, take SIP5
    print("Warning: some samples have missing formation factors, these are removed from the analysis.\nConsider using SIP5_formation_factor_F_3W_unitless if SIP3 is missing.")
df = df.loc[df[ff_col].notnull() & df[surfcond_col].notnull() & df[litho_col].notnull() & df[strat_col].notnull() & df[stratlitho_col].notnull()]

#%% definitions

def collect_group_size(df, group_col):
    """Return a dict with group sizes for each category in group_col."""
    #  group_size_df = df.groupby(group_col).size().reset_index(name="n")
    return df.groupby(group_col).size().to_dict()


def kruskal_per_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    min_group_size: int = 5
) -> dict:
    """
    Perform Kruskal-Wallis test for a variable grouped by categories with minimum group size.

    Returns:
        dict with H-statistic, p-value, n_groups, group_sizes
    """

    # Drop NA values
    data = df[[value_col, group_col]].dropna() 

    # Collect samples per category
    groups = []
    group_sizes = {}

    for group, sub in data.groupby(group_col):
        if len(sub) >= min_group_size:
            groups.append(sub[value_col].values)
            group_sizes[group] = len(sub)

    # Kruskal-Wallis requires >= 2 groups
    if len(groups) < 2:
        return {
            "H": np.nan,
            "p_value": np.nan,
            "n_groups": len(groups),
            "group_sizes": group_sizes
        }

    H, p = kruskal(*groups)

    return {
        "H": H,
        "p_value": p,
        "n_groups": len(groups),
        "group_sizes": group_sizes
    }

def dunn_matrix_refined(df, val_col, group_col, p_adjust="fdr_bh"):
    """Perform Dunn post-hoc test and convert p-value matrix to long format sorted by p-value."""
    
    # perform Dunn post-hoc test
    pval_matrix = sp.posthoc_dunn(df,val_col,group_col,p_adjust)

    # give meaningful names to the group columns in the long format
    if group_col == litho_col:
        group1 = "lithoclass1"
        group2 = "lithoclass2"
    elif group_col == strat_col:
        group1 = "stratigraphy1"
        group2 = "stratigraphy2"
    elif group_col == stratlitho_col:
        group1 = "stratlithoclass1"
        group2 = "stratlithoclass2"

    # convert matrix to long format (only unique pairs)
    rows = []
    for i in pval_matrix.index:
        for j in pval_matrix.columns:
            if i < j:  # avoid duplicates & self-comparisons
                rows.append({
                    group1: i,
                    group2: j,
                    "p_value": pval_matrix.loc[i, j]
                })

    # sort by p-value
    rows_df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)

    return rows_df


#%% normal distribution test 
for col in [ff_col, surfcond_col]:
    if col == ff_col:
        colname = "ff"
    else: "surf_cond"
    
    print(f"Testing normality for {colname}")
    x = df[col]
    x_log = np.log(x[x > 0])

    #shapiro-wilk test

    shapiro_raw = stats.shapiro(x)
    shapiro_log = stats.shapiro(x_log)

    print("Shapiro raw:", shapiro_raw)
    print("Shapiro log:", shapiro_log)

    # Q-Q plot

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    stats.probplot(x, plot=axes[0])
    axes[0].set_title("Q–Q plot (original scale)")

    stats.probplot(x_log, plot=axes[1])
    axes[1].set_title("Q–Q plot (log-transformed)")

    plt.tight_layout()
    plt.show()
    plt.savefig(path_results / f"q-q_plot_{colname}.png")

    # histogram

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(x, kde=True, ax=axes[0])
    axes[0].set_title("Histogram original scale")

    sns.histplot(x_log, kde=True, ax=axes[1])
    axes[1].set_title("Histogram log-transformed")

    plt.tight_layout()
    plt.show()
    plt.savefig(path_results / f"histogram_{colname}.png")



#%% Kruskal-Wallis test

results = []

# - FF -

# FF & lithoclass
results.append({
    "variable": "formation_factor",
    "grouping": "Lithoklasse",
    **kruskal_per_group(df, ff_col, litho_col)
})


# FF & stratigraphy 
results.append({
    "variable": "formation_factor",
    "grouping": "stratigrafie",
    **kruskal_per_group(df, ff_col, strat_col)
})

# FF & lithostraticlass
results.append({
    "variable": "formation_factor",
    "grouping": "StratLithoklasse",
    **kruskal_per_group(df, ff_col, stratlitho_col)
})

# - surface conductivity -

# surface conductivity & lithoclass
results.append({
    "variable": "surface_cond",
    "grouping": "Lithoklasse",
    **kruskal_per_group(df, surfcond_col, litho_col)
})

# surface conductivity & stratigraphy 
results.append({
    "variable": "surface_cond",
    "grouping": "stratigrafie",
    **kruskal_per_group(df, surfcond_col, strat_col)
})

# surface conductivity & lithostraticlass
results.append({
    "variable": "surface_cond",
    "grouping": "StratLithoklasse",
    **kruskal_per_group(df, surfcond_col, stratlitho_col)
})

# create 
results_df = pd.DataFrame(results)

print(results_df[[
    "variable",
    "grouping",
    "H",
    "p_value",
    "n_groups"
]])


#%% # filter for groups with >= 5 samples

group_size_litho = collect_group_size(df, litho_col)
group_size_strat = collect_group_size(df, strat_col)
group_size_stratlitho = collect_group_size(df, stratlitho_col)

valid_litho = {k for k, v in group_size_litho.items() if v >= 5}
valid_strat = {k for k, v in group_size_strat.items() if v >= 5}
valid_stratlitho = {k for k, v in group_size_stratlitho.items() if v >= 5}

df_litho_refined = df[df[litho_col].isin(valid_litho)].copy()
df_strat_refined = df[df[strat_col].isin(valid_strat)].copy()
df_stratlitho_refined = df[df[stratlitho_col].isin(valid_stratlitho)].copy()

#%% Dunn post-hoc test with Benjamini–Hochberg correction

# ff & litho
dunn_ff_litho = dunn_matrix_refined(df_litho_refined, val_col = ff_col, group_col=litho_col, p_adjust="fdr_bh")
dunn_ff_strat = dunn_matrix_refined(df_strat_refined, val_col = ff_col, group_col=strat_col, p_adjust="fdr_bh")
dunn_ff_stratlitho = dunn_matrix_refined(df_stratlitho_refined, val_col = ff_col, group_col=stratlitho_col, p_adjust="fdr_bh")
dunn_surfcond_litho = dunn_matrix_refined(df_litho_refined, val_col = surfcond_col, group_col=litho_col, p_adjust="fdr_bh")
dunn_surfcond_strat = dunn_matrix_refined(df_strat_refined, val_col = surfcond_col, group_col=strat_col, p_adjust="fdr_bh")
dunn_surfcond_stratlitho = dunn_matrix_refined(df_stratlitho_refined, val_col = surfcond_col, group_col=stratlitho_col, p_adjust="fdr_bh")

print(dunn_ff_litho.loc[dunn_ff_litho["p_value"] < 0.1])
print(dunn_ff_strat.loc[dunn_ff_strat["p_value"] < 0.1])
print(dunn_ff_stratlitho.loc[dunn_ff_stratlitho["p_value"] < 0.1])
print(dunn_surfcond_litho.loc[dunn_surfcond_litho["p_value"] < 0.1])
print(dunn_surfcond_strat.loc[dunn_surfcond_strat["p_value"] < 0.1])
print(dunn_surfcond_stratlitho.loc[dunn_surfcond_stratlitho["p_value"] < 0.1])

#%% save output
results_df.to_csv(path_results / "kruskal_results.csv", index=False)
dunn_ff_litho.to_csv(path_results / "dunn_ff_litho.csv", index=False)
dunn_ff_strat.to_csv(path_results / "dunn_ff_strat.csv", index=False)
dunn_ff_stratlitho.to_csv(path_results / "dunn_ff_stratlitho.csv", index=False)
dunn_surfcond_litho.to_csv(path_results / "dunn_surfcond_litho.csv", index=False)
dunn_surfcond_strat.to_csv(path_results / "dunn_surfcond_strat.csv", index=False)
dunn_surfcond_stratlitho.to_csv(path_results / "dunn_surfcond_stratlitho.csv", index=False) 

