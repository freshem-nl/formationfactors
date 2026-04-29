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
import numpy as np

#%% paths and parameters
# run from basedir, assuming script resides in subdir of src/
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

path_labresults = Path("data/3-input/lab_results")
fn_labresults = path_labresults / "20260304_tbl20_WPchloride_FFdata.xlsx"
fn_labresults_inc_grainsize = path_labresults / "20260126_tbl05_Measurementdata_Full.xlsx"
path_results = Path("data/4-output/ff_ecs_uncertainty/dunn_test_results")

alpha = 0.1

path_results.mkdir(exist_ok=True, parents=True)
#%% create dataframe with ff and ECs

# read data
df_all = pd.read_excel(fn_labresults, 
    keep_default_na=False,
    na_values=["", " ", "NULL", "NaN"] # otherwise the formation of Naaldwijk (NA) is set as NaN
)
df = df_all.loc[df_all["Type_name"]=="FF_Disturbed"].copy()

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

df["log10_FF"] = np.log10(df[ff_col])
df["log10_surfcond"] = np.log10(df[surfcond_col])
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
        "group_sizes": group_sizes,
        "strats": list(group_sizes.keys())
    }

def dunn_matrix_refined(df, val_col, group_col, p_adjust="fdr_bh"):
    """Perform Dunn post-hoc test and convert p-value matrix to long format sorted by p-value."""
    
    # perform Dunn post-hoc test
    pval_matrix = sp.posthoc_dunn(df,val_col,group_col,p_adjust)

    # give meaningful names to the group columns in the long format
    if group_col == litho_col:
        group1 = "lithoklasse1"
        group2 = "lithoklasse2"
    elif group_col == strat_col:
        group1 = "stratigrafie1"
        group2 = "stratigrafie2"
    elif group_col == stratlitho_col:
        group1 = "stratlithoklasse1"
        group2 = "stratlithoklasse2"

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

def _order_by_median(subdf, value_col, group_col):
    """ determine order for categories based on median value."""
    med = subdf.groupby(group_col)[value_col].median().sort_values()
    return med.index.tolist()

def plot_box_per_litho(df, value_col, value_label, filename_stub):
    """ create boxplots per lithoclass per stratigraphy """

    for litho in valid_litho:

        df_l = df[df[litho_col] == litho].copy()
        if df_l.empty:
            continue

        # filter stratigrafieën binnen deze litho met voldoende n
        counts = df_l[strat_col].value_counts()
        valid_stratlitho = counts[counts >= min_n_per_strat_in_litho].index
        df_l = df_l[df_l[strat_col].isin(valid_stratlitho)].copy()
        

        # als er na filteren te weinig groepen overblijven, skip
        if df_l[strat_col].nunique() < 2:
            continue

        # kies een volgorde (median-based)
        order = _order_by_median(df_l, value_col=value_col, group_col=strat_col)

        # figuur
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=df_l,
            x=strat_col,
            y=value_col,
            order=order,
            color="#5B8FF9",
            width=0.5,
            showfliers=True
        )

        if show_points:
            sns.stripplot(
                data=df_l,
                x=strat_col,
                y=value_col,
                order=order,
                color="0.35",
                size=4,
                jitter=0.08,
                alpha=0.8
            )

        ax.set_title(f"{litho} — {value_label} per stratigrafie")
        ax.set_xlabel("stratigrafie")
        ax.set_ylabel(value_label)

        if use_log_y:
            ax.set_yscale("log")

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        # opslaan
        out_png = path_figs / f"{filename_stub}_litho-{litho}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()


def calc_stratlitho_medians(
    df,
    stratlitho_combos,              # dataframe met kolommen: lithoklasse + strats (list)
    ff_col,
    surfcond_col,
    litho_col,
    strat_col,
    manual_groups=None,     # list of dicts (zie boven)
):
    manual_groups = manual_groups or []

    # groepeer rules per litho voor snelheid/overzicht
    rules_by_litho = {}
    for r in manual_groups:
        rules_by_litho.setdefault(r["lithoklasse"], []).append(r)

    out_rows = []

    for _, row in stratlitho_combos.iterrows():
        litho = row[litho_col] if litho_col in stratlitho_combos.columns else row["lithoklasse"]
        strats = row["strats"]

        # ommit empty/NaN entries
        if not isinstance(strats, (list, tuple)) or len(strats) == 0:
            continue

        # workingset: only strats that we would (initially) take individually
        remaining = list(strats)

        # 1) first apply manual merges (override)
        for rule in rules_by_litho.get(litho, []):
            wanted = rule["strats"]

            # take only the strats that are still "remaining"
            members = [s for s in wanted if s in remaining]

            # no merge -> continue
            if len(members) == 0:
                continue

            # calc median for this group of members
            mask = (df[litho_col] == litho) & (df[strat_col].isin(members))
            sub = df.loc[mask, [ff_col, surfcond_col]].dropna()
            sub_log = df.loc[mask, ["log10_FF", "log10_surfcond"]].dropna()

            if sub.empty:
                continue

            out_rows.append({
                litho_col: litho,
                "strat_group": rule.get("group", "+".join(members)),
                "members": members,
                "n": len(sub),
                "median_ff": sub[ff_col].median(),
                "median_surfcond": sub[surfcond_col].median(),
                "median_log_ff": sub_log["log10_FF"].median(),
                "median_log_surfcond": sub_log["log10_surfcond"].median(),
                "median_log_ff_based_on_log_transformed_data": 10 ** sub_log["log10_FF"].median(),
                "median_surfcond_based_on_log_transformed_data": 10 ** sub_log["log10_surfcond"].median(),
                "is_manual": True,
            })

            # remove these strats from the workingset
            remaining = [s for s in remaining if s not in members]

        # 2) if there are strats left that were not manually merged, calculate individual medians for those
        for strat in remaining:
            mask = (df[litho_col] == litho) & (df[strat_col] == strat)
            sub = df.loc[mask, [ff_col, surfcond_col]].dropna()
            sub_log = df.loc[mask, ["log10_FF", "log10_surfcond"]].dropna()
            if sub.empty:
                continue

            out_rows.append({
                litho_col: litho,
                "strat_group": strat,
                "members": [strat],
                "n": len(sub),
                "median_ff": sub[ff_col].median(),
                "median_surfcond": sub[surfcond_col].median(),
                "median_log_ff": sub_log["log10_FF"].median(),
                "median_log_surfcond": sub_log["log10_surfcond"].median(),
                "median_log_ff_to_normal_value": 10 ** sub_log["log10_FF"].median(),
                "median_log_surfcond_to_normal_value": 10 ** sub_log["log10_surfcond"].median(),
                "is_manual": False,
            })

    result = pd.DataFrame(out_rows)

    # sort: litho, then non-manual first (optional), then name
    if not result.empty:
        result = result.sort_values([litho_col, "is_manual", "strat_group"], ascending=[True, True, True])

    return result



#%% normal distribution test 
for col in [ff_col, surfcond_col]:
    if col == ff_col:
        colname = "ff"
    else: 
        colname ="surf_cond"
    
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
    plt.savefig(path_results / f"q-q_plot_{colname}.png")
    plt.show()
    plt.close()

    # histogram

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(x, kde=True, ax=axes[0])
    axes[0].set_title("Histogram original scale")

    sns.histplot(x_log, kde=True, ax=axes[1])
    axes[1].set_title("Histogram log-transformed")

    plt.tight_layout()
    plt.savefig(path_results / f"histogram_{colname}.png")
    plt.show()
    plt.close()




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

# FF & stratlithoclass
results.append({
    "variable": "formation_factor",
    "grouping": "stratlithoklasse",
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

# surface conductivity & stratlithoclass
results.append({
    "variable": "surface_cond",
    "grouping": "stratlithoklasse",
    **kruskal_per_group(df, surfcond_col, stratlitho_col)
})

# create df
results_df = pd.DataFrame(results)

print(results_df[[
    "variable",
    "grouping",
    "H",
    "p_value",
    "n_groups"
]])

results_df.to_csv(path_results / "kruskal_results.csv", index=False)
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

#%% kruskal-wallis test for stratigrahpy within each lithoclass
results_litho_strat = []
for litho in valid_litho:
    print(f"Testing stratigraphy within lithoclass {litho}")
    df_litho = df[df[litho_col] == litho]
    results_litho_strat.append({
        "variable": "formation_factor",
        "grouping": f"stratigrafie binnen lithoklasse {litho}",
        "lithoklasse": litho,
        **kruskal_per_group(df_litho, ff_col, strat_col)
    })

for litho in valid_litho:
    df_litho = df[df[litho_col] == litho]
    results_litho_strat.append({
        "variable": "surface_cond",
        "grouping": f"stratigrafie binnen lithoklasse {litho}",
        "lithoklasse": litho,
        **kruskal_per_group(df_litho, surfcond_col, strat_col)
    })

# create df
results_litho_strat_df = pd.DataFrame(results_litho_strat)
results_litho_strat_df.to_csv(path_results / "kruskal_litho_strat_results.csv", index=False)


#%% Dunn post-hoc test with Benjamini–Hochberg correction

dunn_ff_litho = dunn_matrix_refined(df_litho_refined, val_col = ff_col, group_col=litho_col, p_adjust="fdr_bh")
dunn_ff_strat = dunn_matrix_refined(df_strat_refined, val_col = ff_col, group_col=strat_col, p_adjust="fdr_bh")
dunn_surfcond_litho = dunn_matrix_refined(df_litho_refined, val_col = surfcond_col, group_col=litho_col, p_adjust="fdr_bh")
dunn_surfcond_strat = dunn_matrix_refined(df_strat_refined, val_col = surfcond_col, group_col=strat_col, p_adjust="fdr_bh")

dunn_ff_litho.to_csv(path_results / "dunn_ff_litho.csv", index=False)
dunn_ff_strat.to_csv(path_results / "dunn_ff_strat.csv", index=False)
dunn_surfcond_litho.to_csv(path_results / "dunn_surfcond_litho.csv", index=False)
dunn_surfcond_strat.to_csv(path_results / "dunn_surfcond_strat.csv", index=False)

#%% Dunn post-hoc test with Benjamini–Hochberg correction for stratigraphy within each lithoclass
dunn_litho_strat_all = []

for variable in ["formation_factor", "surface_cond"]:
    print(f"Performing Dunn post-hoc test for stratigraphy within lithoclass for {variable}")
    
    if variable == "formation_factor":
        var_short = "ff"
        val_col = ff_col 
    elif variable == "surface_cond":
        var_short = "surfcond"
        val_col = surfcond_col

    dunn_litho_strat_var = []
    for litho in valid_litho:
        print(var_short, litho)
        
        # select stratigraphy within lithoclass with >=5 samples
        sel = (results_litho_strat_df["lithoklasse"] == litho) & (results_litho_strat_df["variable"] == variable)
        # no entry -> continue to next lithoclass
        if not sel.any():
            continue  
        data = results_litho_strat_df.loc[sel]["group_sizes"].values[0]
        strat = list(data.keys())

        # perform dunn test if there are at least 2 stratigraphic groups with >=5 samples
        if len(strat)>1:
            df_litho = df[(df[litho_col] == litho) & (df[strat_col].isin(strat))]
            results_dunn = pd.DataFrame(dunn_matrix_refined(df_litho, val_col = val_col, group_col=strat_col, p_adjust="fdr_bh"))
            results_dunn.insert(loc=0, column="variable", value=variable)
            results_dunn.insert(loc=1, column="lithoclass", value=litho)
            results_dunn.to_csv(path_results / f"dunn_{var_short}_strat_within_{litho}.csv", index=False)
            # create multiple tables            
            dunn_litho_strat_var.append(results_dunn)
            dunn_litho_strat_all.append(results_dunn)

        # combine all lithoclasses for this variable into one table
        if dunn_litho_strat_var:
            dunn_litho_strat_var_df = pd.concat(dunn_litho_strat_var, ignore_index=True)
            # only significant results
            dunn_litho_strat_var_df_sig = dunn_litho_strat_var_df.loc[dunn_litho_strat_var_df["p_value"] < alpha]
            dunn_litho_strat_var_df_sig.sort_values("p_value", inplace=True)
            # save
            dunn_litho_strat_var_df.to_csv(path_results / f"dunn_{var_short}_strat_all_litho.csv", index=False)
            dunn_litho_strat_var_df_sig.to_csv(path_results / f"dunn_{var_short}_strat_litho_significant.csv", index=False)

    # combine for both variables into one table
    if dunn_litho_strat_all:
        dunn_litho_strat_all_df = pd.concat(dunn_litho_strat_all, ignore_index=True)
        dunn_litho_strat_all_df.to_csv(path_results / "dunn_strat_all_litho.csv", index=False)

#%% Boxplots per lithoklasse met stratigrafie op de x-as (los voor FF en surfcond)


# outputmap voor figuren
path_figs = path_results / "boxplots_litho_strat"
path_figs.mkdir(exist_ok=True, parents=True)

# instellingen
min_n_per_strat_in_litho = 5   # pas aan als je wilt (bijv. 3 of 5)
show_points = True             # puntjes (stripplot) aan/uit
use_log_y = False              # optioneel: log-scale y-as


# maak figuren voor FF en surface conductivity
plot_box_per_litho(df, value_col=ff_col,       value_label="formation factor (FF)", filename_stub="box_ff_strat")
plot_box_per_litho(df, value_col=surfcond_col, value_label="surface conductivity σₛ (S/m)", filename_stub="box_surfcond_strat")

print(f"Boxplots opgeslagen in: {path_figs}")



#%% calculate median values for 1) lithoclass and 2) stratlitho combinations, with option to manually merge certain stratigraphies within a lithoclass


# median lithoclass
median_litho_ff = df.groupby(litho_col)["log10_FF"].median()
median_litho_ff  = (10**median_litho_ff).reset_index(name="median_ff_based_on_log_transformed_data")
median_litho_surfcond = df.groupby(litho_col)["log10_surfcond"].median()
median_litho_surfcond = (10**median_litho_surfcond).reset_index(name="median_surfcond_based_on_log_transformed_data")
median_litho = pd.merge(median_litho_ff, median_litho_surfcond, on=litho_col)
median_litho.to_csv(path_results / "median_ff_litho.csv", index=False)


# -- median stratigraphy ---

manual_groups = [
    {"lithoklasse": "zg", "strats": ["AP", "PZ-WG"], "group": "AP+PZ-WG"},
    {"lithoklasse": "kz", "strats": ["BX", "DRGI"], "group": "BX+DRGI"},
    {"lithoklasse": "kz", "strats": ["AAOM", "NAWA"], "group": "AAOM+NAWA"},
    {"lithoklasse": "zm", "strats": ["AAOM", "URVE"], "group": "AAOM+URVE"},
    {"lithoklasse": "zm", "strats": ["NAWA", "NAWO", "NAZA", "OO"], "group": "NAWA+NAWO+NAZA+OO"},
    {"lithoklasse": "zm", "strats": ["BX", "BXWI"], "group": "BX+BXWI"},
]

# create stratlitho_combos
stratlitho_combos = results_litho_strat_df.loc[results_litho_strat_df["variable"]== "formation_factor"][["lithoklasse", "strats"]]

medians_stratlitho = calc_stratlitho_medians(
    df=df,
    stratlitho_combos=stratlitho_combos,#.rename(columns={"lithoklasse": litho_col}) if "lithoklasse" in stratlitho_combos.columns else stratlitho_combos,
    ff_col=ff_col,
    surfcond_col=surfcond_col,
    litho_col=litho_col,
    strat_col=strat_col,
    manual_groups=manual_groups
).reset_index(drop=True)

medians_stratlitho_no_groups = calc_stratlitho_medians(
    df=df,
    stratlitho_combos=stratlitho_combos,
    ff_col=ff_col,
    surfcond_col=surfcond_col,
    litho_col=litho_col,
    strat_col=strat_col,
    manual_groups=None
).reset_index(drop=True)

# save
medians_stratlitho[["LITHOKLASSE_CD", "strat_group", "median_log_ff_to_normal_value", "median_log_surfcond_to_normal_value"]].to_csv(path_results / "median_stratlitho_manual_groups_short.csv", index=False)
medians_stratlitho_no_groups[["LITHOKLASSE_CD", "strat_group", "median_log_ff_to_normal_value", "median_log_surfcond_to_normal_value"]].to_csv(path_results / "median_stratlitho_no_groups_short.csv", index=False)
medians_stratlitho.to_csv(path_results / "median_stratlitho_manual_groups.csv", index=False)
medians_stratlitho_no_groups.to_csv(path_results / "median_stratlitho_no_groups.csv", index=False)
