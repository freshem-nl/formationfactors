"""
Script to perform statistical analysis of the formation factor and surface conductivity, and to check for differences between facies.

output of this script:
1) Normal distribution test
2) kruskal-wallis test for FF and ECs on facies
3) kruskal-wallis test for FF and ECs on facies within each lithoclass
4) Dunn post-hoc test with Bejamini-Hochberg correction for FF and ECs on facies
5) Dunn post-hoc test with Bejamini-Hochberg correction for FF and ECs on facies within lithoclass
6) Boxplots per lithoclass per facies for FF and ECs
7) Median for FF and ECs per lithoclass and for facies within lithoclass based on post-hoc test grouping


project: FRESHEM (11210255-005)
author: Romee van Dam (Deltares)
date: 11-06-26
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
path_results = Path("data/4-output/ff_ecs_uncertainty/dunn_test_results_facies")

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
facies_col = "facies"


# clean up data

# get lithoclass from stratlithoclass if missing
if len(df.loc[~df[litho_col].notnull()])>0:
    print("Warning: some samples have missing lithoclass. The value will be taken from stratlithoclass if available.")
    for idx in df.loc[~df[litho_col].notnull()].index:
        df.loc[idx, litho_col] = df.loc[idx, stratlitho_col][-2:]

# drop rows with missing values in any of the relevant columns
if len(df.loc[~df[ff_col].notnull()])>0: #TODO: if SIP3 is missing, take SIP5?
    print("Warning: some samples have missing formation factors, these are removed from the analysis.\nConsider using SIP5_formation_factor_F_3W_unitless if SIP3 is missing.")
df = df.loc[df[ff_col].notnull() & df[surfcond_col].notnull() & df[litho_col].notnull() & df[strat_col].notnull()].copy()

df["log10_FF"] = np.log10(df[ff_col])
df["log10_surfcond"] = np.log10(df[surfcond_col])

# do not take AAOM (anthropogenic) for analysis 
df = df.loc[df["Stratigrafie"]!='AAOM'].copy()

# remove  MG and WG suffixes
df[strat_col] = df[strat_col].str.replace("-(MG|WG)", "", regex=True)
df[stratlitho_col] = df[stratlitho_col].str.replace("-(MG|WG)", "", regex=True)



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
        "group_members": list(group_sizes.keys())
    }

def dunn_matrix_refined(df, val_col, group_col, p_adjust="fdr_bh"):
    """Perform Dunn post-hoc test and convert p-value matrix to long format sorted by p-value."""
    
    # perform Dunn post-hoc test
    pval_matrix = sp.posthoc_dunn(df,val_col,group_col,p_adjust)

    # give meaningful names to the group columns in the long format
    if group_col == litho_col:
        group1 = "lithoklasse1"
        group2 = "lithoklasse2"
    elif group_col == facies_col:
        group1 = "facies1"
        group2 = "facies2"
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
    """ create boxplots per lithoclass per facies """

    for litho in valid_litho:

        df_l = df[df[litho_col] == litho].copy()
        if df_l.empty:
            continue

        # filter faciesbinnen deze litho met voldoende n
        counts = df_l[facies_col].value_counts()
        valid_facieslitho = counts[counts >= min_n_per_facies_in_litho].index
        df_l = df_l[df_l[facies_col].isin(valid_facieslitho)].copy()
        

        # als er na filteren te weinig groepen overblijven, skip
        if df_l[facies_col].nunique() < 2:
            continue

        # kies een volgorde (median-based)
        order = _order_by_median(df_l, value_col=value_col, group_col=facies_col)

        # figuur
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=df_l,
            x=facies_col,
            y=value_col,
            order=order,
            color="#5B8FF9",
            width=0.5,
            showfliers=True
        )

        if show_points:
            sns.stripplot(
                data=df_l,
                x=facies_col,
                y=value_col,
                order=order,
                color="0.35",
                size=4,
                jitter=0.08,
                alpha=0.8
            )

        ax.set_title(f"{litho} — {value_label} per facies")
        ax.set_xlabel("facies")
        ax.set_ylabel(value_label)

        if use_log_y:
            ax.set_yscale("log")

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        # opslaan
        out_png = path_figs / f"{filename_stub}_litho_facies-{litho}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()


def plot_box_per_facies(df, value_col, value_label, filename_stub):
    """ create boxplots per lithoclass per facies """

    for facies in valid_facies:

        df_l = df[df[facies_col] == facies].copy()
        if df_l.empty:
            continue

        # filter litho binnen deze facies met voldoende n
        counts = df_l[litho_col].value_counts()
        valid_lithofacies = counts[counts >= min_n_per_facies_in_litho].index
        df_l = df_l[df_l[litho_col].isin(valid_lithofacies)].copy()
        

        # als er na filteren te weinig groepen overblijven, skip
        if df_l[litho_col].nunique() < 2:
            continue

        # kies een volgorde (median-based)
        order = _order_by_median(df_l, value_col=value_col, group_col=litho_col)

        # figuur
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=df_l,
            x=litho_col,
            y=value_col,
            order=order,
            color="#5B8FF9",
            width=0.5,
            showfliers=True
        )

        if show_points:
            sns.stripplot(
                data=df_l,
                x=litho_col,
                y=value_col,
                order=order,
                color="0.35",
                size=4,
                jitter=0.08,
                alpha=0.8
            )

        ax.set_title(f"{facies} — {value_label} per litho")
        ax.set_xlabel("litho")
        ax.set_ylabel(value_label)

        if use_log_y:
            ax.set_yscale("log")

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        # opslaan
        out_png = path_figs / f"{filename_stub}_facies_litho-{facies}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()


def calc_lithofacies_medians(
    df,
    lithofacies_combos,              # dataframe met kolommen: lithoklasse + facies (list)
    ff_col,
    surfcond_col,
    litho_col,
    facies_col,
    manual_groups=None,     # list of dicts (zie boven)
):
    manual_groups = manual_groups or []

    # groepeer rules per litho voor snelheid/overzicht
    rules_by_litho = {}
    for r in manual_groups:
        rules_by_litho.setdefault(r["lithoklasse"], []).append(r)

    out_rows = []
    out_rows_std = []

    for _, row in lithofacies_combos.iterrows():
        litho = row[litho_col] if litho_col in lithofacies_combos.columns else row["lithoklasse"]
        facies = row["group_members"]

        # ommit empty/NaN entries
        if not isinstance(facies, (list, tuple)) or len(facies) == 0:
            continue

        # workingset: only facies that we would (initially) take individually
        remaining = list(facies)

        # 1) first apply manual merges (override)
        for rule in rules_by_litho.get(litho, []):
            wanted = rule["facies"]

            # take only the facies that are still "remaining"
            members = [s for s in wanted if s in remaining]

            # no merge -> continue
            if len(members) == 0:
                continue

            # calc median for this group of members
            mask = (df[litho_col] == litho) & (df[facies_col].isin(members))
            sub = df.loc[mask, [ff_col, surfcond_col]].dropna()
            sub_log = df.loc[mask, ["log10_FF", "log10_surfcond"]].dropna()

            if sub.empty:
                continue


            out_rows.append({
                litho_col: litho,
                "facies_group": rule.get("group", "+".join(members)),
                "members": members,
                "n": len(sub),
                "median_ff": sub[ff_col].median(),
                "median_surfcond": sub[surfcond_col].median(),
                "median_log_ff": sub_log["log10_FF"].median(),
                "median_log_surfcond": sub_log["log10_surfcond"].median(),
                "median_log_ff_to_normal_space": 10 ** sub_log["log10_FF"].median(),
                "median_log_surfcond_to_normal_space": 10 ** sub_log["log10_surfcond"].median(),
                "std_ff_normal_space": sub[ff_col].std(),
                "std_surfcond_normal_space": sub[surfcond_col].std(),
                "iqr_ff_normal_space": sub[ff_col].quantile(0.75) - sub[ff_col].quantile(0.25),
                "iqr_surfcond_normal_space": sub[surfcond_col].quantile(0.75) - sub[surfcond_col].quantile(0.25),
                "manual_grouping": True,
            })

            # remove these facies from the workingset
            remaining = [s for s in remaining if s not in members]

        # 2) if there are facies left that were not manually merged, calculate individual medians for those
        for facies in remaining:
            mask = (df[litho_col] == litho) & (df[facies_col] == facies)
            sub = df.loc[mask, [ff_col, surfcond_col]].dropna()
            sub_log = df.loc[mask, ["log10_FF", "log10_surfcond"]].dropna()
            if sub.empty:
                continue

            out_rows.append({
                litho_col: litho,
                "facies_group": facies,
                "members": [facies],
                "n": len(sub),
                "median_ff": sub[ff_col].median(),
                "median_surfcond": sub[surfcond_col].median(),
                "median_log_ff": sub_log["log10_FF"].median(),
                "median_log_surfcond": sub_log["log10_surfcond"].median(),
                "median_log_ff_to_normal_space": 10 ** sub_log["log10_FF"].median(),
                "median_log_surfcond_to_normal_space": 10 ** sub_log["log10_surfcond"].median(),
                "std_ff_normal_space": sub[ff_col].std(),
                "std_surfcond_normal_space": sub[surfcond_col].std(),
                "iqr_ff_normal_space": sub[ff_col].quantile(0.75) - sub[ff_col].quantile(0.25),
                "iqr_surfcond_normal_space": sub[surfcond_col].quantile(0.75) - sub[surfcond_col].quantile(0.25),
                "manual_grouping": False,
            })

    result = pd.DataFrame(out_rows)

    # sort: litho, then non-manual first (optional), then name
    if not result.empty:
        result = result.sort_values([litho_col, "manual_grouping", "facies_group"], ascending=[True, True, True])

    return result


def calc_facieslitho_medians(
    df,
    facieslitho_combos,              # dataframe met kolommen: lithoklasse + facies (list)
    ff_col,
    surfcond_col,
    litho_col,
    facies_col,
    manual_groups=None,     # list of dicts (zie boven)
):
    manual_groups = manual_groups or []

    # groepeer rules per litho voor snelheid/overzicht
    rules_by_facies = {}
    for r in manual_groups:
        rules_by_facies.setdefault(r["facies"], []).append(r)

    out_rows = []
    out_rows_std = []

    for _, row in facieslitho_combos.iterrows():
        facies = row[facies_col] if facies_col in facieslitho_combos.columns else row["klasse"]
        litho = row["group_members"]

        # ommit empty/NaN entries
        if not isinstance(litho, (list, tuple)) or len(litho) == 0:
            continue

        # workingset: only litho that we would (initially) take individually
        remaining = list(litho)

        # 1) first apply manual merges (override)
        for rule in rules_by_facies.get(facies, []):
            wanted = rule["lithoklasse"]

            # take only the litho that are still "remaining"
            members = [s for s in wanted if s in remaining]

            # no merge -> continue
            if len(members) == 0:
                continue

            # calc median for this group of members
            mask = (df[facies_col] == facies) & (df[litho_col].isin(members))
            sub = df.loc[mask, [ff_col, surfcond_col]].dropna()
            sub_log = df.loc[mask, ["log10_FF", "log10_surfcond"]].dropna()

            if sub.empty:
                continue


            out_rows.append({
                "facies": facies,
                "litho_group": rule.get("group", "+".join(members)),
                "members": members,
                "n": len(sub),
                "median_ff": sub[ff_col].median(),
                "median_surfcond": sub[surfcond_col].median(),
                "median_log_ff": sub_log["log10_FF"].median(),
                "median_log_surfcond": sub_log["log10_surfcond"].median(),
                "median_log_ff_to_normal_space": 10 ** sub_log["log10_FF"].median(),
                "median_log_surfcond_to_normal_space": 10 ** sub_log["log10_surfcond"].median(),
                "std_ff_normal_space": sub[ff_col].std(),
                "std_surfcond_normal_space": sub[surfcond_col].std(),
                "iqr_ff_normal_space": sub[ff_col].quantile(0.75) - sub[ff_col].quantile(0.25),
                "iqr_surfcond_normal_space": sub[surfcond_col].quantile(0.75) - sub[surfcond_col].quantile(0.25),
                "manual_grouping": True,
            })

            # remove these facies from the workingset
            remaining = [s for s in remaining if s not in members]

        # 2) if there are litho left that were not manually merged, calculate individual medians for those
        for litho in remaining:
            mask = (df[facies_col] == facies) & (df[litho_col] == litho)
            sub = df.loc[mask, [ff_col, surfcond_col]].dropna()
            sub_log = df.loc[mask, ["log10_FF", "log10_surfcond"]].dropna()
            if sub.empty:
                continue

            out_rows.append({
                "facies": facies,
                "litho_group": litho,
                "members": [litho],
                "n": len(sub),
                "median_ff": sub[ff_col].median(),
                "median_surfcond": sub[surfcond_col].median(),
                "median_log_ff": sub_log["log10_FF"].median(),
                "median_log_surfcond": sub_log["log10_surfcond"].median(),
                "median_log_ff_to_normal_space": 10 ** sub_log["log10_FF"].median(),
                "median_log_surfcond_to_normal_space": 10 ** sub_log["log10_surfcond"].median(),
                "std_ff_normal_space": sub[ff_col].std(),
                "std_surfcond_normal_space": sub[surfcond_col].std(),
                "iqr_ff_normal_space": sub[ff_col].quantile(0.75) - sub[ff_col].quantile(0.25),
                "iqr_surfcond_normal_space": sub[surfcond_col].quantile(0.75) - sub[surfcond_col].quantile(0.25),
                "manual_grouping": False,
            })

    result = pd.DataFrame(out_rows)

    # sort: facies, then non-manual first (optional), then name
    if not result.empty:
        result = result.sort_values([facies_col, "manual_grouping", "litho_group"], ascending=[True, True, True])

    return result

#%% prepare facies groups

facies_list = ['marien' , 'fluviatiel', 'glaciaal', 'eolisch', 'organisch', 'rest']

marien_codes = ['NAWA', 'NAWO', 'NAZA', 'NAWOBE', 'EE', 'OO', 'MS', 'OOSP', 'BR', 'WAWO' ]

fluviatiel_codes = ['URTY', 'URVE', 'AP', 'BXSI', 'UR', 'PZ', 'EC', 'ST', 'WA', 'KK', 'KW' ]

glaciaal_codes = ['DRGI', 'DRGIGA', 'PENI', 'PE', 'DRUI'] 

eolisch_codes = ['BX', 'DN', 'BXWI', 'BXKO', 'NASC' ] 

organisch_codes = ['NIHO', 'NIBA', 'NI']

rest_codes = ['AAOM'] #TODO: 'NA'?


facies_map = {}
for code in marien_codes:
    facies_map[code] = "marien"
for code in fluviatiel_codes:
    facies_map[code] = "fluviatiel"
for code in glaciaal_codes:
    facies_map[code] = "glaciaal"
for code in eolisch_codes:
    facies_map[code] = "eolisch"
for code in organisch_codes:
    facies_map[code] = "organisch"
for code in rest_codes:
    facies_map[code] = "rest"


def normalize_strat_code(code):
    """Normalize stratigraphy code before lookup."""
    if pd.isna(code):
        return np.nan
    return str(code).strip().upper()


def assign_facies(strat_code):
    """Map stratigraphy code to facies."""
    if pd.isna(strat_code):
        return np.nan
    return facies_map.get(normalize_strat_code(strat_code), np.nan)

df[facies_col] = df[strat_col].apply(assign_facies)

df = df.loc[df[facies_col].notnull()].copy() # omit samples with unknown facies

#%% Kruskal-Wallis test

# results = []

# # - FF -

# # FF & facies
# results.append({
#     "variable": "formation_factor",
#     "grouping": "facies",
#     **kruskal_per_group(df, ff_col, facies_col)
# })


# # - surface conductivity -

# # surface conductivity & facies
# results.append({
#     "variable": "surface_cond",
#     "grouping": "facies",
#     **kruskal_per_group(df, surfcond_col, facies_col)
# })


# # create df
# results_df = pd.DataFrame(results)

# print(results_df[[
#     "variable",
#     "grouping",
#     "H",
#     "p_value",
#     "n_groups"
# ]])

# results_df.to_csv(path_results / "kruskal_results_facies.csv", index=False)
#%% # filter for groups with >= 5 samples

group_size_litho = collect_group_size(df, litho_col)
group_size_facies = collect_group_size(df, facies_col)


valid_litho = {k for k, v in group_size_litho.items() if v >= 5}
valid_facies = {k for k, v in group_size_facies.items() if v >= 5}


df_litho_refined = df[df[litho_col].isin(valid_litho)].copy()
df_facies_refined = df[df[facies_col].isin(valid_facies)].copy()

#%% kruskal-wallis test for facies within each lithoclass
results_litho_facies = []
for litho in valid_litho:
    print(f"Testing facies within lithoclass {litho}")
    df_litho = df[df[litho_col] == litho]
    results_litho_facies.append({
        "variable": "formation_factor",
        "grouping": f"facies binnen lithoklasse {litho}",
        "lithoklasse": litho,
        **kruskal_per_group(df_litho, ff_col, facies_col)
    })

for litho in valid_litho:
    df_litho = df[df[litho_col] == litho]
    results_litho_facies.append({
        "variable": "surface_cond",
        "grouping": f"facies binnen lithoklasse {litho}",
        "lithoklasse": litho,
        **kruskal_per_group(df_litho, surfcond_col, facies_col)
    })

# create df
results_litho_facies_df = pd.DataFrame(results_litho_facies)
results_litho_facies_df.to_csv(path_results / "kruskal_litho_facies_results.csv", index=False)

#%% kruskal-wallis test for lithoclass within each facies
results_facies_litho = []
for facies in valid_facies:
    print(f"Testing lithoclass within facies {facies}")
    df_facies = df[df[facies_col] == facies]
    results_facies_litho.append({
        "variable": "formation_factor",
        "grouping": f"lithoklasse binnen facies {facies}",
        "klasse": facies,
        **kruskal_per_group(df_facies, ff_col, litho_col)
    })

for facies in valid_facies:
    df_facies = df[df[facies_col] == facies]
    results_facies_litho.append({
        "variable": "surface_cond",
        "grouping": f"lithoklasse binnen facies {facies}",
        "klasse": facies,
        **kruskal_per_group(df_facies, surfcond_col, litho_col)
    })

# create df
results_facies_litho_df = pd.DataFrame(results_facies_litho)
results_facies_litho_df.to_csv(path_results / "kruskal_facies_litho_results.csv", index=False)



#%% Dunn post-hoc test with Benjamini–Hochberg correction

dunn_ff_facies = dunn_matrix_refined(df_facies_refined, val_col = ff_col, group_col=facies_col, p_adjust="fdr_bh")
dunn_surfcond_facies = dunn_matrix_refined(df_facies_refined, val_col = surfcond_col, group_col=facies_col, p_adjust="fdr_bh")

dunn_ff_facies.to_csv(path_results / "dunn_ff_facies.csv", index=False)
dunn_surfcond_facies.to_csv(path_results / "dunn_surfcond_facies.csv", index=False)

#%% Dunn post-hoc test with Benjamini–Hochberg correction for facies within each lithoclass
dunn_litho_facies_all = []

for variable in ["formation_factor", "surface_cond"]:
    print(f"Performing Dunn post-hoc test for facies within lithoclass for {variable}")
    
    if variable == "formation_factor":
        var_short = "ff"
        val_col = ff_col 
    elif variable == "surface_cond":
        var_short = "surfcond"
        val_col = surfcond_col

    dunn_litho_facies_var = []
    for litho in valid_litho:
        print(var_short, litho)
        
        # select facies within lithoclass with >=5 samples
        sel = (results_litho_facies_df["lithoklasse"] == litho) & (results_litho_facies_df["variable"] == variable)
        # no entry -> continue to next lithoclass
        if not sel.any():
            continue  
        data = results_litho_facies_df.loc[sel]["group_sizes"].values[0]
        facies = list(data.keys())

        # perform dunn test if there are at least 2 facies groups with >=5 samples
        if len(facies)>1:
            df_litho = df[(df[litho_col] == litho) & (df[facies_col].isin(facies))]
            results_dunn = pd.DataFrame(dunn_matrix_refined(df_litho, val_col = val_col, group_col=facies_col, p_adjust="fdr_bh"))
            results_dunn.insert(loc=0, column="variable", value=variable)
            results_dunn.insert(loc=1, column="lithoclass", value=litho)
            #results_dunn.to_csv(path_results / f"dunn_{var_short}_facies_within_{litho}.csv", index=False)
            # create multiple tables            
            dunn_litho_facies_var.append(results_dunn)
            dunn_litho_facies_all.append(results_dunn)

        # combine all lithoclasses for this variable into one table
        if dunn_litho_facies_var:
            dunn_litho_facies_var_df = pd.concat(dunn_litho_facies_var, ignore_index=True)
            # only significant results
            #dunn_litho_facies_var_df_sig = dunn_litho_facies_var_df.loc[dunn_litho_facies_var_df["p_value"] < alpha]
            dunn_litho_facies_var_df_sig.sort_values("p_value", inplace=True)
            # save
            dunn_litho_facies_var_df.to_csv(path_results / f"dunn_{var_short}_facies_within_litho.csv", index=False)
            #dunn_litho_facies_var_df_sig.to_csv(path_results / f"dunn_{var_short}_facies_litho_significant.csv", index=False)

    # combine for both variables into one table
    if dunn_litho_facies_all:
        dunn_litho_facies_all_df = pd.concat(dunn_litho_facies_all, ignore_index=True)
        dunn_litho_facies_all_df.to_csv(path_results / "dunn_facies_within_litho.csv", index=False)

#%% Boxplots per lithoclass with facies on the x-as (separate figures for FF and surfcond)

path_figs = path_results / "boxplots_facies_within_litho"
path_figs.mkdir(exist_ok=True, parents=True)

# settings
min_n_per_facies_in_litho = 5  
show_points = True             
use_log_y = False              

# make boxplots for FF and surface conductivity
plot_box_per_litho(df, value_col=ff_col,       value_label="formation factor (FF)", filename_stub="box_ff")
plot_box_per_litho(df, value_col=surfcond_col, value_label="surface conductivity σₛ (S/m)", filename_stub="box_surfcond")

print(f"Boxplots opgeslagen in: {path_figs}")



#%% calculate median values for litho-facies combinations, with option to manually merge certain facies within a lithoclass

# -- median facies ---

manual_groups = [
    {"lithoklasse": "kz", "facies": ["eolisch", "glaciaal"], "group": "eolisch+glaciaal"},
    {"lithoklasse": "kz", "facies": ["fluviatiel", "marien"], "group": "fluviatiel+marien"},
    {"lithoklasse": "zf", "facies": ["eolisch", "fluviatiel"], "group": "eolisch+fluviatiel"}, # TODO: fluviatiel had ook bij glaciaal + marien gezet kunnen worden
    {"lithoklasse": "zf", "facies": ["glaciaal", "marien"], "group": "glaciaal+marien"},
    {"lithoklasse": "zm", "facies": ["eolisch", "fluviatiel", "glaciaal"], "group": "eolisch+fluviatiel+glaciaal"},

]

# create litho-facies_combos
lithofacies_combos = results_litho_facies_df.loc[results_litho_facies_df["variable"]== "formation_factor"][["lithoklasse", "group_members"]]

medians_lithofacies = calc_lithofacies_medians(
    df=df,
    lithofacies_combos=lithofacies_combos,#.rename(columns={"lithoklasse": litho_col}) if "lithoklasse" in facieslitho_combos.columns else facieslitho_combos,
    ff_col=ff_col,
    surfcond_col=surfcond_col,
    litho_col=litho_col,
    facies_col=facies_col,
    manual_groups=manual_groups
).reset_index(drop=True)

medians_lithofacies_no_groups = calc_lithofacies_medians(
    df=df,
    lithofacies_combos=lithofacies_combos,
    ff_col=ff_col,
    surfcond_col=surfcond_col,
    litho_col=litho_col,
    facies_col=facies_col,
    manual_groups=None
).reset_index(drop=True)

# save
medians_lithofacies[["LITHOKLASSE_CD", "facies_group", "median_log_ff_to_normal_space", "median_log_surfcond_to_normal_space"]].to_csv(path_results / "median_lithofacies_manual_groups_short.csv", index=False)
medians_lithofacies_no_groups[["LITHOKLASSE_CD", "facies_group", "median_log_ff_to_normal_space", "median_log_surfcond_to_normal_space"]].to_csv(path_results / "median_lithofacies_no_groups_short.csv", index=False)
medians_lithofacies.to_csv(path_results / "median_std_lithofacies_manual_groups.csv", index=False)
medians_lithofacies_no_groups.to_csv(path_results / "median_std_lithofacies_no_groups.csv", index=False)


#%%












#%% Dunn post-hoc test with Benjamini–Hochberg correction for lithoclass within each facies
dunn_facies_litho_all = []

for variable in ["formation_factor", "surface_cond"]:
    print(f"Performing Dunn post-hoc test for lithoclass within each facies for {variable}")
    
    if variable == "formation_factor":
        var_short = "ff"
        val_col = ff_col 
    elif variable == "surface_cond":
        var_short = "surfcond"
        val_col = surfcond_col

    dunn_facies_litho_var = []
    for facies in valid_facies:
        print(var_short, facies)
        
        # select litho within facies with >=5 samples
        sel = (results_facies_litho_df["klasse"] == facies) & (results_facies_litho_df["variable"] == variable)
        # no entry -> continue to next facies
        if not sel.any():
            continue  
        data = results_facies_litho_df.loc[sel]["group_sizes"].values[0]
        litho = list(data.keys())

        # perform dunn test if there are at least 2 facies groups with >=5 samples
        if len(litho)>1:
            df_facies = df[(df[facies_col] == facies) & (df[litho_col].isin(litho))]
            results_dunn = pd.DataFrame(dunn_matrix_refined(df_facies, val_col = val_col, group_col=litho_col, p_adjust="fdr_bh"))
            results_dunn.insert(loc=0, column="variable", value=variable)
            results_dunn.insert(loc=1, column="facies", value=facies)
            #results_dunn.to_csv(path_results / f"dunn_{var_short}_litho_within_{facies}.csv", index=False)
            # create multiple tables            
            dunn_facies_litho_var.append(results_dunn)
            dunn_facies_litho_all.append(results_dunn)

        # combine all lithoclasses for this variable into one table
        if dunn_facies_litho_var:
            dunn_facies_litho_var_df = pd.concat(dunn_facies_litho_var, ignore_index=True)
            # only significant results
            #dunn_facies_litho_var_df_sig = dunn_facies_litho_var_df.loc[dunn_facies_litho_var_df["p_value"] < alpha]
            dunn_facies_litho_var_df_sig.sort_values("p_value", inplace=True)
            # save
            dunn_facies_litho_var_df.to_csv(path_results / f"dunn_{var_short}_litho_within_facies.csv", index=False)
            #dunn_facies_litho_var_df_sig.to_csv(path_results / f"dunn_{var_short}_litho_all_facies_significant.csv", index=False)

    # combine for both variables into one table
    if dunn_facies_litho_all:
        dunn_facies_litho_all_df = pd.concat(dunn_facies_litho_all, ignore_index=True)
        dunn_facies_litho_all_df.to_csv(path_results / "dunn_litho_within_facies.csv", index=False)

#%% Boxplots per lithoclass with facies on the x-as (separate figures for FF and surfcond)

path_figs = path_results / "boxplots_litho_within_facies"
path_figs.mkdir(exist_ok=True, parents=True)

# settings
min_n_per_facies_in_litho = 5  
show_points = True             
use_log_y = False              

# make boxplots for FF and surface conductivity
plot_box_per_facies(df, value_col=ff_col,       value_label="formation factor (FF)", filename_stub="box_ff")
plot_box_per_facies(df, value_col=surfcond_col, value_label="surface conductivity σₛ (S/m)", filename_stub="box_surfcond")

print(f"Boxplots opgeslagen in: {path_figs}")



#%% calculate median values for facies-litho combinations, with option to manually merge certain litho within a facies

# # -- median facies ---

manual_groups = [
    {"facies": "eolisch", "lithoklasse": ["kz","zf","zm"], "group": "kz+zf+zm"},
    {"facies": "marien", "lithoklasse": ["zf","zm"], "group": "zf+zm"},
    {"facies": "glaciaal", "lithoklasse": ["kz","zm"], "group": "kz+zm"},
    {"facies": "fluviatiel", "lithoklasse": ["zf","zm"], "group": "zf+zm"}, #TODO: zg + zm had ook gekund
    #{"facies": "fluviatiel", "lithoklasse": ["zg","zm"], "group": "zg+zm"}, #TODO: zg + zm had ook gekund
]

# create facies-litho_combos
facieslitho_combos = results_facies_litho_df.loc[results_facies_litho_df["variable"]== "formation_factor"][["klasse", "group_members"]]

medians_facieslitho = calc_facieslitho_medians(
    df=df,
    facieslitho_combos=facieslitho_combos,#.rename(columns={"lithoklasse": litho_col}) if "lithoklasse" in facieslitho_combos.columns else facieslitho_combos,
    ff_col=ff_col,
    surfcond_col=surfcond_col,
    litho_col=litho_col,
    facies_col=facies_col,
    manual_groups=manual_groups
).reset_index(drop=True)

medians_facieslitho_no_groups = calc_facieslitho_medians(
    df=df,
    facieslitho_combos=facieslitho_combos,
    ff_col=ff_col,
    surfcond_col=surfcond_col,
    litho_col=litho_col,
    facies_col=facies_col,
    manual_groups=None
).reset_index(drop=True)

# save
medians_facieslitho[["facies", "litho_group", "median_log_ff_to_normal_space", "median_log_surfcond_to_normal_space"]].to_csv(path_results / "median_facieslitho_manual_groups_short.csv", index=False)
medians_facieslitho_no_groups[["facies", "litho_group", "median_log_ff_to_normal_space", "median_log_surfcond_to_normal_space"]].to_csv(path_results / "median_facieslitho_no_groups_short.csv", index=False)
medians_facieslitho.to_csv(path_results / "median_std_facieslitho_manual_groups.csv", index=False)
medians_facieslitho_no_groups.to_csv(path_results / "median_std_facieslitho_no_groups.csv", index=False)

#%%