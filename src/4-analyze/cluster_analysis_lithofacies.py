"""
Clusteranalyse + spreidingsplots voor Formation Factor (FF) en Surface Conductivity (sigma_s) for facies and lithoclasses

Outputs:
- scatterplots

Figure 2) overview figure scatterplots with FF and σs per lithoclass, color = facies, only facies ≥ 5 samples (per litho)
Figure 3) overview figure scatterplot with FF and σs, color = facies, only facies ≥ 5 samples for at least one lithoclass
Figure 4) seperate figure per lithoclass with all facies, color = facies
Figure 5) seperate figure per lithoclass with facies, facies with sample size < 5 all together in "other" category (white)
- KMeans clustering log10(FF) and log10(sigma_s)

Project: FRESHEM (11210255-005)
Auteur: Romee van Dam (Deltares)
date: 11-06-26
"""

#%% #imports
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#%% #paths and parameters

# run from basedir, assuming script resides in subdir of src/
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

path_labresults = Path("data/3-input/lab_results")
fn_labresults = path_labresults / "20260304_tbl20_WPchloride_FFdata.xlsx"
path_out = Path("data/4-output/ff_ecs_uncertainty/cluster_plots_lithofacies")

path_out.mkdir(parents=True, exist_ok=True)

# kolomnamen
ff_col = "SIP3_FormationFactor_F_3W_unitless"
surfcond_col = "SIP3_SurfCond_Sigmas_3W_S/m"
litho_col = "LITHOKLASSE_CD"
strat_col = "Stratigrafie"
stratlitho_col = "StratLithoklasse"
facies_col = "facies"

# filters/keuzes
type_filter_col = "Type_name"
type_filter_val = "FF_Disturbed"   # alleen deze gebruiken, want anders mengeling van ongestoorde en verstoorde samples op zelfde locatie
min_group_size = 5                 
use_log_axes_for_plots = True      # log10-as voor FF/σs in de figuren (handig bij scheve verdelingen)
random_state = 42

# settings plots
# cluster range voor silhouette (optioneel)
k_range = range(2, 9)

marker_list = [
    "o",   # circle
    "s",   # square
    "D",   # diamond
    "^",   # triangle up
    "v",   # triangle down
    "<",   # triangle left
    ">",   # triangle right
    "P",   # plus filled
    "X",   # x filled
    "*",   # star
    "h",   # hexagon
    "H",   # filled hexagon
    "8",   # octagon
    "p",   # pentagon
    "d",   # thin diamond
    "o",   # circle again (only if you ever need 16+)
]



#%% 
# definitions

# sorting of classes
def _sorted_categories(series: pd.Series):
    return sorted(series.astype(str).unique())

# save fig
def _savefig(fig, outpath: Path, dpi=300):
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


#%% 
# prepare data

df_all = pd.read_excel(
    fn_labresults,
    keep_default_na=False,
    na_values=["", " ", "NULL", "NaN"]  # zodat NA (Naaldwijk) niet als NaN wordt gezien
)

# select only disturbed samples
if type_filter_col in df_all.columns:
    df = df_all.loc[df_all[type_filter_col] == type_filter_val].copy()
else:
    df = df_all.copy()

# if lithoclass is missing, look if statlithoclass exists
if litho_col in df.columns and stratlitho_col in df.columns:
    missing_litho = df[litho_col].isna() | (df[litho_col].astype(str).str.strip() == "")
    if missing_litho.any():
        print("Warning: missing lithoklasse -> afgeleid uit StratLithoklasse (laatste 2 characters).")
        df.loc[missing_litho, litho_col] = (
            df.loc[missing_litho, stratlitho_col].astype(str).str[-2:] #TODO: works only for sandy classes.(only category where problem arises right now) 
        )

# drop missing values in important columns
needed = [ff_col, surfcond_col, litho_col, strat_col]
for c in needed:
    if c not in df.columns:
        raise KeyError(f"Kolom ontbreekt in Excel: {c}")

df = df.loc[
    df[ff_col].notnull()
    & df[surfcond_col].notnull()
    & df[litho_col].notnull()
    & df[strat_col].notnull()
].copy()

# do not take AAOM (anthropogenic) for analysis 
df = df.loc[df["Stratigrafie"]!='AAOM'].copy()

# remove  MG and WG suffixes
df[strat_col] = df[strat_col].str.replace("-(MG|WG)", "", regex=True)
df[stratlitho_col] = df[stratlitho_col].str.replace("-(MG|WG)", "", regex=True)


# force numeric
df[ff_col] = pd.to_numeric(df[ff_col], errors="coerce")
df[surfcond_col] = pd.to_numeric(df[surfcond_col], errors="coerce")
df = df.loc[df[ff_col].notnull() & df[surfcond_col].notnull()].copy()

# ommit negative values for log transformation
if ((df[ff_col] <= 0) | (df[surfcond_col] <= 0)).any():
    print("warning: negative values/ zero present, problem for log transformation\n these values will be omitted ")
    df = df.loc[(df[ff_col] > 0) & (df[surfcond_col] > 0)].copy()

#%% prepare facies groups

facies_list = ['marien' , 'fluviatiel', 'glaciaal', 'eolisch', 'organisch', 'rest']

marien_codes = ['NAWA', 'NAWO', 'NAZA', 'NAWOBE', 'EE', 'OO', 'MS', 'OOSP', 'BR', 'OO', 'MS', 'OOSP', 'WAWO' ]

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


# %% Figures 
# # Figure 1)  scatterplot FF and σs, color = lithoclass


# palette = {
#     "k":  "#2ca02c",   # green
#     "v":  "#d62728",   # red
    
#     # sandy classes: light → dark
#     "zf": "#f3eaa3",   # very light yellow
#     "zm": "#eebd5a",   # yellow-orange
#     "zg": "#c26d0c",   # orange
#     "z":  "#7c5c04",   # dark orange
#     "kz": "#d3ff11",   # orange
    
#     "sch": "#000000"   # black
# }
# hue_order = ["zf", "zm", "zg", "z","kz", "k", "v", "sch"]

# print( "create figures")

# sns.set_context("talk")
# sns.set_style("whitegrid")


# x = ff_col
# y = surfcond_col

# # color per lithoklasse
# fig, ax = plt.subplots(figsize=(10, 7))
# sns.scatterplot(
#     data=df,
#     x=x,
#     y=y,
#     hue=litho_col,
#     hue_order=hue_order,
#     palette=palette,
#     alpha=0.85,
#     edgecolor="none",
#     ax=ax
# )
# ax.set_title("FF vs σs — kleur per lithoklasse")
# ax.set_xlabel("Formation factor (FF)")
# ax.set_ylabel("Surface conductivity σs (S/m)")
# if use_log_axes_for_plots:
#     ax.set_xscale("log")
#     ax.set_yscale("log")
# ax.legend(title="Lithoklasse", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
# _savefig(fig, path_out / "scatter_FF_vs_sigmas_by_lithoklasse.png")




# %%  
# Figure 2) overview figure scatterplots with FF and σs per lithoclass, color = facies, only facies ≥ 5 samples (per litho)

# select litho's with sample sizes >= 5
litho_counts = df[litho_col].value_counts()
valid_lithos = litho_counts[litho_counts >= min_group_size].index.tolist()

rows = []
for litho in valid_lithos:
    # select litho
    sub = df[df[litho_col] == litho].copy()

    # count facies WITHIN lithoclass
    facies_counts = sub[facies_col].value_counts()
    valid_facies = facies_counts[facies_counts >= min_group_size].index # min 5 samples

    # filter for facies ≥ 5 samples
    sub_filt = sub[sub[facies_col].isin(valid_facies)].copy()

    if sub_filt.empty:
        continue

    rows.append(sub_filt)

# combine
df_overview = pd.concat(rows, ignore_index=True)

# --- create legend ---
# create fixed colors
facies = sorted(df_overview[facies_col].astype(str).unique())
palette_facies = dict(zip(facies, sns.color_palette("tab20", n_colors=len(facies))))
# set order categories to get a fixed legend 
lithofacies_cat = df_overview[facies_col].unique() # only facies ≥ 5 samples for at least one lithoclass
df_overview[facies_col] = pd.Categorical(df_overview[facies_col], categories=facies, ordered=True)
facies_order = df_overview[facies_col].cat.categories.tolist() # fixed facies order
# connect with marker
marker_facies = dict(
    zip(facies_order, marker_list)
)

# create plot grid
g = sns.FacetGrid(
    df_overview,
    col=litho_col,
    col_wrap=3,
    height=3.2,
    sharex=True,
    sharey=True
)

g.map_dataframe(
    sns.scatterplot,
    x=ff_col,
    y=surfcond_col,
    hue=facies_col,
    style=facies_col,
    markers = marker_facies, # <-- fixed marker
    palette=palette_facies,   # <-- fixed mapping
    s=45,
    alpha=0.9,    
    edgecolor="black",
    linewidth=0.3,

)

g.add_legend(
    title="facies",
    bbox_to_anchor=(1.02, 0.5),
    loc="center left"
)


g.set_axis_labels("Formation factor (FF)", "σs (S/m)")
g.fig.suptitle(
    f"FF vs σs — per lithoklasse (kleur = facies, n ≥ {min_group_size} per litho)",
    y=1.02
)

# log axes
if use_log_axes_for_plots:
    for ax in g.axes.flatten():
        ax.set_xscale("log")
        ax.set_yscale("log")

g.fig.tight_layout()
g.fig.savefig(
    path_out / f"facet_scatter_FF_vs_sigmas_per_litho_by_facies_min{min_group_size}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(g.fig)


#%% 
# Figure 3) overview figure scatterplot with FF and σs, color = facies, only facies ≥ 5 samples for at least one lithoclass

# select facies that have ≥ 5 samples for at least one lithoclass
df_overview_lithofacies = df.loc[df[facies_col].isin(lithofacies_cat)]
# fixed order
df_overview_lithofacies[facies_col] = pd.Categorical(df_overview_lithofacies[facies_col], categories=facies_order, ordered=True)


fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(
    data=df_overview_lithofacies,
    x=ff_col,
    y=surfcond_col,
    hue=facies_col,
    style=facies_col,
    markers = marker_facies, # <-- vaste marker
    palette=palette_facies,   # <-- vaste mapping
    s=45,
    alpha=0.9,    
    edgecolor="black",
    linewidth=0.3,
    ax=ax
)
ax.set_title(f"FF vs σs — kleur = facies, n ≥ {min_group_size} voor minimaal één lithoklasse")
ax.set_xlabel("Formation factor (FF)")
ax.set_ylabel("Surface conductivity σs (S/m)")
if use_log_axes_for_plots:
    ax.set_xscale("log")
    ax.set_yscale("log")
ax.legend(title="facies", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
_savefig(fig, path_out / "scatter_FF_vs_sigmas_by_facies.png")




#%% 
# Figure 4) seperate figure per lithoclass with all facies, color = facies

path_out_litho = path_out / "per_lithoklasse"
path_out_litho.mkdir(parents=True, exist_ok=True)



for litho in valid_lithos:
    sub = df[df[litho_col] == litho].copy()

    
    # same order / marker / color for facies as other plots
    sub[facies_col] = (
        sub[facies_col]
        .astype("category")
        .cat.set_categories(facies_order, ordered=True)
    )
    # check which facies is present for legend
    present_facies = (
        sub[facies_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    # correct order
    present_facies = [s for s in facies_order if s in present_facies]

    # create figure
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=sub,
        x=ff_col, y=surfcond_col,
        hue=facies_col,
        style=facies_col,
        hue_order=present_facies,     # only present facies
        style_order=present_facies,   # only present facies
        markers = marker_facies, # <-- fixed marker
        palette=palette_facies,   # <-- fixed mapping
        s=45,
        alpha=0.9,
        ax=ax
    )
    ax.set_title(f"FF vs σs — lithoklasse {litho} (kleur = facies, n={len(sub)})")
    ax.set_xlabel("Formation factor (FF)")
    ax.set_ylabel("Surface conductivity σs (S/m)")
    if use_log_axes_for_plots:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # legend
    ax.legend(title="facies", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    _savefig(fig, path_out_litho / f"scatter_FF_vs_sigmas_litho_{litho}.png")

#%% 
# Figure 5) seperate figure per lithoclass with facies, facies with sample size< 5 all together in "other" category (white)

for litho in valid_lithos:
    sub = df[df[litho_col] == litho].copy() # only with facies with > 4 samples.
    
    # count samples size facies within lithoklasse
    facies_counts = sub[facies_col].value_counts()
    valid_facies = facies_counts[facies_counts >= min_group_size].index

    # select facies with enough samples and keep same order/marker/color as other figures
    # give facies category for plot and give "other" if too few samples
    sub["facies_plot"] = np.where(
        sub[facies_col].isin(valid_facies),
        sub[facies_col],
        "Other" 
    )
    # check which facies is present for legend
    present_facies = (
        sub["facies_plot"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    # correct order
    present_facies = [s for s in facies_order if s in present_facies]
    # add other category
    if "Other" in sub["facies_plot"].values:
        present_facies = present_facies + ["Other"]
    facies_order_other = facies_order + ["Other"]
    # order facies
    sub["facies_plot"] = pd.Categorical(
        sub["facies_plot"],
        categories=facies_order_other,
        ordered=True
    )
    # extend palette and marker for "other"
    palette_other = palette_facies.copy()
    palette_other["Other"] = (1., 1., 1.)
    marker_other = marker_facies.copy()
    marker_other["Other"] = "o"   # ✅ gevuld, dus veilig

    # create figure
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=sub,
        x=ff_col, y=surfcond_col,
        hue="facies_plot",
        style="facies_plot",
        hue_order=present_facies,
        style_order=present_facies,
        markers=marker_other,
        palette=palette_other,
        s=45,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.3,
        ax=ax
    )

    ax.set_title(
        f"FF vs σs — lithoklasse {litho}\n"
        f"facies ≥ {min_group_size} samples (gekleurd), < {min_group_size} = Other "
    )
    ax.set_xlabel("Formation factor (FF)")
    ax.set_ylabel("Surface conductivity σs (S/m)")

    if use_log_axes_for_plots:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.legend(
        title="facies",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True
    )

    _savefig(
        fig,
        path_out_litho / f"scatter_FF_vs_sigmas_litho_{litho}_facies_other_ss{min_group_size}.png"
    )

#%%