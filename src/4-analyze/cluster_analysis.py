"""
Clusteranalyse + spreidingsplots voor Formation Factor (FF) en Surface Conductivity (sigma_s)

Outputs:
- scatterplots
Figure 1) scatterplot FF and σs, color = lithoclass
Figure 2) overview figure scatterplots with FF and σs per lithoclass, color = stratigraphy, only strat ≥ 5 samples (per litho)
Figure 3) overview figure scatterplot with FF and σs, color = stratigraphy, only strat ≥ 5 samples for at least one lithoclass
Figure 4) seperate figure per lithoclass with all strats, color = stratigraphy
Figure 5) seperate figure per lithoclass with strats, strats with sample size < 5 all together in "other" category (white)
- KMeans clustering log10(FF) and log10(sigma_s)

Project: FRESHEM (11210255-005)
Auteur: Romee van Dam (Deltares)
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
path_out = Path("data/4-output/ff_ecs_uncertainty/cluster_plots")

path_out.mkdir(parents=True, exist_ok=True)

# kolomnamen
ff_col = "SIP3_FormationFactor_F_3W_unitless"
surfcond_col = "SIP3_SurfCond_Sigmas_3W_S/m"
litho_col = "LITHOKLASSE_CD"
strat_col = "Stratigrafie"
stratlitho_col = "StratLithoklasse"

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
        print("Warning: missing lithoklasse -> afgeleid uit StratLithoklasse (laatste 2 chars).")
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

# force numeric
df[ff_col] = pd.to_numeric(df[ff_col], errors="coerce")
df[surfcond_col] = pd.to_numeric(df[surfcond_col], errors="coerce")
df = df.loc[df[ff_col].notnull() & df[surfcond_col].notnull()].copy()

# ommit negative values for log transformation
if ((df[ff_col] <= 0) | (df[surfcond_col] <= 0)).any():
    print("warning: negative values/ zero present, problem for log transformation\n these values will be omitted ")
    df = df.loc[(df[ff_col] > 0) & (df[surfcond_col] > 0)].copy()




# %% Figures 
# Figure 1)  scatterplot FF and σs, color = lithoclass
print( "create figures")

sns.set_context("talk")
sns.set_style("whitegrid")


x = ff_col
y = surfcond_col

# color per lithoklasse
fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x=x,
    y=y,
    hue=litho_col,
    palette="tab20",
    alpha=0.85,
    edgecolor="none",
    ax=ax
)
ax.set_title("FF vs σs — kleur per lithoklasse")
ax.set_xlabel("Formation factor (FF)")
ax.set_ylabel("Surface conductivity σs (S/m)")
if use_log_axes_for_plots:
    ax.set_xscale("log")
    ax.set_yscale("log")
ax.legend(title="Lithoklasse", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
_savefig(fig, path_out / "scatter_FF_vs_sigmas_by_lithoklasse.png")




# %%  
# Figure 2) overview figure scatterplots with FF and σs per lithoclass, color = stratigraphy, only strat ≥ 5 samples (per litho)

# select litho's with sample sizes >= 5
litho_counts = df[litho_col].value_counts()
valid_lithos = litho_counts[litho_counts >= min_group_size].index.tolist()

rows = []
for litho in valid_lithos:
    # select litho
    sub = df[df[litho_col] == litho].copy()

    # count stratigraphy WITHIN lithoclass
    strat_counts = sub[strat_col].value_counts()
    valid_strats = strat_counts[strat_counts >= min_group_size].index # min 5 samples

    # filter for strat ≥ 5 samples
    sub_filt = sub[sub[strat_col].isin(valid_strats)].copy()

    if sub_filt.empty:
        continue

    rows.append(sub_filt)

# combine
df_overview = pd.concat(rows, ignore_index=True)

# --- create legend ---
# create fixed colors
strats = sorted(df_overview[strat_col].astype(str).unique())
palette_strat = dict(zip(strats, sns.color_palette("tab20", n_colors=len(strats))))
# set order categories to get a fixed legend 
lithostrat_cat = df_overview[strat_col].unique() # only strat ≥ 5 samples for at least one lithoclass
df_overview[strat_col] = pd.Categorical(df_overview[strat_col], categories=strats, ordered=True)
strat_order = df_overview[strat_col].cat.categories.tolist() # fixed strat order
# connect with marker
marker_strat = dict(
    zip(strat_order, marker_list)
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
    hue=strat_col,
    style=strat_col,
    markers = marker_strat, # <-- fixed marker
    palette=palette_strat,   # <-- fixed mapping
    s=45,
    alpha=0.9,    
    edgecolor="black",
    linewidth=0.3,

)

g.add_legend(
    title="Stratigrafie",
    bbox_to_anchor=(1.02, 0.5),
    loc="center left"
)


g.set_axis_labels("Formation factor (FF)", "σs (S/m)")
g.fig.suptitle(
    f"FF vs σs — per lithoklasse (kleur = stratigrafie, n ≥ {min_group_size} per litho)",
    y=1.02
)

# log axes
if use_log_axes_for_plots:
    for ax in g.axes.flatten():
        ax.set_xscale("log")
        ax.set_yscale("log")

g.fig.tight_layout()
g.fig.savefig(
    path_out / f"facet_scatter_FF_vs_sigmas_per_litho_by_strat_min{min_group_size}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(g.fig)


#%% 
# Figure 3) overview figure scatterplot with FF and σs, color = stratigraphy, only strat ≥ 5 samples for at least one lithoclass

# select strats that have ≥ 5 samples for at least one lithoclass
df_overview_lithostrat = df.loc[df[strat_col].isin(lithostrat_cat)]
# fixed order
df_overview_lithostrat[strat_col] = pd.Categorical(df_overview_lithostrat[strat_col], categories=strat_order, ordered=True)


fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(
    data=df_overview_lithostrat,
    x=ff_col,
    y=surfcond_col,
    hue=strat_col,
    style=strat_col,
    markers = marker_strat, # <-- vaste marker
    palette=palette_strat,   # <-- vaste mapping
    s=45,
    alpha=0.9,    
    edgecolor="black",
    linewidth=0.3,
    ax=ax
)
ax.set_title(f"FF vs σs — kleur = stratigrafie, n ≥ {min_group_size} voor minimaal één lithoklasse")
ax.set_xlabel("Formation factor (FF)")
ax.set_ylabel("Surface conductivity σs (S/m)")
if use_log_axes_for_plots:
    ax.set_xscale("log")
    ax.set_yscale("log")
ax.legend(title="Stratigrafie", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
_savefig(fig, path_out / "scatter_FF_vs_sigmas_by_stratigrafie.png")




#%% 
# Figure 4) seperate figure per lithoclass with all strats, color = stratigraphy

path_out_litho = path_out / "per_lithoklasse"
path_out_litho.mkdir(parents=True, exist_ok=True)



for litho in valid_lithos:
    sub = df[df[litho_col] == litho].copy()

    
    # same order / marker / color for strats as other plots
    sub[strat_col] = (
        sub[strat_col]
        .astype("category")
        .cat.set_categories(strat_order, ordered=True)
    )
    # check which strat is present for legend
    present_strats = (
        sub[strat_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    # correct order
    present_strats = [s for s in strat_order if s in present_strats]

    # create figure
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=sub,
        x=ff_col, y=surfcond_col,
        hue=strat_col,
        style=strat_col,
        hue_order=present_strats,     # only present strats
        style_order=present_strats,   # only present strats
        markers = marker_strat, # <-- fixed marker
        palette=palette_strat,   # <-- fixed mapping
        s=45,
        alpha=0.9,
        ax=ax
    )
    ax.set_title(f"FF vs σs — lithoklasse {litho} (kleur = stratigrafie, n={len(sub)})")
    ax.set_xlabel("Formation factor (FF)")
    ax.set_ylabel("Surface conductivity σs (S/m)")
    if use_log_axes_for_plots:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # legend
    ax.legend(title="Stratigrafie", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    _savefig(fig, path_out_litho / f"scatter_FF_vs_sigmas_litho_{litho}.png")

#%% 
# Figure 5) seperate figure per lithoclass with strats, strats with sample size< 5 all together in "other" category (white)

for litho in valid_lithos:
    sub = df[df[litho_col] == litho].copy() # only with stratigraphy with > 4 samples.
    
    # count samples size stratigrafie within lithoklasse
    strat_counts = sub[strat_col].value_counts()
    valid_strats = strat_counts[strat_counts >= min_group_size].index

    # select strats with enough samples and keep same order/marker/color as other figures
    # give strat category for plot and give "other" if too few samples
    sub["Strat_plot"] = np.where(
        sub[strat_col].isin(valid_strats),
        sub[strat_col],
        "Other" 
    )
    # check which strat is present for legend
    present_strats = (
        sub["Strat_plot"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    # correct order
    present_strats = [s for s in strat_order if s in present_strats]
    # add other category
    if "Other" in sub["Strat_plot"].values:
        present_strats = present_strats + ["Other"]
    strat_order_other = strat_order + ["Other"]
    # order strats
    sub["Strat_plot"] = pd.Categorical(
        sub["Strat_plot"],
        categories=strat_order_other,
        ordered=True
    )
    # extend palette and marker for "other"
    palette_other = palette_strat.copy()
    palette_other["Other"] = (1., 1., 1.)
    marker_other = marker_strat.copy()
    marker_other["Other"] = "o"   # ✅ gevuld, dus veilig

    # create figure
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=sub,
        x=ff_col, y=surfcond_col,
        hue="Strat_plot",
        style="Strat_plot",
        hue_order=present_strats,
        style_order=present_strats,
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
        f"stratigrafieën ≥ {min_group_size} samples (gekleurd), < {min_group_size} = Other "
    )
    ax.set_xlabel("Formation factor (FF)")
    ax.set_ylabel("Surface conductivity σs (S/m)")

    if use_log_axes_for_plots:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.legend(
        title="Stratigrafie",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True
    )

    _savefig(
        fig,
        path_out_litho / f"scatter_FF_vs_sigmas_litho_{litho}_strat_other_ss{min_group_size}.png"
    )

#%% org

# path_out_litho = path_out / "per_lithoklasse"
# path_out_litho.mkdir(parents=True, exist_ok=True)

# # optional: litho's with sample sizes >= 5
# litho_counts = df[litho_col].value_counts()
# valid_lithos = litho_counts[litho_counts >= min_group_size].index.tolist()

# for litho in valid_lithos:
#     sub = df[df[litho_col] == litho].copy()

#     fig, ax = plt.subplots(figsize=(9, 7))
#     sns.scatterplot(
#         data=sub,
#         x=ff_col, y=surfcond_col,
#         hue=strat_col,
# #        style=strat_col,
# #        markers = marker_strat, # <-- vaste marker
# #        palette=palette_strat,   # <-- vaste mapping
#         s=45,
#         alpha=0.9,
#         ax=ax
#     )
#     ax.set_title(f"FF vs σs — lithoklasse {litho} (kleur = stratigrafie, n={len(sub)})")
#     ax.set_xlabel("Formation factor (FF)")
#     ax.set_ylabel("Surface conductivity σs (S/m)")
#     if use_log_axes_for_plots:
#         ax.set_xscale("log")
#         ax.set_yscale("log")

#     # Legenda buiten, maar bij veel strats alsnog lang → zie tip hieronder
#     ax.legend(title="Stratigrafie", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
#     _savefig(fig, path_out_litho / f"scatter_FF_vs_sigmas_litho_{litho}.png")

# # strats with sample size< 5 in seperate "other" category (grey)
# for litho in valid_lithos:
#     sub = df[df[litho_col] == litho].copy() # only with stratigrafie klasse met > 4 samples.

#     # count samples size stratigrafie withon lithoklasse
#     strat_counts = sub[strat_col].value_counts()
#     valid_strats = strat_counts[strat_counts >= min_group_size].index

#     # nieuwe kolom: stratigrafie of "Other"
#     sub["Strat_plot"] = np.where(
#         sub[strat_col].isin(valid_strats),
#         sub[strat_col],
#         "Other"
#     )

#     # vaste volgorde in legenda
#     ordered_strats = list(sorted(valid_strats)) + ["Other"]
#     sub["Strat_plot"] = pd.Categorical(
#         sub["Strat_plot"],
#         categories=ordered_strats,
#         ordered=True
#     )

#     # palette: kleuren voor geldige strat, grijs voor Other
#     palette = dict(
#         zip(
#             sorted(valid_strats),
#             sns.color_palette("tab10", n_colors=len(valid_strats))
#         )
#     )
#     palette["Other"] = (0.7, 0.7, 0.7)

#     fig, ax = plt.subplots(figsize=(9, 7))
#     sns.scatterplot(
#         data=sub,
#         x=ff_col, y=surfcond_col,
#         hue="Strat_plot",
#         palette=palette,
#         s=45,
#         alpha=0.9,
#         edgecolor="none",
#         ax=ax
#     )

    
#     n_other = (sub["Strat_plot"] == "Other").sum()

#     ax.set_title(
#         f"FF vs σs — lithoklasse {litho}\n"
#         f"stratigrafieën ≥ {min_group_size} gekleurd, < {min_group_size} = Other (n_other={n_other})"
#     )
#     ax.set_xlabel("Formation factor (FF)")
#     ax.set_ylabel("Surface conductivity σs (S/m)")

#     if use_log_axes_for_plots:
#         ax.set_xscale("log")
#         ax.set_yscale("log")

#     ax.legend(
#         title="Stratigrafie",
#         bbox_to_anchor=(1.02, 1),
#         loc="upper left",
#         frameon=True
#     )

#     # _savefig(
#     #     fig,
#     #     path_out_litho / f"scatter_FF_vs_sigmas_litho_{litho}_strat_other_ss{min_group_size}.png"
#     # )









#%% 
# Clusteranalysis (KMeans) on log10(FF) and log10(σs)

os.environ["OMP_NUM_THREADS"] = "2"

# FF are σs are mostly right skewed therefore also log-log plot
df_cluster = df.loc[df[surfcond_col]>0.000101].copy()
df_cluster["log10_FF"] = np.log10(df_cluster[ff_col])
df_cluster["log10_sigmas"] = np.log10(df_cluster[surfcond_col])

X = df_cluster[["log10_FF", "log10_sigmas"]].values

# scale to mean=0, std=1 to prevetn clustering dominated by one axis
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# Silhouette-scan to select appropriate k (for reproducibility)
sil_scores = {}
for k in k_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(Xz)
    sil = silhouette_score(Xz, labels)
    sil_scores[k] = sil

best_k = max(sil_scores, key=sil_scores.get)
print("Silhouette scores:", sil_scores)
print("Best k volgens silhouette:", best_k)

# Fit model
kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=random_state)
df_cluster["cluster_kmeans"] = kmeans.fit_predict(Xz)

# save clusterlabels
#df_cluster.to_csv(path_out / "ff_sigmas_log_with_kmeans_clusters.csv", index=False)

# plot clusters in log-space
fig, ax = plt.subplots(figsize=(9, 7))
sns.scatterplot(
    data=df_cluster,
    x="log10_FF",
    y="log10_sigmas",
    hue="cluster_kmeans",
    palette="tab10",
    alpha=0.9,
    edgecolor="none",
    ax=ax
)
ax.set_title(f"KMeans clusters in log-ruimte (k={best_k})")
ax.set_xlabel("log10(FF)")
ax.set_ylabel("log10(σs)")
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
_savefig(fig, path_out / "clusters_kmeans_logbased.png")

# optional cluster transformed to normal values
fig, ax = plt.subplots(figsize=(9, 7))
sns.scatterplot(
    data=df_cluster,
    x=ff_col,
    y=surfcond_col,
    hue="cluster_kmeans",
    palette="tab10",
    alpha=0.9,
    edgecolor="none",
    ax=ax
)
ax.set_title(f"KMeans clusters (k={best_k}) in originele eenheden")
ax.set_xlabel("Formation factor (FF)")
ax.set_ylabel("Surface conductivity σs (S/m)")

ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
_savefig(fig, path_out / "clusters_kmeans_transformed_to_original_units.png")

print(f"Klaar. Figuren + CSV opgeslagen in: {path_out.resolve()}")
#%%