"""
This script creates the look-up table with values and distribution of the formation factor and surface conductivity for the monte carlo analysis to perform statistical analysis of the formation factor and surface conductivity, and to check for differences between facies.

output of this script:
1) look-up table with lithoklasse, stratigraphy, facies, and mean, std, and distribution of FF and ECs 

project: FRESHEM (11210255-005)
author: Romee van Dam (Deltares)
date: 17-08-26
"""

#%% 
# imports

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
import re

#%% 
# paths and parameters

# run from basedir, assuming script resides in subdir of src/
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

path_input = Path("data/3-input")
path_geotop_codes = "data/1-external/GeoTOP_lithostrat_afzettingsmilieus _JanG.csv"
path_regis_codes = "data/1-external/REGIS_lithostrat_afzettingsmilieus _JanG.csv"
path_sample_data = f"{path_input}/20260304_tbl20_WPchloride_FFdata_with_facies.csv"
path_monte_carlo =Path("data/4-output/ff_ecs_uncertainty/for_monte_carlo")
#path_results_facies_litho = f"{path_monte_carlo}/lithofacies_for_monte_carlo_with_stratigrafie.csv"
path_results_facies_litho =  f"{path_monte_carlo}/lithofacies_for_monte_carlo.csv"
path_results_litho = f"{path_monte_carlo}/litho_for_monte_carlo.csv"
path_results_strat_litho = f"data/4-output/ff_ecs_uncertainty/dunn_test_results_lithostrat/median_mean_std_stratlitho_no_groups.csv"

path_monte_carlo.mkdir(exist_ok=True, parents=True)

#%%
# definitions


def simplify_strat_name(strat):
    """ Function to simplify stratigraphy code before lookup. Removes digits and small letters at the end of the code. """
    if pd.isna(strat):
        return np.nan

    strat = str(strat)

    # verwijder cijfers aan het eind
    strat = re.sub(r"\d+$", "", strat)

    # verwijder 2 kleine letters aan het eind (ga, gb, gc, gd, ge)
    strat = re.sub(r"[a-z]{2}$", "", strat)

    return strat

def make_strat_short_name(strat):
    """
    Convert voxel strat code to code used in statistics tables.

    Examples:
    NUNAWA1  -> NAWA
    NUECge   -> EC
    NUNAWOgd -> NAWO
    NUNIHO   -> NIHO
    CKMA     -> CKMA
    NMTO     -> NMTO
    """

    if pd.isna(strat):
        return np.nan

    strat = str(strat)

    # alleen NU-prefix verwijderen
    if strat.startswith("NU"):
        strat = strat[2:]

    # # cijfers aan eind verwijderen
    # strat = re.sub(r"\d+$", "", strat)

    # # kleine letters aan eind verwijderen (ga, gb, gd, ge, nb, ...)
    # strat = re.sub(r"[a-z]+$", "", strat)

    return strat

#%%
# read in data

#strat_codes = pd.read_csv(path_strat_codes, index_col=0).reset_index()
geotop_codes = pd.read_csv(path_geotop_codes)
regis_codes = pd.read_csv(path_regis_codes)
df = pd.read_csv(path_sample_data, index_col=0)
#results_facies_litho = pd.read_csv(path_results_facies_litho, index_col=0).reset_index()
results_litho = pd.read_csv(path_results_litho, index_col=0).reset_index()
results_strat_litho = pd.read_csv(path_results_strat_litho, index_col=0).reset_index()
results_facies_litho = pd.read_csv(path_results_facies_litho, index_col=0).reset_index()

lithoklassen_naam = {"a": "antropogeen",
                     "v": "organisch materiaal (veen)",
                     "k": "klei",
                     "kz": "kleiig zand, zandige klei en leem",
                     "zf": "zand fijn",
                     "zm": "zand midden",
                     "zg": "zand grof",
                     "g": "grind",
                     "sch": "schelpen",
                     "r": "rest"}

#%%
#prepare strat codes

geotop_codes = geotop_codes.rename(
    columns={
        "VOXEL_NR": "user_nr",
        "STR_UNIT_CD": "stratigrafie",
    }
)

regis_codes = regis_codes.rename(
    columns={
        "formation": "stratigrafie",
    }
)


geotop_codes = geotop_codes[
    ["stratigrafie", "user_nr", "facies"]
]

regis_codes = regis_codes[
    ["stratigrafie", "user_nr", "facies"]
]

strat_codes = pd.concat(
    [geotop_codes, regis_codes],
    ignore_index=True
)

#%% 
# prepare facies groups

# facies_list = ['marien' , 'fluviatiel', 'glaciaal', 'eolisch', 'organisch', 'rest']

# marien_codes = ['NAWA', 'NAWO', 'NAZA', 'NAWOBE', 'EE', 'OO', 'MS', 'OOSP', 'BR', 'WAWO' ]

# fluviatiel_codes = ['URTY', 'URVE', 'AP', 'BXSI', 'UR', 'PZ', 'EC', 'ST', 'WA', 'KK', 'KW' ]

# glaciaal_codes = ['DRGI', 'DRGIGA', 'PENI', 'PE', 'DRUI'] 

# eolisch_codes = ['BX', 'DN', 'BXWI', 'BXKO', 'NASC' ] 

# organisch_codes = ['NIHO', 'NIBA', 'NI']

# rest_codes = ['AAOM'] #TODO: 'NA'?

# codes_per_facies = {
#     "marien": marien_codes,
#     "fluviatiel": fluviatiel_codes,
#     "glaciaal": glaciaal_codes,
#     "eolisch": eolisch_codes,
#     "organisch": organisch_codes,
#     "rest": rest_codes,
# }

# facies_map = {}
# for code in marien_codes:
#     facies_map[code] = "marien"
# for code in fluviatiel_codes:
#     facies_map[code] = "fluviatiel"
# for code in glaciaal_codes:
#     facies_map[code] = "glaciaal"
# for code in eolisch_codes:
#     facies_map[code] = "eolisch"
# for code in organisch_codes:
#     facies_map[code] = "organisch"
# for code in rest_codes:
#     facies_map[code] = "rest"


# facies_map_nu = {}

# for code, facies in facies_map.items():
#     facies_map_nu[f"NU{code}"] = facies

# short_strat_name = {}

# for code, _ in facies_map.items():
#     short_strat_name[f"NU{code}"] = code


def normalize_strat_code(code):
    """Normalize stratigraphy code before lookup."""
    if pd.isna(code):
        return np.nan
    return str(code).strip().upper()


#%%
# prepare lithofacies groups
rows = []

for _, row in results_facies_litho.iterrows():

    litho = row["LITHOKLASSE_CD"]
    facies_group = row["facies_group"]

    for facies in facies_group.split("+"):

        rows.append({
            "LITHOKLASSE_CD": litho,
            "facies": facies,
            "facies_group": facies_group,
            "mean_log_ff": row["mean_log_ff"],
            "mean_log_surfcond": row["mean_log_surfcond"],
            "std_log_ff": row["std_log_ff"],
            "std_log_surfcond": row["std_log_surfcond"],
            "n": row["n"],
        })

facies_lookup = pd.DataFrame(rows)

#%%
df_litho_naam = pd.DataFrame({
    "LITHOKLASSE_CD": list(lithoklassen_naam.keys()),
    "Lithoklasse_naam": list(lithoklassen_naam.values())
})

lookup_table = (
    strat_codes.merge(df_litho_naam, how="cross")
)

lookup_table["strat_short_name"] = (
    lookup_table["stratigrafie"]
    .apply(make_strat_short_name)
)

# raise warning if names are not unique after first simplification 
if len(lookup_table["strat_short_name"].unique()) != len(lookup_table["stratigrafie"].unique()):
    raise ValueError(
        f"Aantal unieke strat_short_name ({len(lookup_table['strat_short_name'].unique())}) "
        f"is niet gelijk aan aantal unieke stratigrafie codes ({len(lookup_table['stratigrafie'].unique())})"
    )

lookup_table["strat_match"] = (
    lookup_table["strat_short_name"]
    .apply(simplify_strat_name)
)

# lookup_table["strat_short_name"] = lookup_table["strat_match"].apply(
#     lambda x: short_strat_name.get(x, np.nan)
# )

# lookup_table["facies"] = lookup_table["strat_match"].apply(
#     lambda x: facies_map.get(normalize_strat_code(x), np.nan)
# )


lookup_table["mean_dist_ff"] = np.nan
lookup_table["mean_dist_surfcond"] = np.nan
lookup_table["std_dist_ff"] = np.nan
lookup_table["std_dist_surfcond"] = np.nan
lookup_table["distribution_type"] = "NaN"
lookup_table["statistiek_literatuur"] = "NaN"
lookup_table["groepering_statistiek"] = "NaN"
lookup_table["n"] = np.nan



#%%

# =============================================================================
# STAP 1: specifieke statistieken op stratigrafie-niveau
# =============================================================================

#TODO: UR, 
# NI (krijgt geen facies maar wel de juiste groepering)

special_stats = [
    ("v", "NIHO"),
    ("v", "NIBA"),
]

for litho, strat in special_stats:

    strat_stats = results_strat_litho.loc[
        (results_strat_litho["LITHOKLASSE_CD"] == litho)
        & (results_strat_litho["strat_group"] == strat)
    ].iloc[0]

    mask = (
        (lookup_table["LITHOKLASSE_CD"] == litho)
        & (lookup_table["strat_match"] == strat)
    )

    lookup_table.loc[mask, [
        "mean_dist_ff",
        "mean_dist_surfcond",
        "std_dist_ff",
        "std_dist_surfcond"
    ]] = [
        strat_stats["mean_log_ff"],
        strat_stats["mean_log_surfcond"],
        strat_stats["std_log_ff"],
        strat_stats["std_log_surfcond"],
    ]

    lookup_table.loc[mask, "distribution_type"] = "log"
    lookup_table.loc[mask, "statistiek_literatuur"] = "statistiek"
    lookup_table.loc[mask, "groepering_statistiek"] = (
        f"{litho}-[{strat}]"
    )



#TODO: specifieke waardes toevoegen voor significant verschillende stratigrafien.
#%%

# =============================================================================
# 2. Vul lookup-tabel met facies-binnen-litho statistieken
# =============================================================================
#TODO: volgens mij gaat dit niet overal goed nog. -> classificeren vanuit facies lijst vanuit tno


lookup_table = lookup_table.merge(
    facies_lookup[
        [
            "LITHOKLASSE_CD",
            "facies",
            "facies_group",
            "n",
            "mean_log_ff",
            "mean_log_surfcond",
            "std_log_ff",
            "std_log_surfcond",
        ]
    ],
    left_on=["LITHOKLASSE_CD", "facies"],
    right_on=["LITHOKLASSE_CD", "facies"],
    how="left",
    suffixes=("", "_stat")
)

# vullen waar match gevonden is
mask = (lookup_table["mean_dist_ff"].isna() & lookup_table["mean_log_ff"].notna())

lookup_table.loc[mask, "mean_dist_ff"] = lookup_table.loc[mask, "mean_log_ff"].astype(float)
lookup_table.loc[mask, "mean_dist_surfcond"] = lookup_table.loc[mask, "mean_log_surfcond"].astype(float)
lookup_table.loc[mask, "std_dist_ff"] = lookup_table.loc[mask, "std_log_ff"].astype(float)
lookup_table.loc[mask, "std_dist_surfcond"] = lookup_table.loc[mask, "std_log_surfcond"].astype(float)
lookup_table.loc[mask, "n"] = lookup_table.loc[mask, "n_stat"].astype(float)

lookup_table.loc[mask, "distribution_type"] = "log"
lookup_table.loc[mask, "statistiek_literatuur"] = "statistiek"



lookup_table.loc[mask, "groepering_statistiek"] = (
    lookup_table.loc[mask, "LITHOKLASSE_CD"]
    + "-["
    + lookup_table.loc[mask, "facies_group"]
    + "]"
)

lookup_table = lookup_table.drop(
    columns=[
        "facies_group",
        "mean_log_ff",
        "mean_log_surfcond",
        "std_log_ff",
        "std_log_surfcond",
        "n_stat"
    ]
)



#%%
# =============================================================================
# 3. Vul lookup-tabel met litho statistieken
# =============================================================================

df_litho = results_litho.copy()

lookup_table = lookup_table.merge(
    df_litho[
        [
            "LITHOKLASSE_CD",
            "n",
            "mean_log_ff",
            "mean_log_surfcond",
            "std_log_ff",
            "std_log_surfcond",
        ]
    ],
    left_on=["LITHOKLASSE_CD"],
    right_on=["LITHOKLASSE_CD"],
    how="left",
    suffixes=("", "_stat")
)

# vullen waar match gevonden is
mask = (lookup_table["mean_dist_ff"].isna() & lookup_table["mean_log_ff"].notna())

lookup_table.loc[mask, "mean_dist_ff"] = lookup_table.loc[mask, "mean_log_ff"].astype(float)
lookup_table.loc[mask, "mean_dist_surfcond"] = lookup_table.loc[mask, "mean_log_surfcond"].astype(float)
lookup_table.loc[mask, "std_dist_ff"] = lookup_table.loc[mask, "std_log_ff"].astype(float)
lookup_table.loc[mask, "std_dist_surfcond"] = lookup_table.loc[mask, "std_log_surfcond"].astype(float)
lookup_table.loc[mask, "n"] = lookup_table.loc[mask, "n_stat"].astype(float)

lookup_table.loc[mask, "distribution_type"] = "log"
lookup_table.loc[mask, "statistiek_literatuur"] = "statistiek"

lookup_table.loc[mask, "groepering_statistiek"] = (
    lookup_table.loc[mask, "LITHOKLASSE_CD"]
)

lookup_table = lookup_table.drop(
    columns=[
        "mean_log_ff",
        "mean_log_surfcond",
        "std_log_ff",
        "std_log_surfcond",
        "n_stat"
    ]
)

lookup_table.to_csv(f"{path_monte_carlo}/look_up_table_ff_ECs.csv", index=False)

#%%