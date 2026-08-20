"""
This script creates the look-up table with values and distribution of the formation factor and surface conductivity for the monte carlo analysis
with data used in FRESHEM-zeeland

output of this script:
1) look-up table with stratigraphy, lithoklasse, facies, and mean, std, and distribution of FF and ECs (FRESHEM-ZEELAND data) 

project: FRESHEM (11210255-005)
author: Romee van Dam (Deltares)
date: 20-08-26
"""

#%% 
# imports

import pandas as pd
from pathlib import Path
import os
import numpy as np
import re

#%% 
# paths and parameters

# run from basedir, assuming script resides in subdir of src/
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

path_input = Path("data/3-input")
path_external = Path("data/1-external")
path_geotop_codes = f"{path_external}/GeoTOP_lithostrat_afzettingsmilieus _JanG.csv"
path_regis_codes = f"{path_external}/REGIS_lithostrat_afzettingsmilieus _JanG.csv"
path_sample_data = f"{path_input}/20260304_tbl20_WPchloride_FFdata_with_facies.csv"
path_monte_carlo =Path("data/4-output/ff_ecs_uncertainty/for_monte_carlo")
path_results_facies_litho =  f"{path_monte_carlo}/lithofacies_for_monte_carlo.csv"
path_results_litho = f"{path_monte_carlo}/litho_for_monte_carlo.csv"
path_results_strat_litho = f"data/4-output/ff_ecs_uncertainty/dunn_test_results_lithostrat/median_mean_std_stratlitho_manual_groups.csv"
path_results_litho_old = f"{path_external}/geotop_lithoclass_props_oude_FF_ECs.csv"

# mean_dist_ff_grind = 6.5
# mean_dist_ff_schelpen = 5.0
#replacement_litho = "zm" # for "antropogeen" use statistics of this lithoklasse as replacement for missing values

conversion_to_mS_per_cm = 1000/100 # from S/m to mS/cm
conversion_to_S_per_m = 100/1000 # from mS/cm to S/m

lithoklassen_naam = {"a": "antropogeen",
                     "v": "organisch materiaal (veen)",
                     "k": "klei",
                     "kz": "kleiig zand, zandige klei en leem",
                     "zf": "zand fijn",
                     "zm": "zand midden",
                     "zg": "zand grof",
                     "g": "grind",
                     "sch": "schelpen"}

lithoklassen_index = {"a": 0,
                     "v": 1,
                     "k": 2,
                     "kz": 3,
                     "zf": 5,
                     "zm": 6,
                     "zg": 7,
                     "g": 8,
                     "sch": 9}

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
    strat = re.sub(r"[a-z]+$", "", strat)

    return strat

def make_strat_short_name(strat):
    """
    Convert voxel strat code to code used in statistics tables.

    Examples:
    NUNAWA1  -> NAWA
    NUECge   -> EC
    NUNAWOgd -> NAWOgd
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

    return strat

#%%
# read in data

geotop_codes = pd.read_csv(path_geotop_codes)
regis_codes = pd.read_csv(path_regis_codes)
#df = pd.read_csv(path_sample_data, index_col=0)
results_litho = pd.read_csv(path_results_litho, index_col=0).reset_index()
results_strat_litho = pd.read_csv(path_results_strat_litho, index_col=0).reset_index()
results_facies_litho = pd.read_csv(path_results_facies_litho, index_col=0).reset_index()
results_litho_old = pd.read_csv(path_results_litho_old, index_col=0)
results_litho_old = results_litho_old.iloc[1:].reset_index()

#%%
# prepare old ff and ECs values
results_litho_old["ECs_25_mean_S_per_m"] = results_litho_old["ECs25_mean"].astype(float)*conversion_to_S_per_m
results_litho_old["ECs_25_std_S_per_m"] = results_litho_old["ECs25_std"].astype(float)*conversion_to_S_per_m

#%%
# prepare stratigraphy codes

geotop_codes = geotop_codes.rename(
    columns={
        "STR_UNIT_CD": "unit_cd",
        "VOXEL_NR": "unit_nr",
    }
)

regis_codes = regis_codes.rename(
    columns={
        "formation": "unit_cd",
        "user_nr": "unit_nr",
    }
)


geotop_codes = geotop_codes[
    ["unit_cd", "unit_nr", "facies"]
]

regis_codes = regis_codes[
    ["unit_cd", "unit_nr", "facies"]
]

strat_codes = pd.concat(
    [geotop_codes, regis_codes],
    ignore_index=True
)


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
    "lithoclass_id": [
        lithoklassen_index[k]
        for k in lithoklassen_naam.keys()
    ],
    "Lithoklasse_naam": list(lithoklassen_naam.values()),
})

lookup_table = (
    strat_codes.merge(df_litho_naam, how="cross")
)

lookup_table["strat_short_name"] = (
    lookup_table["unit_cd"]
    .apply(make_strat_short_name)
)

# raise warning if names are not unique after first simplification 
if len(lookup_table["strat_short_name"].unique()) != len(lookup_table["unit_cd"].unique()):
    raise ValueError(
        f"Aantal unieke strat_short_name ({len(lookup_table['strat_short_name'].unique())}) "
        f"is niet gelijk aan aantal unieke unit_cd codes ({len(lookup_table['unit_cd'].unique())})"
    )

lookup_table["strat_match"] = (
    lookup_table["strat_short_name"]
    .apply(simplify_strat_name)
)

lookup_table["mean_dist_ff"] = np.nan
lookup_table["mean_dist_surfcond"] = np.nan
lookup_table["std_dist_ff"] = np.nan
lookup_table["std_dist_surfcond"] = np.nan
lookup_table["dist_type"] = "NaN"
lookup_table["statistiek_literatuur"] = "NaN"
lookup_table["groepering_statistiek"] = "NaN"
lookup_table["n"] = np.nan

#%%
# create lookup table for monte carlo analysis

# =============================================================================
# 1: specifieke statistieken op stratigrafie-niveau
# =============================================================================

# # stratigrafien die binnen de lithoklassen duidelijk significant van elkaar verschillen en vanuit geologisch oogpunt logisch zijn om te onderscheiden
# special_stats = [
#     ("v", "NIHO", ["NIHO"]),
#     ("v", "NIBA", ["NIBA"]),
# ]

# for litho, strat in special_stats:
#     # haal waarden uit stratlitho resultaten tabel
#     strat_stats = results_strat_litho.loc[
#         (results_strat_litho["LITHOKLASSE_CD"] == litho)
#         & (results_strat_litho["strat_group"] == strat)
#     ].iloc[0]

#     mask = (
#         (lookup_table["LITHOKLASSE_CD"] == litho)
#         & (lookup_table["strat_match"] == strat)
#     )

#     lookup_table.loc[mask, [
#         "mean_dist_ff",
#         "mean_dist_surfcond",
#         "std_dist_ff",
#         "std_dist_surfcond"
#     ]] = [
#         strat_stats["mean_log_ff"],
#         strat_stats["mean_log_surfcond"],
#         strat_stats["std_log_ff"],
#         strat_stats["std_log_surfcond"],
#     ]

#     lookup_table.loc[mask, "dist_type"] = "lognorm"
#     lookup_table.loc[mask, "statistiek_literatuur"] = "statistiek"
#     lookup_table.loc[mask, "groepering_statistiek"] = (
#         f"{litho}-[{strat}]"
#     )


#%% 

# # =============================================================================
# # 2. Vul lookup-tabel met facies-binnen-litho statistieken
# # =============================================================================

# # vul vervolgens de lookup-tabel met de facies-binnen-litho statistieken, zodat voor de facies(groepen) die significant van elkaar verschillen de juiste statistiek wordt gebruikt.

# lookup_table = lookup_table.merge(
#     facies_lookup[
#         [
#             "LITHOKLASSE_CD",
#             "facies",
#             "facies_group",
#             "n",
#             "mean_log_ff",
#             "mean_log_surfcond",
#             "std_log_ff",
#             "std_log_surfcond",
#         ]
#     ],
#     left_on=["LITHOKLASSE_CD", "facies"],
#     right_on=["LITHOKLASSE_CD", "facies"],
#     how="left",
#     suffixes=("", "_stat")
# )

# # vullen waar match gevonden is
# mask = (lookup_table["mean_dist_ff"].isna() & lookup_table["mean_log_ff"].notna())

# lookup_table.loc[mask, "mean_dist_ff"] = lookup_table.loc[mask, "mean_log_ff"].astype(float)
# lookup_table.loc[mask, "mean_dist_surfcond"] = lookup_table.loc[mask, "mean_log_surfcond"].astype(float)
# lookup_table.loc[mask, "std_dist_ff"] = lookup_table.loc[mask, "std_log_ff"].astype(float)
# lookup_table.loc[mask, "std_dist_surfcond"] = lookup_table.loc[mask, "std_log_surfcond"].astype(float)
# lookup_table.loc[mask, "n"] = lookup_table.loc[mask, "n_stat"].astype(float)
# lookup_table.loc[mask, "dist_type"] = "lognorm"
# lookup_table.loc[mask, "statistiek_literatuur"] = "statistiek"



# lookup_table.loc[mask, "groepering_statistiek"] = (
#     lookup_table.loc[mask, "LITHOKLASSE_CD"]
#     + "-["
#     + lookup_table.loc[mask, "facies_group"]
#     + "]"
# )

# lookup_table = lookup_table.drop(
#     columns=[
#         "facies_group",
#         "mean_log_ff",
#         "mean_log_surfcond",
#         "std_log_ff",
#         "std_log_surfcond",
#         "n_stat"
#     ]
# )



#%%
# =============================================================================
# 3. Vul lookup-tabel met litho statistieken
# =============================================================================

# Bij ontbrekende facies binnen lithoklassen wordt de statistiek van de lithoklasse gebruikt. Dit is vooral om te zorgen dat het monte carlo scrip niet vastloopt op zeldzame cominaties van lithoklasse en facies

df_litho = results_litho_old.copy()

lookup_table = lookup_table.merge(
    df_litho[
        [
            "LITHOKLASSE_CD",
            "ffi_mean",
            "ffi_std",
            "ECs_25_mean_S_per_m",
            "ECs_25_std_S_per_m",
        ]
    ],
    left_on=["LITHOKLASSE_CD"],
    right_on=["LITHOKLASSE_CD"],
    how="left",
    suffixes=("", "_stat")
)

# vullen waar nog geen gegevens staan
mask = (lookup_table["mean_dist_ff"].isna() & lookup_table["ffi_mean"].notna())

lookup_table.loc[mask, "mean_dist_ff"] = lookup_table.loc[mask, "ffi_mean"].astype(float)
lookup_table.loc[mask, "mean_dist_surfcond"] = lookup_table.loc[mask, "ECs_25_mean_S_per_m"].astype(float)
lookup_table.loc[mask, "std_dist_ff"] = lookup_table.loc[mask, "ffi_std"].astype(float)
lookup_table.loc[mask, "std_dist_surfcond"] = lookup_table.loc[mask, "ECs_25_std_S_per_m"].astype(float)
lookup_table.loc[mask, "n"] = np.nan
lookup_table.loc[mask, "dist_type"] = "norm"
lookup_table.loc[mask, "statistiek_literatuur"] = "FRESHEM-Zeeland"
lookup_table.loc[mask, "groepering_statistiek"] = (lookup_table.loc[mask, "LITHOKLASSE_CD"])

lookup_table = lookup_table.drop(
    columns=[
        "ffi_mean",
        "ECs_25_mean_S_per_m",
        "ffi_std",
        "ECs_25_std_S_per_m",
    ]
)



#%%

#%%
# =============================================================================
# 4. vul ontbrekende lithoklasse aan met literatuur waarde
# =============================================================================

# lithoklasse_uit_literatuur = ["a", "g", "sch"]


# for litho in lithoklasse_uit_literatuur:

#     if litho == "g":
#         mean_dist_ff = mean_dist_ff_grind
#         mean_dist_surfcond = 0
#         std_dist_ff = 0
#         std_dist_surfcond = 0
#         source = "literatuur"
#         group = ""
#         n = np.nan
#     elif litho == "sch":
#         mean_dist_ff = mean_dist_ff_schelpen
#         mean_dist_surfcond = 0
#         std_dist_ff = 0
#         std_dist_surfcond = 0
#         source = "literatuur"
#         group = ""
#         n = np.nan
#     elif litho == "a": # for "antropogeen" use statistics of this lithoklasse as replacement for missing values
#         mean_dist_ff = results_litho.loc[results_litho["LITHOKLASSE_CD"] == replacement_litho, "mean_log_ff"].values[0]
#         mean_dist_surfcond = results_litho.loc[results_litho["LITHOKLASSE_CD"] == replacement_litho, "mean_log_surfcond"].values[0]
#         std_dist_ff = results_litho.loc[results_litho["LITHOKLASSE_CD"] == replacement_litho, "std_log_ff"].values[0]
#         std_dist_surfcond = results_litho.loc[results_litho["LITHOKLASSE_CD"] == replacement_litho, "std_log_surfcond"].values[0]
#         n = results_litho.loc[results_litho["LITHOKLASSE_CD"] == replacement_litho, "n"].values[0]
#         source = "statistiek"
#         group = f"{replacement_litho}"

#     mask = ((lookup_table["LITHOKLASSE_CD"] == litho) & (lookup_table["mean_dist_ff"].isna()))

#     lookup_table.loc[mask, [
#         "mean_dist_ff",
#         "mean_dist_surfcond",
#         "std_dist_ff",
#         "std_dist_surfcond",
#         "n"
#     ]] = [
#         mean_dist_ff,
#         mean_dist_surfcond,
#         std_dist_ff,
#         std_dist_surfcond,
#         n
#     ]

#     lookup_table.loc[mask, "dist_type"] = "norm"
#     lookup_table.loc[mask, "statistiek_literatuur"] = source
#     lookup_table.loc[mask, "groepering_statistiek"] = group


#%%

# verander volgorde van kolommen zodat deze beter leesbaar is in excel
column_order = [
    "unit_cd",
    "unit_nr",
    "facies",
    "LITHOKLASSE_CD",
    "Lithoklasse_naam",
    "lithoclass_id",
    "strat_short_name",
    "strat_match",
    "mean_dist_ff",
    "std_dist_ff",
    "mean_dist_surfcond",
    "std_dist_surfcond",
    "dist_type",
    "statistiek_literatuur",
    "groepering_statistiek",
    "n",
]

lookup_table = lookup_table[column_order]

lookup_table_geotop = lookup_table.loc[(lookup_table["unit_nr"] == 0) | (lookup_table["unit_nr"] >= 1000)]
lookup_table_regis = lookup_table.loc[(lookup_table["unit_nr"] > 0) & (lookup_table["unit_nr"] < 1000)]

# lookup_table.to_csv(f"{path_monte_carlo}/look_up_table_ff_ECs.csv", index=False)
# lookup_table_geotop.to_csv(f"{path_monte_carlo}/look_up_table_ff_ECs_geotop.csv", index=False)
# lookup_table_regis.to_csv(f"{path_monte_carlo}/look_up_table_ff_ECs_regis.csv", index=False)

# add units
# multi index for units as second row in excel

units = {
    "unit_cd": "[-]",
    "unit_nr": "[-]",
    "facies": "[-]",
    "LITHOKLASSE_CD": "[-]",
    "lithoclass_id": "[-]",
    "Lithoklasse_naam": "[-]",
    "strat_short_name": "[-]",
    "strat_match": "[-]",
    "mean_dist_ff": "[-]", # de gemeten waarden van de FF en ECs zijn eerst getransformeerd naar log10, daarna is van deze set de mean, std etc berekend. Na het trekken van de waardes voor de monte carlo anlyse moet de getrokken waarde is terug getransformeed worden (i.e. 10**(np.random.normal(mean_log_ff, std_log_ff)) 
    "std_dist_ff": "[-]",
    "mean_dist_surfcond": "[S/m]",  # zelfde verhaal als voor FF en daarna moet getrokken+teruggetransformeerde waarde nog omgerekend worden naar de juiste unit (van S/m naar naar mS/cm = terug getransformeerde waarde*1000/100)
    "std_dist_surfcond": "[S/m]",
    "dist_type": "[-]",
    "statistiek_literatuur": "[-]",
    "groepering_statistiek": "[-]",
    "n": "[-]",
}


lookup_excel = lookup_table.copy()
lookup_excel.columns = pd.MultiIndex.from_tuples(
    [(col, units.get(col, ""))
        for col in lookup_excel.columns])

lookup_excel_geotop = lookup_table_geotop.copy()
lookup_excel_geotop.columns = pd.MultiIndex.from_tuples(
    [(col, units.get(col, ""))
        for col in lookup_excel_geotop.columns])

lookup_excel_regis = lookup_table_regis.copy()
lookup_excel_regis.columns = pd.MultiIndex.from_tuples(
    [(col, units.get(col, ""))
        for col in lookup_excel_regis.columns])

# lookup_excel.to_excel(f"{path_monte_carlo}/look_up_table_ff_ECs.xlsx")
# lookup_excel_geotop.to_excel(f"{path_monte_carlo}/look_up_table_ff_ECs_geotop.xlsx")
# lookup_excel_regis.to_excel(f"{path_monte_carlo}/look_up_table_ff_ECs_regis.xlsx")

#lookup_excel.to_csv(f"{path_monte_carlo}/look_up_table_ff_ECs_FRESHEM_Zeeland.csv", index=False)
lookup_excel_geotop.to_csv(f"{path_monte_carlo}/look_up_table_ff_ECs_geotop_FRESHEM_Zeeland.csv", index=False)
lookup_excel_regis.to_csv(f"{path_monte_carlo}/look_up_table_ff_ECs_regis_FRESHEM_Zeeland.csv", index=False)

#%%