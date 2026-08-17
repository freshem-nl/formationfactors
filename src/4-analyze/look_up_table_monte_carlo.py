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

#%% 
# paths and parameters

# run from basedir, assuming script resides in subdir of src/
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

path_input = Path("data/3-input")
path_strat_codes = "data/1-external/REF_GTP_STR_UNIT.csv"
path_sample_data = f"{path_input}/20260304_tbl20_WPchloride_FFdata_with_facies.csv"
path_monte_carlo =Path("data/4-output/ff_ecs_uncertainty/for_monte_carlo")

path_monte_carlo.mkdir(exist_ok=True, parents=True)

#%%
# read in data

strat_codes = pd.read_csv(path_strat_codes, index_col=0).reset_index()
df = pd.read_csv(path_sample_data, index_col=0)

lithoklassen_naam = {"a": "antropogeen",
                     "v": "organisch materiaal (veen)",
                     "k": "klei",
                     "kz": "kleiig zand, zandige klei en leem",
                     "zf": "zand fijn",
                     "zm": "zand midden",
                     "zg": "zand grof",
                     "g": "grind",
                     "sch": "schelpen"}

#%% 
# prepare facies groups

facies_list = ['marien' , 'fluviatiel', 'glaciaal', 'eolisch', 'organisch', 'rest']

marien_codes = ['NAWA', 'NAWO', 'NAZA', 'NAWOBE', 'EE', 'OO', 'MS', 'OOSP', 'BR', 'WAWO' ]

fluviatiel_codes = ['URTY', 'URVE', 'AP', 'BXSI', 'UR', 'PZ', 'EC', 'ST', 'WA', 'KK', 'KW' ]

glaciaal_codes = ['DRGI', 'DRGIGA', 'PENI', 'PE', 'DRUI'] 

eolisch_codes = ['BX', 'DN', 'BXWI', 'BXKO', 'NASC' ] 

organisch_codes = ['NIHO', 'NIBA', 'NI']

rest_codes = ['AAOM'] #TODO: 'NA'?

codes_per_facies = {
    "marien": marien_codes,
    "fluviatiel": fluviatiel_codes,
    "glaciaal": glaciaal_codes,
    "eolisch": eolisch_codes,
    "organisch": organisch_codes,
    "rest": rest_codes,
}

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


facies_map_nu = {}

for code, facies in facies_map.items():
    facies_map_nu[f"NU{code}"] = facies

short_strat_name = {}

for code, _ in facies_map.items():
    short_strat_name[f"NU{code}"] = code


def normalize_strat_code(code):
    """Normalize stratigraphy code before lookup."""
    if pd.isna(code):
        return np.nan
    return str(code).strip().upper()

#%%
df_litho_naam = pd.DataFrame({
    "LITHOKLASSE_CD": list(lithoklassen_naam.keys()),
    "Lithoklasse_naam": list(lithoklassen_naam.values())
})

lookup_table = (
    strat_codes[["STR_UNIT_CD", "VOXEL_NR"]]
    .rename(columns={"STR_UNIT_CD": "stratigrafie"})
    .merge(df_litho_naam, how="cross")
)

lookup_table["strat_short_name"] = lookup_table["stratigrafie"].apply(
    lambda x: short_strat_name.get(x, np.nan)
)

lookup_table["facies"] = lookup_table["stratigrafie"].apply(
    lambda x: facies_map_nu.get(normalize_strat_code(x), np.nan)
)


lookup_table["mean_ff"] = np.nan
lookup_table["mean_surfcond"] = np.nan
lookup_table["std_ff"] = np.nan
lookup_table["std_surfcond"] = np.nan
lookup_table["distribution_type"] = np.nan
lookup_table["statistiek_literatuur"] = np.nan
lookup_table["groepering_statistiek"] = np.nan


lookup_table.to_csv(f"{path_monte_carlo}/look_up_table_ff_ECs.csv")

#%%

#TODO: mappen/mergen van lookup table met uitkomst tabellen.
