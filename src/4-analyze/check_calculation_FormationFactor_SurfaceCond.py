"""
Script to calculate formation factor and surface conductivity based on measurements. 
This calculation is a check of the determined parameters in Nextcloud\FreshEM (Projectfolder)\Team1-Fieldwork\05_Sampling&FFresults\20260126_tbl05_Measurementdata_Full.xlsx

Project: FRESHEM (11210255-005)
Author: Romee van Dam (Deltares)
Date: 10 March 2026
"""
#%% imports
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
import os

#%% paths and parameters
path_labresults = Path("data/3-input/lab_results")
fn_labresults = path_labresults / "20260304_tbl20_WPchloride_FFdata.xlsx"
fn_labresults_inc_grainsize = path_labresults / "20260126_tbl05_Measurementdata_Full.xlsx"
path_results = Path("data/4-output/ff_ecs_uncertainty")

os.chdir(r"c:\Users\dam_re\OneDrive - Stichting Deltares\Documents\Projecten\FRESHEM\scripts\formationfactors")

#%% definitions

def log_model(x, a, b):
    """ Model used for log-log fit:
    log(sigma_accent) = log(a * sigma_pore + b)"""

    return np.log(a * x + b)

#%% # prepare data

# read data
df_all = pd.read_excel(fn_labresults, 
    keep_default_na=False,
    na_values=["", " ", "NULL", "NaN"] # otherwise the formation of Naaldwijk (NA) is set as NaN
)
df = df_all.copy()

# Column names
pore_cols = [
    "SIP3a_PoreWaterCond_Sigmaw_S/m",
    "SIP3b_PoreWaterCond_Sigmaw_S/m",
    "SIP3c_PoreWaterCond_Sigmaw_S/m"
]

accent_cols = [
    "SIP3a_InPhaseCond_Sigmaaccent1Hz_S/m",
    "SIP3b_InPhaseCond_Sigmaaccent1Hz_S/m",
    "SIP3c_InPhaseCond_Sigmaaccent1Hz_S/m"
]




#%% fit log model for each sample

results = []

for idx, row in df.iterrows():

    # Get pore and accent conductivities as numeric arrays
    sigma_pore = pd.to_numeric(row[pore_cols], errors='coerce').values.astype(float)
    sigma_accent = pd.to_numeric(row[accent_cols], errors='coerce').values.astype(float)

    # Need at least 2 points for fitting
    if len(sigma_pore) < 2:
        results.append({
            "index": idx,
            "a_fit": np.nan,
            "FormationFactor_fit": np.nan,
            "SurfCond_fit": np.nan,
            "FormationFactor_SIP3": row.get("SIP3_FormationFactor_F_3W_unitless"),
            "SurfCond_SIP3": row.get("SIP3_SurfCond_Sigmas_3W_S/m")
        })
        continue

    # Take log of sigma_accent (left-hand side of model)
    y_log = np.log(sigma_accent)

    # Initial guesses
    p0 = (0.1, 0.001)

    # Fit the model
    try:
        popt, pcov = curve_fit(log_model, sigma_pore, y_log, p0=p0, maxfev=5000)
        a_fit, b_fit = popt
    except:
        a_fit, b_fit = np.nan, np.nan

    # Derived values
    FF_fit = 1 / a_fit if a_fit not in [0, np.nan] else np.nan
    b_surface = b_fit

    # Store results
    results.append({
        "index": idx,
        "alpha_fit": a_fit,
        "FormationFactor_fit": FF_fit,
        "SurfCond_fit": b_surface,
        "FormationFactor_SIP3": row.get("SIP3_FormationFactor_F_3W_unitless"),
        "SurfCond_SIP3": row.get("SIP3_SurfCond_Sigmas_3W_S/m")
    })


# Convert to DataFrame
res = pd.DataFrame(results).set_index("index")

# Merge back with identifying columns (optional)
output = pd.concat([df[["Boornummer", "LocSampleDepth_ID"]], res], axis=1)
print(output.head())

#%% output
output.to_excel(rf"{path_results}\SIP3_check_fitted_results.xlsx", index=False)



