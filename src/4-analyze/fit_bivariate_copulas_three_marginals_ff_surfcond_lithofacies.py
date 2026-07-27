
"""
Fit bivariate copula's voor FF en surface conductivity met drie marginale opties:

1. empirical  : empirische marginale verdelingen
2. lognormal  : log10(FF) en log10(sigma_s) normaal verdeeld, dus FF en sigma_s lognormaal
3. normal     : FF en sigma_s zelf normaal verdeeld

Voor elke optie worden copula's gefit voor:
- lithoklasse
- facies binnen lithoklasse, optioneel met samengevoegde faciesgroepen

Outputs:
- CSV met fit summary per marginale optie en groep
- CSV met simulaties per marginale optie en groep
- plots observaties vs copula-simulaties
- QQ-plots voor normale en lognormale aanname
- histogrammen observaties vs simulaties

Benodigd:
    pixi add scipy pyvinecopulib openpyxl

Project: FRESHEM
Auteur: Romee van Dam / scriptvoorstel Copilot
"""
#%%
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata, kendalltau, probplot
import pyvinecopulib as pv

#%%

# =============================================================================
# Instellingen
# =============================================================================

# run from basedir, assuming script resides in subdir of src/
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

path_labresults = Path("data/3-input/lab_results")
fn_labresults = path_labresults / "20260304_tbl20_WPchloride_FFdata.xlsx"

path_results = Path("data/4-output/ff_ecs_uncertainty/copula_results_three_marginals_low_values_corrected")
path_results.mkdir(exist_ok=True, parents=True)

# kolommen uit je bestaande script
ff_col = "SIP3_FormationFactor_F_3W_unitless"
surfcond_col = "SIP3_SurfCond_Sigmas_3W_S/m"
litho_col = "LITHOKLASSE_CD"
strat_col = "Stratigrafie"
stratlitho_col = "StratLithoklasse"
facies_col = "facies"

# welke marginale opties wil je draaien?
marginal_options = ["empirical", "lognormal", "normal"]

# algemene instellingen
min_n = 10
n_sim = 5000
random_seed = 1
make_plots = True
use_manual_lithofacies_groups = True
plot_log10_space = True # True = alle obs-vs-sim plots in log10-ruimte voor vergelijkbaarheid

# bij normal kunnen negatieve FF of sigma_s simulaties ontstaan; voor log10-plots worden die niet geplot
keep_negative_normal_simulations = True

#%%
# =============================================================================
# prepare input data
# =============================================================================

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


## clean up data

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

df = df.loc[df["Remarks"]!= "too slow σs_3W"]



#%%
# =============================================================================
# prepare facies groups
# =============================================================================


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

#%%
# =============================================================================
# Handmatige faciesgroepen binnen lithoklasse
# Pas deze lijst aan op basis van je Dunn-resultaten.
# =============================================================================

manual_lithofacies_groups = [
    {"lithoklasse": "kz", "facies": ["eolisch", "glaciaal"], "group": "eolisch+glaciaal"},
    {"lithoklasse": "kz", "facies": ["fluviatiel", "marien"], "group": "fluviatiel+marien"},
    {"lithoklasse": "zf", "facies": ["eolisch", "fluviatiel"], "group": "eolisch+fluviatiel"}, # TODO: fluviatiel had ook bij glaciaal + marien gezet kunnen worden
    {"lithoklasse": "zf", "facies": ["glaciaal", "marien"], "group": "glaciaal+marien"},
    {"lithoklasse": "zm", "facies": ["eolisch", "fluviatiel", "glaciaal"], "group": "eolisch+fluviatiel+glaciaal"},

]

#%%
# =============================================================================
# Basisfuncties
# =============================================================================


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_+.-]+", "_", str(value))


def clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Voorkom exact 0 of 1, want copula's en ppf's kunnen daar problemen mee hebben."""
    return np.clip(np.asarray(p, dtype=float), eps, 1 - eps)


def pseudo_obs(values: np.ndarray) -> np.ndarray:
    """Empirische CDF via ranks: waarden worden standaard-uniforme pseudo-observaties."""
    values = np.asarray(values, dtype=float)
    return rankdata(values, method="average") / (len(values) + 1.0) # method=avegare for ties because of even n (should have the same rank) ;n+1 because U=1 gives problems (norm.ppf(1) =infinity+) within domain [0,1]


def empirical_ppf(u: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Empirische inverse CDF."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    p = (np.arange(1, n + 1) - 0.5) / n
    return np.interp(u, p, values, left=values[0], right=values[-1])


def flatten_parameters(cop: pv.Bicop) -> str:
    """help functie to get a clean csv (in string format) with data from copula parameters"""
    pars = np.asarray(cop.parameters).ravel()
    if pars.size == 0:
        return ""
    return ";".join([f"{p:.6g}" for p in pars])


# =============================================================================
# Data voorbereiden
# =============================================================================


# def read_and_prepare_data() -> pd.DataFrame:
#     df_all = pd.read_excel(
#         fn_labresults,
#         keep_default_na=False,
#         na_values=["", " ", "NULL", "NaN"],
#     )

#     df = df_all.loc[df_all["Type_name"] == "FF_Disturbed"].copy()

#     missing_litho = ~df[litho_col].notna()
#     if missing_litho.any():
#         print("Warning: sommige samples missen lithoklasse; probeer waarde uit StratLithoklasse te halen.")
#         for idx in df.loc[missing_litho].index:
#             val = df.loc[idx, stratlitho_col]
#             if pd.notna(val):
#                 df.loc[idx, litho_col] = str(val)[-2:]

#     df[ff_col] = pd.to_numeric(df[ff_col], errors="coerce")
#     df[surfcond_col] = pd.to_numeric(df[surfcond_col], errors="coerce")

#     df = df.loc[
#         df[ff_col].notna()
#         & df[surfcond_col].notna()
#         & df[litho_col].notna()
#         & df[strat_col].notna()
#     ].copy()

#     # voor lognormale optie zijn alleen positieve waarden bruikbaar;
#     # omdat FF en sigma_s fysisch positief zijn, nemen we niet-positieve waarden niet mee.
#     df = df.loc[(df[ff_col] > 0) & (df[surfcond_col] > 0)].copy()

#     # anthropogeen niet meenemen, conform je bestaande analyse
#     df = df.loc[df[strat_col] != "AAOM"].copy()

#     # MG/WG suffix verwijderen
#     df[strat_col] = df[strat_col].astype(str).str.replace("-(MGWG)", "", regex=True)
#     df[stratlitho_col] = df[stratlitho_col].astype(str).str.replace("-(MGWG)", "", regex=True)

#     df[facies_col] = df[strat_col].apply(assign_facies)
#     df = df.loc[df[facies_col].notna()].copy()

#     df["log10_FF"] = np.log10(df[ff_col])
#     df["log10_surfcond"] = np.log10(df[surfcond_col])

#     return df


def add_lithofacies_group(df: pd.DataFrame, manual_groups: list[dict] | None) -> pd.DataFrame:
    manual_groups = manual_groups or []
    df = df.copy()

    def get_group(row):
        litho = row[litho_col]
        facies = row[facies_col]
        for g in manual_groups:
            if litho == g["lithoklasse"] and facies in g["facies"]:
                return g["group"]
        return facies

    df["facies_group"] = df.apply(get_group, axis=1)
    return df


# =============================================================================
# Marginale modellen
# =============================================================================


def get_marginal_params(sub: pd.DataFrame, marginal: str) -> dict:
    """Schat parameters van de gekozen marginale verdelingen."""
    if marginal == "empirical":
        return {
            "ff_values": sub[ff_col].to_numpy(),
            "ecs_values": sub[surfcond_col].to_numpy(),
        }

    if marginal == "lognormal":
        return {
            "mu_log10_ff": sub["log10_FF"].mean(),
            "sd_log10_ff": sub["log10_FF"].std(ddof=1), # bessel-correction, delta deree of freedom is 1 (-> n-1) because sample group not whole population (ddof =0)
            "mu_log10_ecs": sub["log10_surfcond"].mean(),
            "sd_log10_ecs": sub["log10_surfcond"].std(ddof=1),
        }

    if marginal == "normal":
        return {
            "mu_ff": sub[ff_col].mean(),
            "sd_ff": sub[ff_col].std(ddof=1),
            "mu_ecs": sub[surfcond_col].mean(),
            "sd_ecs": sub[surfcond_col].std(ddof=1),
        }

    raise ValueError(f"Onbekende marginal: {marginal}")


def transform_to_uv(sub: pd.DataFrame, marginal: str, params: dict) -> np.ndarray:
    """Transformeer observaties naar U,V in [0,1] op basis van de marginale keuze."""
    if marginal == "empirical":
        u = pseudo_obs(sub[ff_col].to_numpy())
        v = pseudo_obs(sub[surfcond_col].to_numpy())

    elif marginal == "lognormal":
        u = norm.cdf(sub["log10_FF"], loc=params["mu_log10_ff"], scale=params["sd_log10_ff"])
        v = norm.cdf(sub["log10_surfcond"], loc=params["mu_log10_ecs"], scale=params["sd_log10_ecs"])

    elif marginal == "normal":
        u = norm.cdf(sub[ff_col], loc=params["mu_ff"], scale=params["sd_ff"])
        v = norm.cdf(sub[surfcond_col], loc=params["mu_ecs"], scale=params["sd_ecs"])

    else:
        raise ValueError(f"Onbekende marginal: {marginal}")

    return np.column_stack((clip_prob(u), clip_prob(v)))


def inverse_transform_from_uv(uv: np.ndarray, marginal: str, params: dict) -> pd.DataFrame:
    """Transformeer copula-samples U,V terug naar FF en sigma_s."""
    u = clip_prob(uv[:, 0])
    v = clip_prob(uv[:, 1])

    if marginal == "empirical":
        ff_sim = empirical_ppf(u, params["ff_values"])
        ecs_sim = empirical_ppf(v, params["ecs_values"])

    elif marginal == "lognormal":
        log_ff_sim = norm.ppf(u, loc=params["mu_log10_ff"], scale=params["sd_log10_ff"])
        log_ecs_sim = norm.ppf(v, loc=params["mu_log10_ecs"], scale=params["sd_log10_ecs"])
        ff_sim = 10 ** log_ff_sim
        ecs_sim = 10 ** log_ecs_sim

    elif marginal == "normal":
        ff_sim = norm.ppf(u, loc=params["mu_ff"], scale=params["sd_ff"])
        ecs_sim = norm.ppf(v, loc=params["mu_ecs"], scale=params["sd_ecs"])

    else:
        raise ValueError(f"Onbekende marginal: {marginal}")

    out = pd.DataFrame({
        "u_FF": u,
        "v_surfcond": v,
        "FF": ff_sim,
        "surfcond_S_m": ecs_sim,
    })

    # logkolommen alleen waar positief; bij normale marginals kunnen negatieve simulaties ontstaan
    out["log10_FF"] = np.where(out["FF"] > 0, np.log10(out["FF"]), np.nan)
    out["log10_surfcond"] = np.where(out["surfcond_S_m"] > 0, np.log10(out["surfcond_S_m"]), np.nan)
    out["inv_FF"] = np.where(out["FF"] != 0, 1 / out["FF"], np.nan)

    if marginal == "normal" and not keep_negative_normal_simulations:
        out = out.loc[(out["FF"] > 0) & (out["surfcond_S_m"] > 0)].copy()

    return out


# =============================================================================
# Plotfuncties
# =============================================================================


def plot_observed_vs_simulated(sub, sim, marginal, group_type, group_label):
    path_figs = path_results / "plots_observed_vs_simulated" / marginal
    path_figs.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    if plot_log10_space:
        x_obs = sub["log10_FF"]
        y_obs = sub["log10_surfcond"]
        sim_plot = sim.dropna(subset=["log10_FF", "log10_surfcond"])
        x_sim = sim_plot["log10_FF"]
        y_sim = sim_plot["log10_surfcond"]
        xlabel = "log10 formation factor (FF)"
        ylabel = "log10 surface conductivity sigma_s (S/m)"
    else:
        x_obs = sub[ff_col]
        y_obs = sub[surfcond_col]
        x_sim = sim["FF"]
        y_sim = sim["surfcond_S_m"]
        xlabel = "formation factor (FF)"
        ylabel = "surface conductivity sigma_s (S/m)"

    ax.scatter(x_obs, y_obs, s=28, c="green", alpha=0.8, label="observaties", edgecolor="none")
    ax.scatter(x_sim, y_sim, s=8, c="red", alpha=0.25, label="copula-simulaties", edgecolor="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{marginal} | {group_label}") # f"{marginal} | {group_type}: {group_label}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    fn = path_figs / f"obs_vs_sim_{safe_name(group_label)}.png" # path_figs / f"obs_vs_sim_{safe_name(group_type)}_{safe_name(group_label)}.png"
    fig.savefig(fn, dpi=200)
    plt.close(fig)


def plot_histograms(sub, sim, marginal, group_type, group_label):
    path_figs = path_results / "plots_histograms" / marginal
    path_figs.mkdir(exist_ok=True, parents=True)

    variables = [
        (ff_col, "FF", "formation factor (FF)"),
        (surfcond_col, "surfcond_S_m", "surface conductivity sigma_s (S/m)"),
        ("log10_FF", "log10_FF", "log10 formation factor (FF)"),
        ("log10_surfcond", "log10_surfcond", "log10 surface conductivity sigma_s (S/m)"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()

    for ax, (obs_col, sim_col, label) in zip(axs, variables):
        obs = sub[obs_col].dropna()
        simv = sim[sim_col].dropna()
        if len(obs) == 0 or len(simv) == 0:
            ax.set_title(f"{label}: geen geldige waarden")
            continue
        ax.hist(simv, bins=35, alpha=0.45, color="red", density=True, label="simulaties")
        ax.hist(obs, bins=min(15, max(5, len(obs) // 2)), alpha=0.65, color="green", density=True, label="observaties")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.grid(True)
        ax.legend()

    fig.suptitle(f"{marginal} | {group_type}: {group_label}")
    fig.tight_layout()
    fn = path_figs / f"hist_{safe_name(group_type)}_{safe_name(group_label)}.png"
    fig.savefig(fn, dpi=200)
    plt.close(fig)


def plot_qq_for_group(sub, group_type, group_label):
    """QQ-plots: normaal in originele ruimte versus normaal in log10-ruimte."""
    path_figs = path_results / "plots_qq"
    path_figs.mkdir(exist_ok=True, parents=True)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    probplot(sub[ff_col].dropna(), dist="norm", plot=axs[0, 0])
    axs[0, 0].set_title("FF normaal")

    probplot(sub[surfcond_col].dropna(), dist="norm", plot=axs[0, 1])
    axs[0, 1].set_title("sigma_s normaal")

    probplot(sub["log10_FF"].dropna(), dist="norm", plot=axs[1, 0])
    axs[1, 0].set_title("log10(FF) normaal = FF lognormaal")

    probplot(sub["log10_surfcond"].dropna(), dist="norm", plot=axs[1, 1])
    axs[1, 1].set_title("log10(sigma_s) normaal = sigma_s lognormaal")

    for ax in axs.ravel():
        ax.grid(True)

    fig.suptitle(f"QQ-plots | {group_type}: {group_label}")
    fig.tight_layout()
    fn = path_figs / f"qq_{safe_name(group_type)}_{safe_name(group_label)}.png"
    fig.savefig(fn, dpi=200)
    plt.close(fig)


def plot_uv_space(uv_obs, uv_sim, marginal, group_type, group_label):
    """Plot in copula-ruimte: hiermee beoordeel je alleen de afhankelijkheidsstructuur."""
    path_figs = path_results / "plots_uv_space" / marginal
    path_figs.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(uv_obs[:, 0], uv_obs[:, 1], s=28, c="green", alpha=0.8, label="observaties U,V", edgecolor="none")
    ax.scatter(uv_sim[:, 0], uv_sim[:, 1], s=8, c="red", alpha=0.25, label="simulaties U,V", edgecolor="none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("U_FF")
    ax.set_ylabel("V_sigma_s")
    ax.set_title(f"Copula-ruimte | {marginal} | {group_type}: {group_label}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    fn = path_figs / f"uv_{safe_name(group_type)}_{safe_name(group_label)}.png"
    fig.savefig(fn, dpi=200)
    plt.close(fig)


# =============================================================================
# Copula-fit
# =============================================================================


def fit_one_copula(sub, marginal, group_type, group_label, seed):
    sub = sub[[ff_col, surfcond_col, "log10_FF", "log10_surfcond", litho_col, facies_col]].dropna().copy()
    n = len(sub)

    if n < min_n:
        return {
            "marginal": marginal,
            "group_type": group_type,
            "group_label": group_label,
            "n": n,
            "status": f"skipped: n < {min_n}",
        }, None

    params = get_marginal_params(sub, marginal)

    # als sd 0 is, kan norm.cdf/ppf niet goed werken
    if marginal == "lognormal" and (params["sd_log10_ff"] == 0 or params["sd_log10_ecs"] == 0):
        return {"marginal": marginal, "group_type": group_type, "group_label": group_label, "n": n, "status": "skipped: sd=0"}, None
    if marginal == "normal" and (params["sd_ff"] == 0 or params["sd_ecs"] == 0):
        return {"marginal": marginal, "group_type": group_type, "group_label": group_label, "n": n, "status": "skipped: sd=0"}, None

    uv_obs = transform_to_uv(sub, marginal, params)

    cop = pv.Bicop()
    cop.select(uv_obs)

    uv_sim = cop.simulate(n=n_sim, seeds=[seed])
    sim = inverse_transform_from_uv(uv_sim, marginal, params)
    sim.insert(0, "sim_id", np.arange(len(sim)))
    sim.insert(0, "group_label", group_label)
    sim.insert(0, "group_type", group_type)
    sim.insert(0, "marginal", marginal)

    tau_log, tau_log_p = kendalltau(sub["log10_FF"], sub["log10_surfcond"])
    tau_raw, tau_raw_p = kendalltau(sub[ff_col], sub[surfcond_col])

    summary = {
        "marginal": marginal,
        "group_type": group_type,
        "group_label": group_label,
        "n": n,
        "status": "ok",
        "copula_family": str(cop.family).split(".")[-1],
        "parameters": flatten_parameters(cop),
        "loglik": cop.loglik(uv_obs),
        "aic": cop.aic(),
        "bic": cop.bic(),
        "kendall_tau_log_values": tau_log,
        "kendall_tau_log_p_value": tau_log_p,
        "kendall_tau_raw_values": tau_raw,
        "kendall_tau_raw_p_value": tau_raw_p,
        "mean_FF": sub[ff_col].mean(),
        "std_FF": sub[ff_col].std(ddof=1),
        "mean_surfcond": sub[surfcond_col].mean(),
        "std_surfcond": sub[surfcond_col].std(ddof=1),
        "mean_log10_FF": sub["log10_FF"].mean(),
        "std_log10_FF": sub["log10_FF"].std(ddof=1),
        "mean_log10_surfcond": sub["log10_surfcond"].mean(),
        "std_log10_surfcond": sub["log10_surfcond"].std(ddof=1),
        "n_negative_FF_sim": int((sim["FF"] <= 0).sum()),
        "n_negative_surfcond_sim": int((sim["surfcond_S_m"] <= 0).sum()),
    }

    if make_plots:
        plot_observed_vs_simulated(sub, sim, marginal, group_type, group_label)
        plot_histograms(sub, sim, marginal, group_type, group_label)
        plot_uv_space(uv_obs, uv_sim, marginal, group_type, group_label)
        # QQ-plots hoeven niet per marginal opnieuw; maar overschrijven is niet erg.
        plot_qq_for_group(sub, group_type, group_label)

    return summary, sim


def fit_copulas_by_group(df, marginal, group_cols, group_type):
    summaries = []
    simulations = []

    grouped = df.groupby(group_cols, dropna=True)
    for i, (keys, sub) in enumerate(grouped):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_label = " | ".join([f"{col}={val}" for col, val in zip(group_cols, keys)])
        print(f"Fit {marginal} copula voor {group_type}: {group_label} (n={len(sub)})")

        summary, sim = fit_one_copula(
            sub=sub,
            marginal=marginal,
            group_type=group_type,
            group_label=group_label,
            seed=random_seed + i,
        )
        summaries.append(summary)
        if sim is not None:
            simulations.append(sim)

    summary_df = pd.DataFrame(summaries)
    sim_df = pd.concat(simulations, ignore_index=True) if simulations else pd.DataFrame()
    return summary_df, sim_df

#%%
# =============================================================================
# Main
# =============================================================================


# def main():
# df = read_and_prepare_data()

groups = manual_lithofacies_groups if use_manual_lithofacies_groups else None
df = add_lithofacies_group(df, groups)
df.to_csv(path_results / "input_data_with_copula_groups.csv", index=False)

all_summaries = []
all_sims = []

for marginal in marginal_options:
    print("\n" + "=" * 80)
    print(f"Marginale optie: {marginal}")
    print("=" * 80)

    # 1. per lithoklasse
    summary_litho, sim_litho = fit_copulas_by_group(
        df=df,
        marginal=marginal,
        group_cols=[litho_col],
        group_type="lithoklasse",
    )

    # 2. facies binnen lithoklasse
    summary_lithofacies, sim_lithofacies = fit_copulas_by_group(
        df=df,
        marginal=marginal,
        group_cols=[litho_col, "facies_group"],
        group_type="facies_binnen_lithoklasse",
    )

    # per marginale optie apart opslaan
    summary_litho.to_csv(path_results / f"copula_fit_summary_lithoklasse_{marginal}.csv", index=False)
    summary_lithofacies.to_csv(path_results / f"copula_fit_summary_lithofacies_{marginal}.csv", index=False)

    if not sim_litho.empty:
        sim_litho.to_csv(path_results / f"simulated_samples_lithoklasse_{marginal}.csv", index=False)
    if not sim_lithofacies.empty:
        sim_lithofacies.to_csv(path_results / f"simulated_samples_lithofacies_{marginal}.csv", index=False)

    all_summaries.extend([summary_litho, summary_lithofacies])
    if not sim_litho.empty:
        all_sims.append(sim_litho)
    if not sim_lithofacies.empty:
        all_sims.append(sim_lithofacies)

# gecombineerde bestanden
pd.concat(all_summaries, ignore_index=True).to_csv(path_results / "copula_fit_summary_all_marginals.csv", index=False)
if all_sims:
    pd.concat(all_sims, ignore_index=True).to_csv(path_results / "simulated_samples_all_marginals.csv", index=False)

print("\nKlaar. Resultaten staan in:", path_results)


# if __name__ == "__main__":
#     main()
#%%