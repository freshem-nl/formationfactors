#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

from scipy.integrate import quad
from pathlib import Path


def archie(poros, m):
    return poros**-m

assert abs(archie(0.35, 1.3) - 3.914) < 1e-3

def d_n(n):
    """
    Expected range of n samples from N(0,1):
        d_n = E[max - min]
    """

    def integrand(z):
        return (
            z
            * n
            * st.norm.pdf(z)
            * st.norm.cdf(z) ** (n - 1)
        )

    ezmax, _ = quad(
        integrand,
        -np.inf,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-10
    )

    return 2 * ezmax

def plot_dist(litho, F, dist, fn=None, title=None):
    plt.hist(F, 100, alpha=.5)
    ymin,ymax = plt.ylim()
    mu = dist.mean()
    p50 = dist.median()
    sigma = dist.std()
    confint = dist.interval(.68)
    plt.vlines(mu, ymin,ymax, ls="-", label=f"mu = {mu:.3f}")
    plt.vlines(p50, ymin,ymax, ls="-.", label=f"median = {p50:.3f}")
    plt.vlines(confint, ymin,ymax, ls=":", label=f"stdev = {sigma:.3f}")
    plt.ylim(ymin,ymax)
    plt.grid()
    plt.legend()
    ax2 = plt.twinx()
    X = np.linspace(0,np.max(F),1000)
    ax2.plot(X,dist.pdf(X))
    ymin2,ymax2 = plt.ylim()
    plt.ylim(0,ymax2)

    if title is None:
        plt.title(f"Estimated formation factor for {litho}")
    else: 
        plt.title(title)
        
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        plt.close()



#%% Paths

path_input = Path(
    r"C:\Users\dam_re\OneDrive - Stichting Deltares\Documents\Projecten"
    r"\FRESHEM\scripts\formationfactors\data\3-input"
)

path_output = Path("data/4-output/unc_propagation_shellsgravel")

fn_labresults_full = (
    path_input
    / "lab_results/20260126_tbl05_Measurementdata_Full.xlsx"
    ""
)



SIP5 = True # include (True) of exclude (False) SIP5

#%% prepare dataset

# create dataframe with ff and ECs

# read data
df_all = pd.read_excel(fn_labresults_full, 
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

# add SIP5 data to dataset
if SIP5 == True:
    df.loc[df[surfcond_col].isna() & (df["FF_SIP5"] == "Yes"), [surfcond_col,ff_col]] = df.loc[df[surfcond_col].isna() & (df["FF_SIP5"] == "Yes"), ["SIP5_SurfCond_Sigmas_S/m", "SIP5_FormationFactor_F5_unitless"]].values

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

df["log_FF"] = np.log(df[ff_col])
df["log_surfcond"] = np.log(df[surfcond_col])

# do not take AAOM (anthropogenic) for analysis 
df = df.loc[df["Stratigrafie"]!='AAOM'].copy()

# remove  MG and WG suffixes
df[strat_col] = df[strat_col].str.replace("-(MG|WG)", "", regex=True)
df[stratlitho_col] = df[stratlitho_col].str.replace("-(MG|WG)", "", regex=True)

#%% Input data

# Use measured m values for fine, medium and coarse sand
sand = df.loc[
    df["LITHOKLASSE_CD"].isin(["zf", "zm", "zg"])
].copy()

m_data = sand["CementationExponent_m_unitless"].dropna()

print(f"Number of m measurements: {len(m_data)}")
print(f"m mean:   {m_data.mean():.3f}")
print(f"m median: {m_data.median():.3f}")
print(f"m min:    {m_data.min():.3f}")
print(f"m max:    {m_data.max():.3f}")


#%% Fit lognormal distribution directly to measured m

# loc=0 because m must be > 0
shape_m, loc_m, scale_m = st.lognorm.fit(
    m_data,
    floc=0
)

mu_log_m = np.log(scale_m)
sigma_log_m = shape_m
msd_m = np.exp(sigma_log_m)

m_dist = st.lognorm(
    s=sigma_log_m,
    loc=0,
    scale=scale_m
)

print("\nLognormal distribution m:")
print(f"mu_log     = {mu_log_m:.3f}")
print(f"sigma_log  = {sigma_log_m:.3f}")
print(f"scale       = {scale_m:.3f}")
print(f"MSD         = {msd_m:.3f}")
print(f"mean        = {m_dist.mean():.3f}")
print(f"median      = {m_dist.median():.3f}")


#%% Porosity data

# Measured porosity ranges
poros_shells = [0.36, 0.502]
n_samples_shells = 14

poros_gravel = [0.245, 0.346]
n_samples_gravel = 16

characteristics = {
    "shells": {
        "poros": {
            "range": poros_shells,
            "n": n_samples_shells
        },
        "m": {
            "dist": m_dist
        }
    },
    "gravel": {
        "poros": {
            "range": poros_gravel,
            "n": n_samples_gravel
        },
        "m": {
            "dist": m_dist
        }
    }
}


#%% Main

if __name__ == "__main__":

    path_output.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Estimate lognormal porosity distributions from
    # observed min/max and number of measurements.
    #
    # X ~ LogNormal(mu, sigma)
    # Y = ln(X) ~ N(mu, sigma)
    #
    # mu is estimated from the geometric midpoint of the range.
    #
    # sigma is estimated from:
    #
    # sigma_hat = ln(xmax / xmin) / d_n
    #
    # where d_n is the expected range of n observations
    # from a standard normal distribution.
    # ----------------------------------------------------------

    for litho in ["shells", "gravel"]:

        n_samples = characteristics[litho]["poros"]["n"]
        xmin, xmax = characteristics[litho]["poros"]["range"]

        # Transform endpoints to log-space
        ymin = np.log(xmin)
        ymax = np.log(xmax)

        # Estimate mu from geometric midpoint
        mu_hat = 0.5 * (ymin + ymax)

        # Expected range of n standard-normal observations
        dn = d_n(n_samples)

        # Estimate sigma in log-space
        sigma_dn = (ymax - ymin) / dn

        # Corresponding lognormal distribution
        poros_dist = st.lognorm(
            s=sigma_dn,
            scale=np.exp(mu_hat)
        )

        characteristics[litho]["poros"]["dist"] = poros_dist

        print(
            f"\n{litho} porosity:"
            f"\nmu_log = {mu_hat:.3f}"
            f"\nsigma_log = {sigma_dn:.3f}"
            f"\nmedian = {poros_dist.median():.3f}"
        )


    #%% Monte Carlo uncertainty propagation

    rows = []

    n = 1_000_000

    for litho in ["shells", "gravel"]:

        # m sampled from lognormal distribution fitted
        # directly to measured fine, medium and coarse sand
        m = characteristics[litho]["m"]["dist"].rvs(n)

        # porosity sampled from distribution inferred
        # from observed porosity range
        poros = characteristics[litho]["poros"]["dist"].rvs(n)

        # Archie
        F = archie(poros, m)

        # Fit resulting FF distribution with lognormal
        shape, loc, scale = st.lognorm.fit(
            F,
            floc=0
        )

        fitted_dist = st.lognorm(
            s=shape,
            loc=0,
            scale=scale
        )

        print(
            f"\nFF {litho}:"
            f"\nsigma_log = {shape:.3f}"
            f"\nscale = {scale:.3f}"
            f"\nmu_log = {np.log(scale):.3f}"
            f"\nMSD = {np.exp(shape):.3f}"
        )

        rows.append(
            {
                "parameter": litho,
                "FF scale": round(scale, 1),
                "FF mu_log": round(np.log(scale), 2),
                "FF MSD": round(np.exp(shape), 2),
                "FF sigma_log": round(shape, 2),
            }
        )

        # Optional plot
        plot_dist(
            litho,
            F,
            fitted_dist,
            fn=path_output / f"FF_{litho}.png"
        )


    # Save results

    results_df = pd.DataFrame(rows)

    results_df.to_csv(
        path_output / "FF_shells_gravel.csv",
        index=False
    )

    print("\nResults:")
    print(results_df)

#%%

# old version Joost

# ## from information Ronald(measured porosities) and literature review on ranges of m (copilot)
# m_shells = [1.7, 3.] #copilot 
# poros_shells = [.36, .502]    # breder: [.36, .502] #[.44, .502]
# n_samples_shells = 14

# m_gravel = [1.4, 1.8] #copilot
# poros_gravel = [.245,.346]
# avg_poros_gravel = .304
# n_samples_gravel = 16

# characteristics = {"shells":{"m":{"range":m_shells, "n":n_samples_shells},
#                              "poros":{"range":poros_shells, "n":n_samples_shells},
#                              },
#                    "gravel":{"m":{"range":m_gravel, "n":n_samples_gravel},
#                              "poros":{"range":poros_gravel, "n":n_samples_gravel},
#                              },
#                     }

# path_output = Path("data/4-output/unc_propagation_shellsgravel")

# if __name__ == "__main__":

#     path_output.mkdir(parents=True, exist_ok=True)


#     # First step is to infer a distribution from observed ranges
#     # Let's say X ~ LogNormal(mu,sigma),
#     # Then Y = ln(X) ~ N(mu,sigma)
#     # mu is geometric midpoint of range
#     # d_n is de range van een standaardnormale steekproef: d_n = E[Z(n) - Z(1)], dat geeft: sigma_hat = ln(xmax/xmin) / d_n
#     # alternatief is Blom's approximation. Maakt weinig uit...

#     # transform endpoints to log-space
#     for litho in ["shells","gravel"]:
#         for prm in ["m", "poros"]:
#             n = characteristics[litho][prm]["n"]
#             xmin,xmax = characteristics[litho][prm]["range"]
#             # log-space bounds
#             ymin = np.log(xmin)
#             ymax = np.log(xmax)

#             # estimate mu from geometric midrange
#             mu_hat = 0.5 * (ymin + ymax)

#             # Blom approximation
#             # TODO: volgens mij moet de -CDF-1 positief zijn (dus CDF-1) in E(r,n) = -CDF-1 ((r - a) / (n - 2a+1))
#             z = st.norm.ppf((n - 0.375)/(n + 0.25))  # Blom's approximation is: E(r,n) = -CDF-1 ((r - a) / (n - 2a+1))   --> so, r expected largest order. rangnummer r van het maximum is n
#             # estimate lognormal parameters
#             sigma_blom = np.log(xmax/xmin)/(2*z) # geeft sigma_log (sigma van onderliggende normaal verdeling van log getransformeerde dataset)

#             # wat ook kan: integratie van d_n
#             dn = d_n(n)
#             sigma_dn = (
#                     ymax - ymin
#                 ) / dn
            
#             # corresponding distribution:
#             print(f"{litho},{prm}: {mu_hat}, {sigma_blom}, {sigma_dn}")
#             dist = st.lognorm(s=sigma_dn, scale=np.exp(mu_hat)) # st.lognorm(s=sigma_log, scale=np.exp(mu_log)) -> geeft waardes terug in de oorspronkeljke ruimte
#             characteristics[litho][prm]["dist"] = dist



#     #### Dan: MonteCarlo om F te karakteriseren
#     # draw n samples and calculate F
#     rows = []
#     n = 1000000
#     for litho in ["shells","gravel"]:
#         m = characteristics[litho]["m"]["dist"].rvs(n)
#         poros = characteristics[litho]["poros"]["dist"].rvs(n)
#         F = archie(poros, m)
#         res = st.lognorm.fit(F)
#         shape, loc, scale = res
#         fitted_dist = st.lognorm(*res)

#         print(f"FF {litho}: sigma_log = {shape}, scale or exp(mu_log) = {scale}, mu_log = {np.log(scale)}")

#         rows.append(
#             {
#                 "parameter": litho,
#                 "FF scale": round(scale,1),
#                 "FF mu_log": round(np.log(scale),1),
#                 "FF MSD": round(np.exp(shape),1),
#                 "FF sigma_log": round(shape,1),
#             }
#         )

#     results_df = pd.DataFrame(rows)

#     results_df.to_csv(f"{path_output}/FF_shells_gravel")

#     #%%
