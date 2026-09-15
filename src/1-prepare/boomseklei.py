"""In absence of samples of the Boomse klei, attempt to derive relation from resistivity vs measurements"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns
import scipy.stats as st
import sys
sys.path.append("src/1-prepare")
from mc_propagation_shellsgravel import archie, d_n, plot_dist

def match_closest(a, b):
    """
    For each value in a, return the closest value in b.

    Parameters
    ----------
    a : array-like, shape (n,)
    b : array-like, shape (m,)

    Returns
    -------
    ndarray, shape (n,)
        Closest values from b.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    idx = np.abs(a[:, None] - b).argmin(axis=1)
    return b[idx]


path_nextcloud = Path("c:/Users/delsman/Nextcloud/FreshEM (Projectfolder)/Team4-Chloride")   # read-only access
path_parquets = path_nextcloud / "Results_Monte_Carlo/ZLZ-C3_28-08-26"
fn_cl = path_parquets / "ZLZ-C3-3D-sharp_nmc500_chloride.parquet"
fn_regis = path_parquets / "ZLZ-C3-3D-sharp_nmc500_regis_litho_unitcd.parquet"
fn_resis = path_parquets / "ZLZ-C3-3D-sharp_nmc500_hem_sharp_rho_mean.parquet"
fn_geotop = path_parquets / "ZLZ-C3-3D-sharp_nmc500_geotop_litho_unitcd.parquet"
fn_cldata = path_nextcloud / "Validation_dataset/Validation_dataset_wp4_chloride_v1.2.csv"
path_output = Path("data/4-output/boomseklei")
path_output.mkdir(exist_ok=True,parents=True)

fact_ect = 1.39  # uitgegaam van 11 deg gw temp


gdf_resis = gpd.read_parquet(fn_resis)

try:
    rubo_regis_xy = gpd.read_parquet(path_output / "rubo_regis_xy.parquet")
except FileNotFoundError:
    # gdf_cl = gpd.read_parquet(fn_cl)
    # gdf_geotop = gpd.read_parquet(fn_geotop)
    gdf_regis = gpd.read_parquet(fn_regis)
    # df_cldata = pd.read_csv(fn_cldata)

    # clip z to 1 decimal
    gdf_resis["z"] = gdf_resis["z"].round(1)


    # shift REGIS by 25m to match up with resistivity
    gdf_regis["x"] += 25
    gdf_regis["y"] += 25
    gdf_regis["geometry"] = gpd.points_from_xy(x=gdf_regis["x"],y=gdf_regis["y"],z=gdf_regis["z"])

    # select resistivities from Boomse klei
    # first select boomse klei from regis
    rubo_regis = gdf_regis.loc[gdf_regis["unit_cd"]=="NMRUBOk1"]
    # match z values 
    rubo_regis_xy = rubo_regis.groupby(["x","y"])
    gdf_resis_xy = gdf_resis.groupby(["x","y"])

    def apply_group_xy(df1, df2=gdf_resis):
        try:
            x,y = df1.name
            z1 = df1["z"]
            z2 = df2.loc[(df2["x"]==x)&(df2["y"]==y), "z"]
            if len(z2):
                df1["z"] = match_closest(z1, z2)
            return df1
        except:
            print(df1, df1.name)
            raise


    rubo_regis_xy = rubo_regis_xy.apply(apply_group_xy)  # sloooow

    rubo_regis_xy["x"] = rubo_regis_xy.geometry.x
    rubo_regis_xy["y"] = rubo_regis_xy.geometry.y
    rubo_regis_xy.to_parquet(path_output / "rubo_regis_xy.parquet")

# now merge on xyz
rubo_regis_xyz = rubo_regis_xy.reset_index().set_index(["x","y","z"])
resis_xyz = gdf_resis.set_index(["x","y","z"])
rubo_merged = pd.merge(rubo_regis_xyz,resis_xyz,left_index=True,right_index=True)
rubo_merged = rubo_merged.reset_index()
sns.boxplot(data=rubo_merged, x="unit_cd", y="sharp_rho_mean")
plt.savefig(path_output / "boxplot_resis_boomseklei.png")
plt.close()

avg_resis = rubo_merged["sharp_rho_mean"].mean()   # Ohmm
p50_resis = rubo_merged["sharp_rho_mean"].median()  # Ohmm

resis_fitted_dist = st.lognorm(*st.lognorm.fit(rubo_merged["sharp_rho_mean"]))


# stel water is 16 g/L Cl
# dat is dan ECw van ?
# Cl = .36*ECw - .45
# ECw = (Cl + 0.45) / 0.36
ecw25 = (16.+.45)/.36  # = ~ 45.7 mS/cm # EC25
ecw11 = ecw25 / fact_ect

p50_cond = 10 / p50_resis # mS/cm

# So, apparent formation factor?? given ECs = 0
# F = ECw/ECb
F_app = ecw11 / p50_cond  # 8!!  zou voor zout water toch weinig moeten uitmaken... Waarom zo vreemd hoog?
print(f"F apparent: {F_app}")
# with ECb as average for clays
# F_app2 = ()

# er is uitgebreide informatie over de CEC van de Boomse klei, aanwijzing voor ECs?
# CEC tussen 10 en 30 meq/100g

# -> discussie met Jan, wat lijkt meest op Boomse klei?
# wat voor porositeit hebben we het over?
# F = 1/poros^m
# F = poros^-m
# poros = F^1/-m
m = np.linspace(1.8,3,10)
poros = F_app**(1/-m)
#  --> geeft porositeit tussen .3 en .5.
# >>> poros
# array([0.3031397 , 0.32914828, 0.35361251, 0.37660803, 0.39822128,
#        0.4185423 , 0.43766049, 0.45566202, 0.4726285 , 0.48863626])
# >>> m
# array([1.8       , 1.93333333, 2.06666667, 2.2       , 2.33333333,
#        2.46666667, 2.6       , 2.73333333, 2.86666667, 3.        ])
# Uit Deltares rapport 11211277-003-GEO-0001_v1.0-Laboratory testing of Boom clay samples for Lubricated Pile Installation field - (voor publicatie).pdf
# porositeit ~ 0.4, afgeleid van watergehalte van 30%
# dat zou een m van 2.5 betekenen, vrij standaard voor kleien


# Boomse klei smectite rich  Zeelmaekers E, Honty M, Derkowski A, et al. Qualitative and quantitative mineralogical composition of the Rupelian Boom Clay in Belgium. Clay Minerals. 2015;50(2):249-272. doi:10.1180/claymin.2015.050.2.08
# dus significante ECs verwacht
# ECs =~ 0.01 - 0.05S/m, op basis smectiet gehalte en CEC van ~15-30 meq/100g
ECs = 0.025 # S/m
ECs11 = (10 * ECs) / fact_ect
# correctie voor F_app met max ECs
# ECb = 1/F*ECw + ECs
# 1/FECw = (ECb - ECs)
# F = ECw / (ECb - ECs)
F_int = ecw11 / (p50_cond - ECs11)  # -> F van 9
# komt precies overeen met door copilot voorgestelde waarde obv verwachte m, gemeten porositeit, smectiet gehalte, CEC...


################ from estimation to uncertainty
def fit_lognormal_from_expert_range(
    lower,
    upper,
    p_lower=0.05,
    p_upper=0.95,
):
    """
    Fit a lognormal distribution from an expert-defined range.

    Assumes:
        lower = p_lower quantile
        upper = p_upper quantile

    Returns mu, sigma and scipy distribution.
    """

    z_lower = st.norm.ppf(p_lower)
    z_upper = st.norm.ppf(p_upper)

    sigma = (
        np.log(upper) - np.log(lower)
    ) / (z_upper - z_lower)

    mu = (
        np.log(lower)
        - z_lower * sigma
    )

    dist = st.lognorm(
        s=sigma,
        scale=np.exp(mu)
    )

    return mu, sigma, dist


# --------------------------------------------------
# For surface conductivity: expert range 0.01 - 0.05 S/m
# interpreted as 5th-95th percentile
# --------------------------------------------------

mu_ecs, sigma_ecs, dist_ecs = fit_lognormal_from_expert_range(
    lower=0.01,
    upper=0.05,
    p_lower=0.05,
    p_upper=0.95,
)

print(f"mu     = {mu_ecs:.3f}")
print(f"sigma  = {sigma_ecs:.3f}")

print(f"median = {dist_ecs.median():.4f}")
print(f"mean   = {dist_ecs.mean():.4f}")
print(f"std    = {dist_ecs.std():.4f}")

print(f"5%     = {dist_ecs.ppf(0.05):.4f}")
print(f"95%    = {dist_ecs.ppf(0.95):.4f}")




#### Dan: MonteCarlo om F te karakteriseren
# beschikbaar: eindig aantal resis metingen (~90000)
# gefitte distribution van resis
# hoe verlengen? --> trek uit gefitte distributie
# draw n samples for bulk cond and ECs and calculate F
rows = []
n = 1_000_000

cond_draw = 10 / resis_fitted_dist.rvs(n)  # van Ohmm naar mS/cm
ecs11_draw = (10 * dist_ecs.rvs(n)) / fact_ect   # van EC 25 S/m naar 11 graden mS/cm
F_int_draw = ecw11 / (cond_draw - ecs11_draw)   
# throw out if <  zero / > 50
F_int_draw = F_int_draw[F_int_draw > 0]
F_int_draw = F_int_draw[F_int_draw < 50]

res = st.lognorm.fit(F_int_draw)
ff_shape, ff_loc, ff_scale = res
ecs_shape, ecs_loc, ecs_scale = dist_ecs
fitted_dist_F = st.lognorm(*res)

# TODO: schrijf parameters weg

# print(f"FF Boomse klei: sigma_log = {shape}, scale or exp(mu_log) = {scale}, mu_log = {np.log(scale)}")

# plot_dist("Boomse klei",F=F_int_draw, dist=fitted_dist_F, fn=path_output / "FF_dist_boomseklei.png")
# plot_dist("Boomse klei",F=dist_ecs.rvs(n), dist=dist_ecs, fn=path_output / "ECs_dist_boomseklei.png")#, title="ECs for Boomse klei")


# rows.append(
#     {
#         "parameter": ["FF Boomse klei", "ECs Boomse klei"],
#         "FF scale": [round(scale,1),
#         "FF mu_log": round(np.log(scale),1),
#         "FF MSD": round(np.exp(shape),1),
#         "FF sigma_log": round(shape,1),
#     }
# )

# results_df = pd.DataFrame(rows)

# results_df.to_csv(f"{path_output}/FF_boomseklei.csv")
