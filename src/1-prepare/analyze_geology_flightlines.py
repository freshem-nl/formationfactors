"""
Script to make preliminary fresh-salt plots for regional meetings FRESHEM

- no uncertainty
- based on most-occurring lithology GeoTOP
- formation factors of FRESHEM Zeeland
- gw temp 11 degrees

"""
import imod
import xugrid as xu
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import nearest_points
import numba
# import zarr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import re
import sys
sys.path.append("src/4-analyze")
sys.path.append("4-analyze")
import plot_skytem_section_simple as skytem

def set_geo_type(s):
    """klei: 2, zand: 5, complex: 3"""
    geotypes = {"k":2, "z":5, "c":3, "b":1, "v":1}
    if s[-1] in "cqsb":
        zkc = "c"
    else:
        zkc = s[-2]
    return geotypes[zkc]

@numba.njit
def _sum_lithostrat_length(litho, strat, lithoindex, stratindex, include, out):
    nz, ny, nx = litho.shape
    # loop per voxel,
    # tel lithostrat categorie op waar include=True
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if include[y,x] and not np.isnan(litho[z,y,x]):
                    out[stratindex[int(strat[z,y,x])], lithoindex[int(litho[z,y,x])]] += 1

def lithostrat_length_geotop(geotop, stratname, lithoname, include=None):
    stratindex = np.zeros(max(stratname.index)+1, dtype=int)-1
    for i,v in stratname.reset_index()["unit_nl"].items():  # vertaaltabel stratcode -> stratindex
        stratindex[v] = i
    lithoindex = np.array([0,1,2,3,-1,4,5,6,7,8])  # vertaaltabel lithocode -> lithoindex  (alleen 4 mist)
    out = xr.DataArray(np.zeros((len(stratname),len(lithoname))), dims=["strat","litho"],coords={"strat":stratname,"litho":lithoname})
    if include is None:
        include = xr.ones_like(geotop["strat"].isel(z=0)).astype(bool)
    _sum_lithostrat_length(geotop["lithok"].values, geotop["strat"].values, lithoindex, stratindex, include.values, out.values)
    return out / 2.  # nvoxels to length in m
    
@numba.njit
def _sum_formation_length(top, bot, include, minbot, maxtop, out):
    nform, ny, nx = top.shape
    # loop per formatie,
    # kijk welke z coordinaten top en bot vallen
    # plak waarde daartussen
    # even niet moeilijk doen met overlap, laatste heeft geluk
    for f in range(nform):
        for y in range(ny):
            for x in range(nx):
                if include[y,x] and not np.isnan(top[f,y,x]) and top[f,y,x] > minbot and bot[f,y,x] < maxtop:
                    topi = min(maxtop, top[f,y,x])
                    boti = max(minbot, bot[f,y,x])
                    out[f] += (topi-boti)



def formation_length_regis(regis, include=None):
    out = xr.zeros_like(regis["formation"], dtype=float)
    if include is None:
        include = xr.ones_like(regis["top"].isel(formation=0))

    _sum_formation_length(regis["top"].values, regis["bot"].values, include.values, minbot=-150, maxtop=-50, out=out.values)
    return out

@numba.njit
def mode(a:float) -> float:
    b = np.unique(a)
    c = np.zeros_like(b)
    for i in range(len(b)):
        c[i] = (a==b[i]).sum()
    return b[np.argmax(c)]

# function to get most occurring lithology per rho top/bot
@numba.njit
def _get_litho_mode(rho, top, bot, litho, litho_top, litho_bot, out):
    nlay,ns = rho.shape
    for l in range(nlay):
        for s in range(ns):
            if not np.isnan(rho[l,s]):
                topi = top[l,s]
                boti = bot[l,s]
                b = (litho_bot < topi) & (litho_top > boti)
                if b.any():
                    lithoi = litho[s,b]
                    out[l,s] = mode(lithoi)

def litho_per_rho(rho, litho):
    out = xr.ones_like(rho) * np.nan
    _get_litho_mode(rho.values, rho["top"].values, rho["bottom"].values, litho.values, litho["top"].values, litho["bottom"].values, out.values)
    return out


path_geotop = "data/1-external/geotop/geotop_2026.nc"
geotopcsv = "data/1-external/geotop/GeoTOP_k_values_2.0.csv"
path_regis = "data/1-external/regis/REGIS_v2_2.nc"
path_cl = "data/1-external/cl-analyses/xyzv_analyses_gw_aggregate_csv.csv"
path_flightlines = Path("data/1-external/vlieglijnen")
path_depthbrack = "data/1-external/3dchloride/3dchloride_depthfreshbrack_mMSL_filtered.tif"
path_output = Path("data/4-output/lithostrat_op_vlieglijnen")
# path_profiles = "data/1-external/shapes/test.shp"

path_labresults = Path("data/3-input/lab_results")
fn_labresults = path_labresults / "20260304_tbl20_WPchloride_FFdata.xlsx"

leg_litho = "data/1-external/legends/lithology.leg"
leg_cl = "data/1-external/legends/chloride.leg"
max_ds = 40

# load data
geotop = xr.open_dataset(path_geotop)
geotop = geotop.reindex(y=geotop.y[::-1])  # flip over y
geotop.coords["y"] = geotop["y"] - 50.
geotop.coords["x"] = geotop["x"] + 50.
geotop["dx"] = 100.
geotop["dy"] = -100.
geotop = geotop.transpose("z","y","x")
geotopdf = pd.read_csv(geotopcsv, sep=";")
most_occurring = geotop["lithok"]
strat = geotop["strat"]
stratname = geotopdf.groupby("unit_nl")["unit_name"].first()
strat.attrs["stratname"] = stratname
lithoname = pd.Series(["a","v","k","kz","kz","zf","zm","zg","g","sch"])
lithoname = lithoname.loc[lithoname.index!=4]

regis = xr.open_dataset(path_regis)
dfcl = df = pd.read_csv(path_cl)
gdfcl = gpd.GeoDataFrame(dfcl, geometry=gpd.points_from_xy(x=dfcl["x"],y=dfcl["y"]), crs="epsg:28992")
# legends
gt_col,gt_lev,gt_lab = imod.visualize.read_imod_legend(leg_litho)
# cl_col,cl_lev,cl_lab = imod.visualize.read_imod_legend(leg_cl)
cl_lev = [150,300,500,1000,1500,3000,5000,10000,15000]
cl_col = ["#00007fff", "#0000faff", "#0058feff", "#00c4ffff", "#3cffbaff","#93ff63ff","#eaff0cff", "#ff9f00ff", "#ff3b00ff","#b60000ff"]

regis_geotopcode = np.vectorize(set_geo_type)(regis["formation"])

path_output.mkdir(exist_ok=True, parents=True)

# maak kaart van alle vlieglijnen
flines = None
for fn in path_flightlines.glob("*.shp"):
    fl = gpd.read_file(fn)
    flines = pd.concat((flines,fl))
flines.to_file(path_output / "vlieglijnen.gpkg")
da_flines_regis = imod.prepare.rasterize(flines, regis["top"].isel(formation=0, drop=True))
da_flines_gt = imod.prepare.rasterize(flines, geotop["lithok"].isel(z=0, drop=True))

# selecteer geotop en regis op vlieglijnen
geotop_lithostrat = lithostrat_length_geotop(geotop,stratname,lithoname,include=da_flines_gt.notnull())
gtl = geotop_lithostrat.to_pandas()
gtl.to_csv(path_output / "lithostrat_geotop.csv")
gtl.stack().sort_values(ascending=False).to_csv(path_output / "lithostrat_geotop_sorted.csv")

regis_frm_length = formation_length_regis(regis, include=da_flines_regis.notnull())
regis_frm_length.to_series().to_csv(path_output / "regis_frm_length.csv")
regis_frm_length.to_series().sort_values(ascending=False).to_csv(path_output / "regis_frm_length_sorted.csv")


# alleen in zout: selecteer waar cl > 150m oid
da_depthbrack = imod.rasterio.open(path_depthbrack)
regridder = xu.CentroidLocatorRegridder(source=da_depthbrack, target=da_flines_regis)
da_depthbrack = regridder.regrid(da_depthbrack)
include_regis = (da_flines_regis.notnull())&(da_depthbrack>-150)
regridder = xu.CentroidLocatorRegridder(source=da_depthbrack, target=da_flines_gt)
da_depthbrack = regridder.regrid(da_depthbrack)
include_gt = (da_flines_gt.notnull())&(da_depthbrack>-150)


geotop_lithostrat = lithostrat_length_geotop(geotop,stratname,lithoname,include=include_gt)
gtl = geotop_lithostrat.to_pandas()
gtl.to_csv(path_output / "lithostrat_geotop_insaline.csv")
gtl.stack().sort_values(ascending=False).to_csv(path_output / "lithostrat_geotop_insaline_sorted.csv")

regis_frm_length = formation_length_regis(regis, include=include_regis)
regis_frm_length.to_series().to_csv(path_output / "regis_frm_length_insaline.csv")
regis_frm_length.to_series().sort_values(ascending=False).to_csv(path_output / "regis_frm_length_insaline_sorted.csv")



##### wat hebben we nu al?
df = pd.read_excel(fn_labresults)
dfg = df.groupby(['Stratigrafie','LITHOKLASSE_CD']).count()["Boornummer"]
dfg.unstack().to_csv(path_output / "nsamples_nu.csv")

