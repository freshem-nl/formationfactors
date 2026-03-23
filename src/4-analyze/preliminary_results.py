"""
Script to make preliminary fresh-salt plots for regional meetings FRESHEM

- no uncertainty
- based on most-occurring lithology GeoTOP
- formation factors of FRESHEM Zeeland
- gw temp 11 degrees

"""
import imod
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
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

def get_bearing(line):
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[-1]

    # 2. Calculate delta X and Y
    dx = x2 - x1
    dy = y2 - y1

    # 3. Calculate angle in radians, then convert to degrees
    # np.arctan2(y, x) returns angle in radians
    angles = np.arctan2(dy, dx)
    degrees = np.degrees(angles)

    # 4. Normalize to compass bearing (0-360)
    # This maps 0 degrees to East, 90 to North, etc.
    # Use (90 - degrees) % 360 to map 0 to North.
    return (degrees + 360) % 360

def get_crs(fn):
    with open(fn, "r") as f:
        for line in f:
            if "epsg:" in line:
                crs = re.findall("(?:epsg:[0-9]+)", line)[0]
                break
    return crs


# funcs to round to 0.5 edges of geotop z  (centres at .25, .75)
@numba.njit
def up_to_z(x):
    return round(x * 2) / 2
@numba.njit
def down_to_z(x):
    return round(x* 2) / 2 - 0.5

@numba.njit
def _conform_regis_to_geotop(data, top, bot, z, out):
    nform, ns = data.shape
    # loop per formatie,
    # kijk welke z coordinaten top en bot vallen
    # plak waarde daartussen
    # even niet moeilijk doen met overlap, laatste heeft geluk
    for f in range(nform):
        for s in range(ns):
            topi = up_to_z(top[f,s])
            boti = down_to_z(bot[f,s])
            val = data[f,s]
            b = (z < topi) & (z > boti)
            out[b,s] = val

def regis_to_geotop(points_regis):

    maxtop = up_to_z(points_regis["top"].max().data)
    minbot = down_to_z(points_regis["bot"].min().data)
    z = np.arange(minbot - 0.25, maxtop+.25, 0.5)
    
    out = np.ones((len(z),len(points_regis["s"])))*np.nan
    out = xr.DataArray(out, dims=["z","s"], coords={"z":z, 
                                                    "s":points_regis["s"], 
                                                    "x":("s", points_regis["x"].data), 
                                                    "y":("s", points_regis["y"].data), 
                                                    "ds":("s", points_regis["ds"].data),
                                                    "top":("z", z+0.25),
                                                    "bot":("z", z-0.25),
                                                    })

    _conform_regis_to_geotop(points_regis["litho"].values, points_regis["top"].values, points_regis["bot"].values, z, out.values)
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


######## UIT FRESHEM ZEELAND
litho_ff = {0:4., 1:2.66, 2:4.1, 3:4.53, 5:5.98, 6:6.6, 7:5., 8:6.5, 9:5.}
apply_lithoff = np.vectorize(lambda x:litho_ff.get(x,np.nan))
litho_ECs = {0:0., 1:2.95, 2:2.97, 3:2.65, 5:1.61, 6:1.1, 7:0., 8:0., 9:0.} #NB = ECs values! in mS/cm
apply_lithoecs = np.vectorize(lambda x:litho_ECs.get(x,np.nan))
fact_ect = 1.39  # uitgegaam van 11 deg gw temp
###########################

path_geotop = "p:/gis-data/GEOTOP/geotop.zarr"
path_regis = "p:/gis-data/REGIS/REGIS2.2/REGIS_v2_2.nc"
path_profiles = Path("data/3-input/priority_lines")
path_output = Path("data/4-output/priority_lines_processed")
# path_profiles = "data/1-external/shapes/test.shp"
leg_litho = "data/1-external/legends/lithology.leg"
leg_cl = "data/1-external/legends/chloride.leg"
max_ds = 40

# load data
geotop = xr.open_zarr(path_geotop)
most_occurring = geotop["meest_waarschijnlijke_lithoklasse"]
regis = xr.open_dataset(path_regis)
# legends
gt_col,gt_lev,gt_lab = imod.visualize.read_imod_legend(leg_litho)
# cl_col,cl_lev,cl_lab = imod.visualize.read_imod_legend(leg_cl)
cl_lev = [150,300,500,1000,1500,3000,5000,10000,15000]
cl_col = ["#00007fff", "#0000faff", "#0058feff", "#00c4ffff", "#3cffbaff","#93ff63ff","#eaff0cff", "#ff9f00ff", "#ff3b00ff","#b60000ff"]

regis_geotopcode = np.vectorize(set_geo_type)(regis["formation"])

path_output.mkdir(exist_ok=True, parents=True)

# loop through profiles
lines = {}
lines_flipped = {}
for i,fn in enumerate(path_profiles.glob("*.xyz")):
    print(fn)
    df = skytem.read_skytem_xyz(fn)

    # extract crs from header
    crs = get_crs(fn)

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["X"], y=df["Y"]), crs=crs)
    gdf = gdf.to_crs("epsg:28992")
    gdf[["x_rd","y_rd"]] = gdf.get_coordinates()

    for line_no in gdf["LINE_NO"].unique():
        print(line_no)
        ########### HAAL RHO DATA PER LIJN  
        data = skytem.prepare_line_arrays(gdf, line_no)
        # maak van data een xarray
        ds = np.diff(data["lengths"])
        ds = np.append(ds, ds[-1])
        s = data["lengths"]
        rho = xr.DataArray(data["rho"], dims=["s","layer"], coords={"s":s, 
                                                                    "layer":np.arange(data["rho"].shape[1]),
                                                                    "ds":("s",ds),
                                                                    "top":(("s","layer"),(data["surface"]-data["depth_top"].T).T),
                                                                    "bottom":(("s","layer"),(data["surface"]-data["depth_bottom"].T).T),
                                                                    "doi":("s",data["surface"]-data["doi"]),
                                                                    }).transpose("layer","s")
        
        # if line_no == 680201:
        #     break

        ########### HAAL REGIS/GEOTOP DATA  
        gdfi = gdf.loc[gdf["LINE_NO"]==line_no]
        lines[line_no] = LineString(gdfi["geometry"].tolist())  # maak gisbestand vlieglijnen
        
        # flip als O-W of N-Z --> is tussen 45 en 135 graden
        bearing = get_bearing(lines[line_no])
        print(line_no, bearing)
        if (bearing > 135) and (bearing < 315):
            print("flip!")
            lines_flipped[line_no] = lines[line_no].reverse()
            gdfi = gdfi.iloc[::-1]
            rho = rho.reindex(s=rho.s[::-1])
        else:
            lines_flipped[line_no] = lines[line_no]
        new_bearing = get_bearing(lines_flipped[line_no])

        ### MISSENDE DATA
        # laat weg onder doi
        rho = rho.where(rho["top"] > rho["doi"])
        # laat weg waar de ds te groot is: missende data
        rho = rho.where(rho["ds"].shift(s=-1, fill_value=True) < max_ds)  #


        try:
            points_geotop = imod.select.points_values(most_occurring, x=gdfi["x_rd"], y=gdfi["y_rd"], out_of_bounds="ignore").compute()
            points_geotop = points_geotop.assign_coords({"s":("index", s), "ds":("index",ds)}).swap_dims(index="s")
            points_geotop = points_geotop.where(points_geotop>-1).dropna("z", how="all")
            points_regis = imod.select.points_values(regis, x=gdfi["x_rd"], y=gdfi["y_rd"], out_of_bounds="ignore").compute()
            points_regis = points_regis.assign_coords({"s":("index", s), "ds":("index",ds)}).swap_dims(index="s")
            points_regis["litho"] = (xr.ones_like(points_regis["kh"]).T * regis_geotopcode).T
            points_regis_geotop = regis_to_geotop(points_regis)
        except:
            print(f"Problem with line {line_no}")
            break
    
        points_combined = points_geotop.combine_first(points_regis_geotop)
        points_combined = points_combined.assign_coords({"layer":("z",np.arange(points_combined.shape[1], 0, -1))}).swap_dims(z="layer")
        points_combined = points_combined.rename({"bot":"bottom"})

        
        ####### Calculate Cl from rho
        rho["litho"] = litho_per_rho(rho, points_combined)
        ##### Get FF and ECs for lithos
        rho["FF"] = (("layer","s"),apply_lithoff(rho["litho"]))
        rho["ECs"] = (("layer","s"),apply_lithoecs(rho["litho"]))
        rho["ECb"] = 1. / rho * 10  # Ohmm -> 1/ohmm = S/m   /100=S/cm *1000 = mS.cm -> * 10...
        rho["ECb"] *= fact_ect  # temp conversion
        rho["ECw"] = rho["FF"]*rho["ECb"]-rho["ECs"]
        rho["Cl"] = 360.*rho["ECw"]-450
        rho["Cl"] = rho["Cl"].where((rho["Cl"] > 0) | rho["Cl"].isnull(), 0)  # set all negative values to 0

        ####### MAKE A PLOT
        smax = rho["s"].max().data / 1000.  # in km
        width = 24/13. * smax
        fig, axes = plt.subplots(3, 1, figsize=(width,16), sharex=True)

        # bottom: chloride
        fig, ax_cl = imod.visualize.cross_section(rho["Cl"], cl_col, cl_lev, fig=fig, ax=axes[0], kwargs_colorbar={"label":"Chloride (mg/L)", "whiten_triangles":False})
        ax_cl.set_title("Chlorideconcentratie (mg/L)")
        ax_cl.grid()

        # top: resistivity
        fig, ax_rho = imod.visualize.cross_section(rho, "RdYlBu", np.logspace(0,2.4,25), fig=fig, ax=axes[1], kwargs_colorbar={"label":"Resistivity (Ohmm)", "whiten_triangles":False})
        ax_rho.set_title("Resistiviteit (Ohmm)")
        ax_rho.grid()

        # mid: lithology
        fig, ax_litho = imod.visualize.cross_section(rho["litho"], gt_col, gt_lev, fig=fig, ax=axes[2], kwargs_colorbar={"format":mticker.FixedFormatter(gt_lab), "whiten_triangles":False})
        ax_litho.set_title("Lithologie")
        ax_litho.grid()

        fig.suptitle(f"Eerste resultaten vlieglijn {line_no}", fontsize=14)

        if (new_bearing < 45) | (new_bearing > 315):
            # W-O
            fig.text(0.04, 0.93, "W", fontsize=14, fontweight="bold")
            fig.text(0.83, 0.93, "O", fontsize=14, fontweight="bold")
        else:
            # Z-N
            fig.text(0.04, 0.93, "Z", fontsize=14, fontweight="bold")
            fig.text(0.83, 0.93, "N", fontsize=14, fontweight="bold")


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path_output / f"vlieglijn_{line_no}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # if line_no == 680201:
    #     break
    
# maak shapefile van vlieglijnen
lines = gpd.GeoDataFrame(list(lines.keys()),geometry=list(lines.values()), crs="epsg:28992")
lines = lines.rename(columns={0:"LINE_NR"})
lines.to_file(path_output / "vlieglijnen.gpkg")
lines_flipped = gpd.GeoDataFrame(list(lines_flipped.keys()),geometry=list(lines_flipped.values()), crs="epsg:28992")
lines_flipped = lines_flipped.rename(columns={0:"LINE_NR"})
lines_flipped.to_file(path_output / "vlieglijnen_flipped.gpkg")