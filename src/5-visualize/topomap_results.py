#%%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

path_output = r"c:\Users\dam_re\OneDrive - Stichting Deltares\Documents\Projecten\FRESHEM\scripts\formationfactors\data\5-visualization"
path_vlieglijnen = r"c:\Users\dam_re\Nextcloud\Team4-Chloride\Results_Monte_Carlo\ZLZ-C3_20-08-26_clip\flightlines\ZLZ-C3-hem_lines.shp"

# vlieglijnen
gdf_lines = gpd.read_file(path_vlieglijnen)

# huidige lijnnummer
lijn_id = 890201.0

# selectie huidige lijn
gdf_current = gdf_lines[gdf_lines["line_no"] == lijn_id]

import contextily as cx
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.geometry import CAP_STYLE
#%%
# --- W-O lijn ---

line = gdf_current.geometry.iloc[0]
buffer_y = 1300  # m loodrecht op profiel
# buffer_corridor = 1

# corridor = line.buffer(
#     buffer_corridor,
#     cap_style=CAP_STYLE.flat  # geen verlenging aan de uiteinden
# )


fig, ax = plt.subplots(figsize=(14, 2))
xmin, ymin, xmax, ymax = line.bounds
cy = (ymin + ymax) / 2

ax.set_xlim(xmin, xmax)
ax.set_ylim(cy-buffer_y, cy+buffer_y)



# lijn tekenen
gdf_current.plot(
    ax=ax,
    color="red",
    linewidth=3,
)

cx.add_basemap(
    ax,
    crs="EPSG:28992",
    source=cx.providers.Esri.WorldTopoMap,
    alpha=1,
    attribution=False
)

# cx.add_basemap(
#     ax,
#     crs="EPSG:28992",
#     source=cx.providers.Esri.WorldBoundariesAndPlaces,
#     alpha=0.8
# )

ax.set_axis_off()


plt.savefig(fr"{path_output}/topomap_{lijn_id}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
)

plt.show()

#%%
#%%
# --- Z-N lijn ---

# huidige lijnnummer
lijn_id = 106601.0

# selectie huidige lijn
gdf_current = gdf_lines[gdf_lines["line_no"] == lijn_id]

line = gdf_current.geometry.iloc[0]
buffer_x = 1300  # m loodrecht op profiel
# buffer_corridor = 1

# corridor = line.buffer(
#     buffer_corridor,
#     cap_style=CAP_STYLE.flat  # geen verlenging aan de uiteinden
# )


fig, ax = plt.subplots(figsize=(2, 14))
xmin, ymin, xmax, ymax = line.bounds
c_x = (xmin + xmax) / 2

ax.set_xlim(c_x-buffer_x, c_x+buffer_x)
ax.set_ylim(ymin, ymax)



# lijn tekenen
gdf_current.plot(
    ax=ax,
    color="red",
    linewidth=3,
)

cx.add_basemap(
    ax,
    crs="EPSG:28992",
    source=cx.providers.Esri.WorldTopoMap,
    alpha=1,
    attribution=False
)

# cx.add_basemap(
#     ax,
#     crs="EPSG:28992",
#     source=cx.providers.Esri.WorldBoundariesAndPlaces,
#     alpha=0.8
# )

ax.set_axis_off()


plt.savefig(fr"{path_output}/topomap_{lijn_id}_ZN.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
)

plt.show()