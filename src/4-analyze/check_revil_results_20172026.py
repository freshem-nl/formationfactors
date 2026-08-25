

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

path_data = Path("data/1-external/revil")
fn_2026 = path_data / "20260304_tbl20_WPchloride_FFdata.xlsx"
fn_2017 = path_data / "data_sampling_surface_conductance_results2JD.xlsx"
path_output = Path("data/4-output/revil_analysis")
path_output.mkdir(exist_ok=True)

df26 = pd.read_excel(fn_2026)
df17 = pd.read_excel(fn_2017)

df26["FF"] = df26['SIP3_FormationFactor_F_3W_unitless']#combine_first()
df26["ECs"] = df26['SIP3_SurfCond_Sigmas_3W_S/m']#combine_first()
df17["FF"] = df17['Formation factor (-)']
df17["ECs"] = df17['Surface conductivity (S/m)']

# gelijktrekken lithostrat codes
#"2026: Stratigrafie	LITHOKLASSE_CD  StratLithoklasse"  NAWO-kz
# 2017: Stratigraphy	Lithology (field description)	Combined code  NAWOKZ
df26["strat"] = df26["Stratigrafie"]
df26["litho"] = df26["LITHOKLASSE_CD"]
df26["lithostrat"] = df26["strat"] + "-"+df26["litho"]
df17["strat"] = df17["Stratigraphy"]
df17["litho"] = df17["Lithology (field description)"].apply(lambda x:x.lower())
df17["lithostrat"] = df17["strat"] + "-"+df17["litho"]

# maak figuren FF-ECs van data 2017 vs 2026
# eerst per lithostrat
for ls in df17["lithostrat"].unique():
    sel17 = df17.loc[df17["lithostrat"]==ls]
    sel26 = df26.loc[df26["lithostrat"]==ls]
    ax = sel17.plot.scatter(x="FF",y="ECs", c="orange", marker="o", loglog=True, label="results 2017")
    ax = sel26.plot.scatter(ax=ax, x="FF",y="ECs", c="b", marker="D", loglog=True, label="results 2026")
    plt.ylim(1e-4,1e-0)
    plt.xlim(1e0,1e1)
    plt.title(ls)
    plt.legend()
    plt.grid()
    plt.savefig(path_output / f"{ls}_loglog.png")
    plt.close()


for ls in df17["lithostrat"].unique():
    sel17 = df17.loc[df17["lithostrat"]==ls]
    sel26 = df26.loc[df26["lithostrat"]==ls]
    ax = sel17.plot.scatter(x="FF",y="ECs", c="orange", marker="o", loglog=False, label="results 2017")
    ax = sel26.plot.scatter(ax=ax, x="FF",y="ECs", c="b", marker="D", loglog=False, label="results 2026")
    plt.ylim(0,1.)
    plt.xlim(1.,10)
    plt.title(ls)
    plt.legend()
    plt.grid()
    plt.savefig(path_output / f"{ls}.png")

for ls in df17["lithostrat"].unique():
    sel17 = df17.loc[df17["lithostrat"]==ls]
    print(f"{ls}: {len(sel17)}")


# for ls in df17["lithostrat"].unique():
#     sel17 = df17.loc[df17["lithostrat"]==ls]
#     sel17["ECs"] /= 10.
#     sel26 = df26.loc[df26["lithostrat"]==ls]
#     ax = sel17.plot.scatter(x="FF",y="ECs", c="orange", marker="o", loglog=True, label="results 2017")
#     ax = sel26.plot.scatter(ax=ax, x="FF",y="ECs", c="b", marker="D", loglog=True, label="results 2026")
#     plt.title(ls)
#     plt.legend()
#     plt.savefig(path_output / f"{ls}_ecsdiv10.png")

