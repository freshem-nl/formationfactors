"""Script to make boxplots per lithostrat category"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

path_labresults = Path("data/3-input/lab_results")
fn_labresults = path_labresults / "20260304_tbl20_WPchloride_FFdata.xlsx"
fn_labresults_inc_grainsize = path_labresults / "20260126_tbl05_Measurementdata_Full.xlsx"
path_results = Path("data/4-output/ff_ecs_uncertainty")
path_geotop = "p:/gis-data/GEOTOP/geotop.zarr"
path_regis = "p:/gis-data/REGIS/REGIS2.2/REGIS_v2_2.nc"

path_results.mkdir(exist_ok=True, parents=True)

df_all = pd.read_excel(fn_labresults)

# Get all data per litho, per lithostrat
# make boxplots of FF and ECs
df = df_all.loc[df_all["Type_name"]=="FF_Disturbed"]
for column in ["LITHOKLASSE_CD", "StratLithoklasse"]:
    for code in df[column].unique().dropna():
        print(f"{column} {code}")
        dfsel = df.loc[df[column] == code]

        fig, axes = plt.subplots(1, 2, figsize=(8,5))
        do_bootstrap = len(dfsel) > 2
        dfsel.boxplot(column=["SIP3_FormationFactor_F_3W_unitless"], ax=axes[0], notch=do_bootstrap, bootstrap=1000)
        dfsel.boxplot(column=["SIP3_SurfCond_Sigmas_3W_S/m"], ax=axes[1], notch=do_bootstrap, bootstrap=1000)
        n = dfsel[["SIP3_FormationFactor_F_3W_unitless","SIP3_SurfCond_Sigmas_3W_S/m"]].count()
        axes[0].set_title(f"n = {n['SIP3_FormationFactor_F_3W_unitless']}")
        axes[1].set_title(f"n = {n['SIP3_SurfCond_Sigmas_3W_S/m']}")
        fig.suptitle(f"{column} = {code.upper()}", fontsize="x-large")
        plt.savefig(path_results / f"{column}_{code.upper()}.png", dpi=150, bbox_inches="tight")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.close()


df_n = df.groupby("StratLithoklasse")[["SIP3_FormationFactor_F_3W_unitless","SIP3_SurfCond_Sigmas_3W_S/m"]].count()
df_n.to_csv(path_results / "n_StratLithoklasse.csv")

### test significant differences
# Interaction Term Analysis (Recommended): Combine datasets and create a categorical variable for group (e.g., A=0, B=1). Run a linear regression: . If the interaction term (
# ) is significant (p < 0.05), the slopes are significantly different.

# for now: run ttest for each combination
# significantie tabel
stratlithos = sorted(df["LITHOKLASSE_CD"].unique().dropna().tolist())+sorted(df["StratLithoklasse"].unique().dropna().tolist())
pval_ff = pd.DataFrame(index=stratlithos, columns=stratlithos)
pval_ecs = pd.DataFrame(index=stratlithos, columns=stratlithos)
n = pd.DataFrame(index=stratlithos, columns=stratlithos)

def col(code):
    if "-" in code:
        return "StratLithoklasse"
    else:
        return "LITHOKLASSE_CD"

for code_a in stratlithos:
    print(code_a)
    A = df.loc[df[col(code_a)]==code_a]
    for code_b in stratlithos:
        if code_b == code_a:
            pval_ff.loc[code_a,code_b] = 1.
            pval_ecs.loc[code_a,code_b] = 1.
            n.loc[code_a,code_b] = A["SIP3_FormationFactor_F_3W_unitless"].count()
            continue

        B = df.loc[(df[col(code_b)]==code_b)&(df[col(code_a)]!=code_a)]
        # ttest
        res_ff = stats.ttest_ind(A["SIP3_FormationFactor_F_3W_unitless"],B["SIP3_FormationFactor_F_3W_unitless"],nan_policy="omit")
        res_ecs = stats.ttest_ind(A["SIP3_SurfCond_Sigmas_3W_S/m"],B["SIP3_SurfCond_Sigmas_3W_S/m"],nan_policy="omit")
        pval_ff.loc[code_a,code_b] = res_ff.pvalue
        pval_ecs.loc[code_a,code_b] = res_ecs.pvalue
        n.loc[code_a,code_b] = B["SIP3_FormationFactor_F_3W_unitless"].count()

pval_ff.to_csv(path_results / "pval_ff.csv")
pval_ecs.to_csv(path_results / "pval_ecs.csv")
n.to_csv(path_results / "n.csv")
