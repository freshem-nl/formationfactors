"""Reconstruct the final FF/ECs statistical groups that feed the Monte Carlo look-up table.

Mirrors the grouping logic (not the tests) of
- src/4-analyze/stat_analysis_ff_surfcond_lithofacies.py  -> lithofacies_for_monte_carlo.csv
- src/4-analyze/stat_analysis_ff_surfcond_lithostrat.py   -> median_mean_std_stratlitho_manual_groups.csv
- src/4-analyze/look_up_table_monte_carlo.py              -> priority of special_stats / facies / litho

The groups only depend on group sizes (n >= 5) and the hard-coded manual merges,
so the Kruskal-Wallis / Dunn tests themselves are not needed to reproduce them.
Keep MANUAL_* and SPECIAL_STATS in sync with those scripts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FN_LAB = Path("data/3-input/lab_results/20260304_tbl20_WPchloride_FFdata.xlsx")
MIN_N = 5

FF_COL = "SIP3_FormationFactor_F_3W_unitless"
ECS_COL = "SIP3_SurfCond_Sigmas_3W_S/m"
LITHO, STRAT, STRATLITHO = "LITHOKLASSE_CD", "Stratigrafie", "StratLithoklasse"

FACIES_CODES = {
    "marien": ["NAWA", "NAWO", "NAZA", "NAWOBE", "EE", "OO", "MS", "OOSP", "BR", "WAWO"],
    "fluviatiel": ["URTY", "URVE", "AP", "BXSI", "UR", "PZ", "EC", "ST", "WA", "KK", "KW"],
    "glaciaal": ["DRGI", "DRGIGA", "PENI", "PE", "DRUI"],
    "eolisch": ["BX", "DN", "BXWI", "BXKO", "NASC"],
    "organisch": ["NIHO", "NIBA", "NI"],
    "rest": ["AAOM"],
}
FACIES_OF = {c: f for f, codes in FACIES_CODES.items() for c in codes}

# stat_analysis_ff_surfcond_lithofacies.py (active list, used for SIP3 and SIP3+SIP5)
MANUAL_FACIES = [
    ("kz", ["eolisch", "glaciaal"]),
    ("kz", ["fluviatiel", "marien"]),
    ("zf", ["eolisch", "fluviatiel"]),
    ("zf", ["glaciaal", "marien"]),
    ("zm", ["eolisch", "fluviatiel", "glaciaal"]),
]
# stat_analysis_ff_surfcond_lithostrat.py (active "only SIP3" list)
MANUAL_STRAT = [
    ("zg", ["AP", "PZ"]),
    ("kz", ["BX", "DRGI"]),
    ("kz", ["NAWA", "NAWO"]),
    ("zm", ["NAWA", "NAWO", "NAZA", "OO"]),
    ("zm", ["BX", "BXWI"]),
]
# look_up_table_monte_carlo.py: (lithoclass, strat_group) overriding facies statistics
SPECIAL_STATS = [("v", "NIHO"), ("v", "NIBA"), ("kz", "EC"), ("kz", "WA"),
                 ("kz", "NAWA+NAWO"), ("kz", "BX+DRGI")]


def load_samples(sip5: bool = False) -> pd.DataFrame:
    df = pd.read_excel(FN_LAB, keep_default_na=False, na_values=["", " ", "NULL", "NaN"])
    df = df.loc[df["Type_name"] == "FF_Disturbed"].copy()
    if sip5:
        m = df[ECS_COL].isna() & (df["FF_SIP5"] == "Yes")
        df.loc[m, [ECS_COL, FF_COL]] = df.loc[m, ["SIP5_SurfCond_Sigmas_S/m",
                                                  "SIP5_FormationFactor_F5_unitless"]].values
    miss = df[LITHO].isna() & df[STRATLITHO].notna()
    df.loc[miss, LITHO] = df.loc[miss, STRATLITHO].str[-2:]
    df = df.loc[df[FF_COL].notna() & df[ECS_COL].notna() & df[LITHO].notna() & df[STRAT].notna()]
    df = df.loc[df[STRAT] != "AAOM"].copy()
    df[STRAT] = df[STRAT].str.replace("-(MG|WG)", "", regex=True)
    df[STRATLITHO] = df[STRATLITHO].str.replace("-(MG|WG)", "", regex=True)
    return df


def _groups(df, col, manual, keep_single):
    """Groups per lithoclass: members with n >= MIN_N, manual merges first, rest individually."""
    rows = []
    valid_global = {k for k, v in df[col].value_counts().items() if v >= MIN_N}
    for litho, sub in df.groupby(LITHO):
        if len(sub) < MIN_N:
            continue
        counts = sub[col].value_counts()
        eligible = sorted(counts[counts >= MIN_N].index)
        if len(eligible) < 2:
            if not keep_single:
                continue  # lithostrat script drops single-strat lithoclasses
            eligible = [e for e in eligible if e in valid_global]
        remaining = list(eligible)
        for lit, wanted in manual:
            if lit != litho:
                continue
            members = [w for w in wanted if w in remaining]
            if members:
                rows.append((litho, "+".join(members), members, int(counts[members].sum()), True))
                remaining = [r for r in remaining if r not in members]
        for m in remaining:
            rows.append((litho, m, [m], int(counts[m]), False))
    return pd.DataFrame(rows, columns=["litho", "group", "members", "n", "merged"])


def build_groups(sip5: bool = False):
    df = load_samples(sip5)
    df_strat = df.loc[df[STRATLITHO].notna()]
    df_fac = df.assign(facies=df[STRAT].map(FACIES_OF)).dropna(subset=["facies"])
    facies = _groups(df_fac, "facies", MANUAL_FACIES, keep_single=True)
    strat = _groups(df_strat, STRAT, MANUAL_STRAT, keep_single=False)
    special = []
    for litho, grp in SPECIAL_STATS:
        hit = strat.loc[(strat["litho"] == litho) & (strat["group"] == grp)]
        if hit.empty:
            raise ValueError(f"look-up table expects lithostrat group {litho}-[{grp}] but it is missing")
        special.append(hit.iloc[0])
    special = pd.DataFrame(special).reset_index(drop=True)
    litho_n = df_fac.groupby(LITHO).size()
    return facies, special, strat, litho_n


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parents[2])
    f, s, st, n = build_groups()
    print(f.to_string()); print(s.to_string()); print(st.to_string()); print(n)
