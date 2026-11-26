"""Write the report of the direct bulk-EC -> chloride test avenue (Word).

Uses the results of fit_direct_ecbulk_chloride.py and compare_direct_vs_hem2chloride.py
(reports/direct_ecbulk_chloride/) and the Report helper of the chloride_uncertainties
methodology report (read only). Requires python-docx.

Run from the repository root:
    python src/formationfactor_from_xyzdata/write_direct_ecbulk_report.py [output.docx]
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CU_ROOT = Path(os.environ.get("CHLORIDE_UNCERTAINTIES_ROOT", ROOT.parent / "chloride_uncertainties"))
REP = ROOT / "reports/direct_ecbulk_chloride"
OUT = REP / (sys.argv[1] if len(sys.argv) > 1 else "direct_ECbulk_chloride_test.docx")

_argv, sys.argv = sys.argv, sys.argv[:1]  # the imported module reads sys.argv at import
spec = importlib.util.spec_from_file_location("ec_report", CU_ROOT / "src/5-visualize/write_ec_chloride_report.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)
sys.argv = _argv
Report, fmt, n, add_field = rep.Report, rep.fmt, rep.n, rep.add_field
from docx.shared import Pt  # noqa: E402

AREAS = [("north", "North", "N", "N-B2"), ("noord-holland", "Noord-Holland", "NH", "NH-B2"),
         ("south", "South", "Z", "Z-C3")]
RELS = ["chain_linear", "chain_regional", "direct", "direct_fa"]
RLAB = {"chain_linear": "chain, previous linear", "chain_regional": "chain, area relation",
        "direct": "direct, generic", "direct_fa": "direct, formation-aware"}


def pct(v, nd=0):
    return f"{100 * v:.{nd}f}%"


def load():
    fit = json.loads((REP / "direct_relation_fit.json").read_text())
    cmp_ = json.loads((REP / "comparison_summary.json").read_text())
    val = pd.read_csv(REP / "validation_table.csv")
    for c in [c for c in val.columns if c.endswith("_ci")]:
        val[c] = val[c].map(ast.literal_eval)
    cov = pd.read_csv(REP / "coverage_direct_relation.csv", index_col=0)
    resid = pd.read_csv(REP / "residuals_by_lithology.csv")
    lines = {ab: pd.read_csv(REP / f"line_ranking_{ab}.csv").iloc[0] for _, _, ab, _ in AREAS}
    return fit, cmp_, val, cov, resid, lines


def fig_groups(fit, path):
    """Median Cl of the generic relation and of selected group relations."""
    p = fit["recommended"]["params"]
    groups = fit["formation_aware"]["groups"]

    def med(a, e0, x):
        s = (x - e0) / p["tau"]
        return np.exp(a + p["b"] * np.log(p["tau"] * np.logaddexp(0.0, s)))

    x = np.logspace(-2, 1, 300)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    show = [("unit:NUhlc", "Holocene (NUhlc)", "#eb6834"), ("formation:NUPZ-WA", "Peize-Waalre (NUPZ-WA)", "#1baf7a"),
            ("formation:NUPE", "Peelo (NUPE)", "#a1320a"), ("formation:NUBX", "Boxtel (NUBX)", "#9cc1ec"),
            ("formation:NUMS", "Maassluis (NUMS)", "#0b3d91")]
    for key, lab, col in show:
        if key in groups:
            g = groups[key]
            ax.plot(x, med(g["a"], g["e0"], x), color=col, lw=1.6, label=f"{lab}, n = {g['n']}")
    ax.plot(x, med(p["a"], p["e0"], x), color="#1f1f1d", lw=2.4, label="generic")
    ax.set(xscale="log", yscale="log", ylim=(5, 3e4))
    ax.set_xlabel("bulk EC (in situ) [mS/cm]", fontsize=9)
    ax.set_ylabel("median Cl [mg/l]", fontsize=9)
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build(fit, cmp_, val, cov, resid, lines):
    R = Report()
    d, P = fit["data"], fit["recommended"]["params"]
    lo, hi = fit["recommended"]["boot_lo"], fit["recommended"]["boot_hi"]
    fa = fit["formation_aware"]
    cv = fa["cv"]
    V = lambda a, r: val[(val.area == a) & (val.relation == r)].iloc[0]

    # --- title ---------------------------------------------------------------
    t = R.doc.add_paragraph()
    t.paragraph_format.space_before = Pt(60)
    r = t.add_run("Converting HEM bulk conductivity directly to chloride")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(22), rep.ACCENT
    R.p("An exploratory test of bypassing the formation-factor and surface-conductivity chain of "
        "hem2chloride", size=13)
    R.p("FRESHEM-NL — formation factors / chloride uncertainties — technical note (test avenue, "
        "not for incorporation)", size=11, italic=True)
    R.p("")
    vs = {a: {r_: V(a, r_) for r_ in RELS} for a, *_ in AREAS}
    R.box("Summary", [
        ["hem2chloride converts HEM resistivity to chloride through the pore-water EC, using GeoTOP "
         "lithology, formation factors and surface conductivities. This note tests a shortcut: a "
         "direct empirical relation between bulk EC and chloride, fitted on "
         f"{n(d['fit_pairs'])} collocated chloride samples and borehole/CPT conductivities "
         f"({d['fit_logs']} logs)."],
        ["Two versions were tested: a geology-unaware (generic) relation and a formation-aware "
         "version with relations per REGIS unit or formation, falling back to the generic relation "
         "where there are too few collocated samples. The formation-aware version predicts held-out "
         f"logs better (+{cv['delta_mean']:.3f} ± {cv['delta_se']:.3f} log density per sample), "
         "almost entirely through the Holocene units."],
        ["On 48 HEM test lines the direct conversion changes the detail chloride class of 30–70% of "
         "the layers compared with the chain, gives much more chloride in fresh water and about "
         "three times wider uncertainty intervals."],
        [("Validation ", "b"), "(measurements from 1980 onwards): the generic direct relation has "
         "the smallest errors in all three areas and well-calibrated 90% intervals, but is too "
         "saline in Noord-Holland. The formation-aware version is best in Noord-Holland but fails "
         "for the saline Holocene of Zeeland, because one Holocene relation cannot represent both "
         "clay/peat and sand."],
        [("Conclusion: ", "b"), "a direct relation is competitive with the chain and more honest "
         "about uncertainty, but lithology matters where it differs most (Holocene). A lithology-"
         "aware direct relation (GeoTOP lithoclass rather than REGIS unit) is the logical next "
         "test."],
    ])
    R.page_break()
    R.p(("Contents", "b"), size=13)
    add_field(R.doc.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u')
    R.page_break()

    # --- 1 introduction ------------------------------------------------------
    R.h("1  Introduction")
    R.p("In hem2chloride, the bulk EC of each HEM layer (ECbulk = 10/ρ, in mS/cm) is converted to "
        "the EC of the pore water (ECw) with the formation factor F and the surface conductivity "
        "ECs of the lithology in that layer, and ECw is converted to chloride:")
    R.eq("ECbulk = ECw / F + ECs   →   ECw = F·(ECbulk − ECs)   →   Cl = f(ECw)")
    R.p("F and ECs are sampled per lithoclass and stratigraphic unit from look-up tables derived "
        "from laboratory and logging data, and the lithology from the GeoTOP probabilities. Each "
        "step adds assumptions and uncertainty. This note explores the alternative of skipping the "
        "petrophysical step altogether, with an empirical relation between in-situ bulk EC and "
        "chloride:")
    R.eq("Cl = g(ECbulk)")
    R.p("fitted on locations where a chloride sample and a bulk-conductivity log are collocated. "
        "The question is how much the predicted chloride changes and whether it validates better or "
        "worse. This is a test avenue only; hem2chloride itself was not changed.")

    # --- 2 data --------------------------------------------------------------
    R.h("2  Collocated data")
    R.p("Chloride samples of the FRESHEM validation data set were paired with conductivity logs "
        "(build_dataset.py): borehole geophysical logs (LAS) and electrical cone penetration tests "
        "(ECPT) within 250 m horizontally, with the log sample inside the filter screen or at most "
        "1 m from it. Each pair was assigned the REGIS II unit at the filter. ECbulk is ten times "
        "the logged conductivity (S/m); no temperature correction was applied, because HEM "
        "resistivity is converted in the same way, so that both represent the in-situ "
        "temperature.")
    R.table(["Step", "Pairs", "Remark"], [
        ["Collocated pairs", n(d["input_pairs"]), "3 638 LAS, 105 ECPT"],
        ["Plausible conductivity", n(d["after_plausibility"]), "0 < conductivity ≤ 2.5 S/m"],
        ["Held out for validation", n(d["held_out_validation"]),
         "wells of the HEM validation points and pairs within 500 m of them"],
        [("Used for fitting", "b"), (n(d["fit_pairs"]), "b"),
         f"{d['fit_logs']} logs, {n(d['fit_wells'])} wells"],
    ], "Collocated chloride – bulk conductivity pairs.", widths=[4.5, 2.0, 9.9], align="lrl")
    lc = fit["lithology_counts"]
    R.p(f"The pairs are dominated by fresh groundwater in Pleistocene sand: {pct(1 - fit['share_cl_ge_150'])} "
        f"have Cl < 150 mg/l and {pct(fit['share_cl_ge_1000'])} Cl ≥ 1000 mg/l; {n(lc['sand'])} pairs "
        f"are in sandy REGIS units, {n(lc['complex'])} in complex units (mainly the Holocene, "
        f"NUhlc) and {n(lc['clay'])} in clayey units. Most filters are deep (median about −50 m "
        "NAP). The bulk EC was not checked against the chloride beyond the plausibility limit.")

    # --- 3 relations ---------------------------------------------------------
    R.h("3  Direct relations")
    R.h("3.1  Model", 2)
    R.p("The model library of the EC25–chloride analysis (chloride_uncertainties, "
        "fit_ec_chloride.py) was used with bulk EC as predictor. The selected form is the "
        "smoothed shifted power law with a Student-t error and a quadratic log-scale:")
    R.eq("ln Cl = a + b·ln[τ·ln(1 + exp((ECbulk − e0)/τ))] + σ·ε,   ln σ = s0 + s1·u + s2·u²,   "
         "u = ln ECbulk,   ε ~ t(ν)")
    R.p("Here e0 acts as the background bulk conductivity of fresh sediment (surface conduction "
        "and fresh pore water), and b and a describe the increase with salinity (for a constant "
        "formation factor b = 1 and exp(a) ∝ F). Table 2 compares four model variants.")
    aic = fit["aic"]
    best = min(aic.values())
    R.table(["Median model", "σ model", "AIC", "ΔAIC"], [
        [k.split("_t_")[0].replace("shiftedpower", "shifted power").replace("loglinear", "power law"),
         k.split("_t_")[1], f"{v:,.0f}".replace(",", " "), f"{v - best:,.0f}".replace(",", " ")]
        for k, v in sorted(aic.items(), key=lambda kv: kv[1])],
        "Model comparison (Student-t errors, all pairs).", widths=[4.5, 3.0, 3.0, 3.0], align="llrr")

    R.h("3.2  Generic (geology-unaware) relation", 2)
    keys = [("a", "level"), ("b", "exponent"), ("e0", "background bulk EC (mS/cm)"),
            ("tau", "smoothing width (mS/cm)"), ("s0", "log-scale, constant"), ("s1", "log-scale, linear"),
            ("s2", "log-scale, quadratic"), ("nu", "degrees of freedom")]
    R.table(["Parameter", "Meaning", "Estimate", "95% CI (bootstrap of logs)"],
            [[{"tau": "τ", "nu": "ν"}.get(k, k), lab, fmt(P[k], 3), f"{lo[k]:.3f} – {hi[k]:.3f}"]
             for k, lab in keys],
            f"Parameters of the generic direct relation (n = {n(d['fit_pairs'])}; "
            f"{fit['settings']['n_boot']} bootstrap resamples of logs).",
            widths=[2.2, 5.6, 2.4, 4.6], align="llrr")
    R.figure(REP / "fig_direct_relation.png",
             "Collocated pairs coloured by REGIS lithology group, with the median and the 50% and 90% "
             "predictive intervals of the generic direct relation. Left: log scale; right: linear "
             "chloride axis.", width=16.2)
    cv_rows = [[idx, n(r_["n"]), fmt(r_["cover_50"]), fmt(r_["cover_90"])] for idx, r_ in cov.iterrows()]
    R.table(["Bulk EC (mS/cm)", "n", "inside 50%", "inside 90%"], cv_rows,
            "Coverage of the predictive intervals of the generic relation on the fitted pairs.",
            widths=[4.0, 2.0, 2.5, 2.5], align="lrrr")
    med = lambda x: np.exp(P["a"] + P["b"] * np.log(P["tau"] * np.logaddexp(0.0, (x - P["e0"]) / P["tau"])))
    R.p(f"The median chloride is about {med(0.05):.0f} mg/l at 0.05 mS/cm, {med(P['e0']):.0f} mg/l at "
        f"e0 = {P['e0']:.2f} mS/cm, {med(1.0):,.0f} mg/l at 1 mS/cm and {med(10.0):,.0f} mg/l at "
        "10 mS/cm. The scatter is large "
        f"(σ ≈ {np.exp(P['s0']):.2f} in ln Cl, i.e. a factor of about three for one standard "
        "deviation) and almost constant, because it contains the variation of lithology as well as "
        "the measurement and collocation errors. The intervals are well calibrated below 2 mS/cm "
        "and too wide above, where few pairs exist (Table 4).")

    R.h("3.3  Formation-aware relations", 2)
    R.p("Residuals of the generic relation differ between lithology groups (median residual in "
        "ln Cl: complex units +0.57 below 0.5 mS/cm, clayey units −0.2 to −0.6 below 2 mS/cm). In the formation-aware version, "
        "a relation is fitted per REGIS unit if it has at least 50 pairs from at least 10 logs, "
        "otherwise per formation (the unit code without its lithological suffix), otherwise the "
        "generic relation is used. Each group gets its own level a, and its own background e0 if "
        "it has at least 15 pairs with Cl ≥ 150 mg/l; the exponent, smoothing and error model are "
        "those of the generic relation. This is the empirical counterpart of a formation factor "
        "(a) and a surface conductivity (e0) per unit.")
    groups = fa["groups"]
    rows = []
    unit_n = {(g["key"], g["n"]) for g in groups.values() if g["level"] == "unit"}
    dup = {k for k, g in groups.items() if g["level"] == "formation" and any(
        u.startswith(g["key"]) and nn == g["n"] for u, nn in unit_n)}  # formation = one unit
    for key, g in sorted(((k, g) for k, g in groups.items() if k not in dup), key=lambda kv: -kv[1]["n"])[:14]:
        rows.append([key.split(":")[1], key.split(":")[0], str(g["n"]), str(g["n_logs"]),
                     str(g["n_cl_ge_150"]), fmt(g["a"]), fmt(g["e0"]) if "e0" in g["free"] else "(generic)"])
    R.table(["Group", "Level", "Pairs", "Logs", "Cl ≥ 150", "a", "e0 (mS/cm)"], rows,
            f"The 14 largest of the {len(groups)} formation-aware groups (formations consisting of one fitted unit not repeated; generic: a = {P['a']:.2f}, "
            f"e0 = {P['e0']:.2f} mS/cm).", widths=[3.0, 2.0, 1.6, 1.6, 1.8, 1.6, 2.4], align="llrrrrr", size=8.5)
    sp = fa["share_pairs"]
    R.figure(REP / "fig_group_relations.png",
             "Median chloride of the generic relation and of selected formation-aware relations.",
             width=12.5)
    bl = cv["by_litho"]
    R.p(f"Of the fitted pairs, {pct(sp.get('unit', 0))} receive a unit relation, "
        f"{pct(sp.get('formation', 0))} a formation relation and {pct(sp.get('generic', 0))} the generic "
        "relation. The Holocene (NUhlc) stands out (Figure 2): above a bulk EC of about 0.4 mS/cm it "
        "contains much less chloride than the other units at the same bulk EC, because clay and "
        "peat conduct through their surfaces; below that its fresh background is higher. The Pleistocene units differ in level and, less, in background.")
    R.p("A five-fold cross-validation, leaving out whole logs, confirms the gain: the mean log "
        f"predictive density per held-out sample increases from {cv['mean_lpd_generic']:.3f} to "
        f"{cv['mean_lpd_formation']:.3f} (+{cv['delta_mean']:.3f} ± {cv['delta_se']:.3f}, standard "
        f"error with logs as clusters). The gain is concentrated in the complex units "
        f"(+{bl['complex']['mean']:.2f} per sample) and small in sandy (+{bl['sand']['mean']:.2f}) and "
        f"clayey units (+{bl['clay']['mean']:.2f}).")

    # --- 4 comparison on HEM lines ------------------------------------------
    R.h("4  Comparison on HEM test lines")
    R.p("The test set of the EC25–chloride analysis was used: 16 flight lines in each of the "
        "blocks N-B2, NH-B2 and Z-C3, with all HEM layers above the depth of investigation that lie "
        "within GeoTOP (to about −50 m NAP). For the direct conversion the HEM resistivity was "
        "sampled from its log-normal uncertainty as in hem2chloride (200 realisations), converted to "
        "bulk EC and to chloride. For the formation-aware version the REGIS unit at the midpoint of "
        "each HEM layer selects the relation. The chain results (GeoTOP lithology, formation factor, "
        "surface conductivity, previous linear or current area relation) were taken from the "
        "EC25–chloride analysis.")
    rows = []
    for a, lab, ab, blk in AREAS:
        s = cmp_[a]
        for r_ in ("direct", "direct_fa"):
            cd = s[f"class_change_detail_{r_}_vs_chain_regional"]
            cc = s[f"class_change_coarse_{r_}_vs_chain_regional"]
            rows.append([lab if r_ == "direct" else "", RLAB[r_],
                         *[f"{pct(cd[q]['any'])} ({pct(cd[q]['lower'])}/{pct(cd[q]['higher'])})"
                           for q in ("q25", "q50", "q75")], pct(cc["q50"]["any"]),
                         f"{s[f'log10_iqr_{r_}']:.2f}"])
        rows.append(["", "chain, area relation", "–", "–", "–", "–", f"{s['log10_iqr_chain_regional']:.2f}"])
    R.table(["Area", "Conversion", "Q25 detail class", "Q50 detail class", "Q75 detail class",
             "Q50 coarse class", "IQR (log10)"], rows,
            "Share of HEM layers whose class changes compared with the chain with the area relation "
            "(in brackets: lower / higher class), and the median interquartile range of chloride "
            "(log10 Q75/Q25). Detail classes: 150, 300, 500, 1000, 1500, 3000, 5000, 10 000 mg/l; "
            "coarse: 1000 and 10 000 mg/l.",
            widths=[2.4, 3.2, 2.5, 2.5, 2.5, 1.8, 1.6], align="llrrrrr", size=8)
    R.figure(REP / "fig_overview.png",
             "Top: detail class of the median chloride of all test-line layers per conversion. "
             "Bottom: median chloride against the median HEM bulk EC.", width=16.2)
    sz = cmp_["south"]
    R.p("The direct conversion changes the result far more than the choice of EC25–chloride "
        "relation within the chain (4–9% of the layers): 30–70% of the layers change detail class "
        "and 15–50% coarse class. Two effects dominate:")
    R.bullets([
        [("Wider uncertainty. ", "b"), "The interquartile range of the direct conversion is about "
         "three times that of the chain (0.6 vs 0.2–0.3 log10), because the scatter of the direct "
         "relation contains all lithological variation, whereas the chain attributes part of the "
         "variation to the (sampled) lithology. Consequently Q25 moves down and Q75 moves up by one "
         "or more classes in most layers."],
        [("Higher chloride in fresh water. ", "b"), "Below a bulk EC of about 1 mS/cm the generic "
         "direct relation gives much more chloride than the chain (median ratio about 4–50), where "
         "the chain yields almost zero chloride; the difference is largest in the Holocene, where "
         "surface conduction of clay and peat raises the bulk EC. Above 3 mS/cm the medians agree "
         "within about 0.1 log10. The formation-aware "
         "version corrects the Holocene in Noord-Holland, but in Zeeland it makes the saline "
         f"Holocene two to three classes fresher (Q50 coarse class changed in "
         f"{pct(sz['class_change_coarse_direct_fa_vs_chain_regional']['q50']['any'])} of the layers, "
         f"almost all towards fresher; Figure 5)."],
    ])
    nh_line = int(lines["NH"]["line_no"])
    z_line = int(lines["Z"]["line_no"])
    R.figure(REP / f"section_NH_line{nh_line}.png",
             f"Noord-Holland, line {nh_line}: median chloride of the chain (area relation) and of the "
             "formation-aware direct conversion, and the change of the detail class of Q25, Q50 and "
             "Q75.", width=14.5)
    R.figure(REP / f"section_Z_line{z_line}.png", f"South, line {z_line}; as Figure 4.", width=14.5)

    # --- 5 validation --------------------------------------------------------
    R.h("5  Validation")
    R.p("The validation points of the EC25–chloride analysis were used: chloride measurements "
        "from 1980 onwards within 100 m of a HEM sounding of the test blocks, deeper than 3 m below "
        "NAP, in the HEM layer containing the filter. These wells and all collocated pairs within "
        "500 m were excluded from the fit of the direct relations. Confidence intervals were "
        "obtained by resampling HEM soundings (measurements at one sounding share their errors); "
        "differences with the chain were tested pairwise (McNemar).")
    rows = []
    for a, lab, ab, blk in AREAS:
        for r_ in RELS:
            v = V(a, r_)
            ci = v["coverage_50_ci"]
            rows.append([lab if r_ == RELS[0] else "", RLAB[r_],
                         f"{int(v['n'])}" if r_ == RELS[0] else "",
                         fmt(v["coarse_hit_q50"]), fmt(v["detail_hit_q50"]),
                         f"{v['coverage_50']:.2f} ({ci[0]:.2f}–{ci[1]:.2f})", fmt(v["coverage_90"]),
                         f"{v['brier_1000']:.3f}", f"{v['median_log10_error']:+.2f}", fmt(v["mean_abs_log10_error"])])
    R.table(["Area", "Conversion", "n", "coarse class", "detail class", "inside 50% (95% CI)",
             "inside 90%", "Brier ≥1000", "median log10 error", "mean |log10 error|"], rows,
            "Validation (measurements ≥ 1980): share of measurements in the class of the median, "
            "interval coverage (nominal 0.50 and 0.90), Brier score of P(Cl ≥ 1000 mg/l) and the error "
            "of the median (log10 predicted − log10 measured, floor 10 mg/l).",
            widths=[2.0, 3.0, 0.8, 1.4, 1.4, 2.4, 1.3, 1.4, 1.5, 1.5], align="llrrrrrrrr", size=7.5)
    R.figure(REP / "fig_validation.png",
             "Validation per area: class hit rates and interval coverage with 95% cluster-bootstrap "
             "intervals, and the class error of the median (coarse classes). CL/CR chain with the "
             "previous linear / area relation, DG/DF direct generic / formation-aware.", width=16.2)
    vn, vh, vz = vs["north"], vs["noord-holland"], vs["south"]
    p_z = V("south", "direct_fa")["p_mcnemar_coarse_vs_chain_regional"]
    p_zc = V("south", "direct")["p_mcnemar_cov50_vs_chain_regional"]
    R.bullets([
        [("Generic direct relation. ", "b"), "Smallest errors in all areas (mean |log10 error| "
         f"{vn['direct']['mean_abs_log10_error']:.2f}, {vh['direct']['mean_abs_log10_error']:.2f} and "
         f"{vz['direct']['mean_abs_log10_error']:.2f} vs {vn['chain_regional']['mean_abs_log10_error']:.2f}, "
         f"{vh['chain_regional']['mean_abs_log10_error']:.2f} and {vz['chain_regional']['mean_abs_log10_error']:.2f} "
         "for the chain) and 90% intervals close to nominal "
         f"({vn['direct']['coverage_90']:.2f}, {vh['direct']['coverage_90']:.2f}, {vz['direct']['coverage_90']:.2f}). "
         "Its 50% intervals are too wide in the north and south, where those of the chain are too "
         f"narrow (south {vz['direct']['coverage_50']:.2f} vs {vz['chain_regional']['coverage_50']:.2f}, "
         f"McNemar p = {p_zc:.0e}). In Noord-Holland it is too saline: one third of the medians are "
         f"one coarse class too high and the Brier score for 1000 mg/l is "
         f"{vh['direct']['brier_1000']:.3f} vs {vh['chain_regional']['brier_1000']:.3f}."],
        [("Formation-aware direct relation. ", "b"), "Best in Noord-Holland (coarse class "
         f"{vh['direct_fa']['coarse_hit_q50']:.2f}, mean |log10 error| {vh['direct_fa']['mean_abs_log10_error']:.2f}), "
         "where the Holocene correction removes the saline bias. In the south it fails: the median "
         f"is in the measured coarse class for only {vz['direct_fa']['coarse_hit_q50']:.2f} of the "
         f"measurements (chain {vz['chain_regional']['coarse_hit_q50']:.2f}, p = {p_z:.0e}), because the "
         "Holocene relation, fitted mostly on fresh clay and peat, is applied to saline Holocene "
         "sand. In the north it is too saline in the Peelo Formation."],
        [("Chain. ", "b"), "Class hit rates comparable to the generic direct relation (better in "
         "Noord-Holland, worse in the north), but too narrow intervals: the chain is overconfident."],
    ])
    R.p("With 34 and 49 measurements at 10 and 19 soundings in the north and Noord-Holland, most "
        "differences there are not statistically significant; the south (277 measurements, 67 "
        "soundings) is the only area where the conversions can be told apart with confidence.")

    # --- 6 discussion --------------------------------------------------------
    R.h("6  Discussion")
    R.bullets([
        [("A direct relation is a serious competitor. ", "b"), "Without any petrophysics, the "
         "generic relation validates at least as well as the chain on the median and better on "
         "calibration. Much of the information in the chain appears to be lost in the uncertainty "
         "of lithology, formation factor and surface conductivity, while the chain's intervals do not "
         "fully reflect that uncertainty."],
        [("Lithology still matters where it differs most. ", "b"), "The Holocene clay and peat "
         "versus sand contrast is real (Figure 2, cross-validation gain) but cannot be represented by "
         "REGIS units: the Holocene is one complex unit. The formation-aware version therefore "
         "improves fresh Holocene settings and degrades saline ones. GeoTOP lithoclasses resolve "
         "this contrast and are already used by the chain."],
        [("Training data limit the direct relation. ", "b"), "The collocated pairs are mostly fresh, "
         f"deep and sandy ({pct(fit['share_cl_ge_1000'])} with Cl ≥ 1000 mg/l); saline Holocene settings "
         "— the most important for FRESHEM — are poorly represented."],
        [("Support. ", "b"), "Borehole logs measure decimetre-scale conductivity next to the filter, "
         "HEM layers average metres vertically and about 100 m laterally. The direct relation "
         "transfers the log-scale scatter to HEM; part of the wide intervals may be scale-related."],
        [("Independence of the validation. ", "b"), "Training pairs and validation measurements both "
         "come from the FRESHEM validation data set. Wells and pairs within 500 m of the validation "
         "points were held out, but regional similarity remains."],
    ])

    # --- 7 conclusions -------------------------------------------------------
    R.h("7  Conclusions and next steps")
    R.bullets([
        "A direct bulk-EC to chloride relation, fitted on collocated borehole/CPT data, is a "
        "viable and much simpler alternative to the formation-factor chain for the median chloride, "
        "and gives better calibrated uncertainty.",
        "A geology-unaware relation is biased where lithology deviates (Holocene clay/peat); a "
        "REGIS-based formation-aware relation helps in fresh Holocene settings but fails in saline "
        "ones.",
        "Next test: a direct relation conditioned on the GeoTOP lithoclass (sampled as in the "
        "chain), with a fallback to the generic relation, and more saline collocated data "
        "(e.g. from the Zeeland and Noord-Holland coastal zones).",
        "A hybrid is also conceivable: the direct relation for its calibrated uncertainty, the "
        "chain to inform lithology-specific shifts.",
        "These results are exploratory; no change to hem2chloride is proposed.",
    ])

    R.h("Appendix A  Reproducibility")
    R.table(["Step", "Script / output"], [
        ["Collocated pairs", "src/formationfactor_from_xyzdata/build_dataset.py → "
                             "data/3-input/formationfactor_from_xyzdata/"],
        ["Direct relations", "src/formationfactor_from_xyzdata/fit_direct_ecbulk_chloride.py"],
        ["Comparison and validation", "src/formationfactor_from_xyzdata/compare_direct_vs_hem2chloride.py"],
        ["Chain results, test set, validation points",
         "chloride_uncertainties: src/4-analyze/compare_cl_relations.py (branch ec-chloride-relation)"],
        ["This document", "src/formationfactor_from_xyzdata/write_direct_ecbulk_report.py"],
        ["Outputs", "reports/direct_ecbulk_chloride/"],
    ], "Scripts and outputs (repository formationfactors, branch direct-ecbulk-chloride).",
        widths=[5.0, 11.4], align="ll")
    return R


def main():
    fit, cmp_, val, cov, resid, lines = load()
    fig_groups(fit, REP / "fig_group_relations.png")
    R = build(fit, cmp_, val, cov, resid, lines)
    R.save(OUT)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
