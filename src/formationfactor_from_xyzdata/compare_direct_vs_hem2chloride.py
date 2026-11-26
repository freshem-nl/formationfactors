"""Test avenue: compare a direct bulk-EC -> chloride conversion with the formation-factor /
surface-conductivity chain of hem2chloride on HEM test lines, and validate both.

Direct conversion (this repository, fit_direct_ecbulk_chloride.py):
    HEM resistivity rho (sampled from its log-normal uncertainty, as in hem2chloride)
    -> ECbulk = 10 / rho [mS/cm] -> Cl from the direct relation
    * generic:          one relation, geology unaware
    * formation-aware:  relation of the REGIS unit at the HEM layer midpoint, else of its
                        formation, else generic (see fit_direct_ecbulk_chloride.py)

Chain (chloride_uncertainties, compare_cl_relations.py; results read from its cache):
    rho -> ECbulk -> pore-water EC via GeoTOP lithology, formation factor and surface
    conductivity -> Cl with the previous linear relation or the area relation.

Same test set (16 lines in N-B2, NH-B2, Z-C3; HEM layers within GeoTOP) and the same
validation points (measurements >= 1980 within 100 m) as in chloride_uncertainties.
hem2chloride itself is only imported, not changed. Exploratory, not for incorporation.

Run from the repository root (after fit_direct_ecbulk_chloride.py):
    python src/formationfactor_from_xyzdata/compare_direct_vs_hem2chloride.py [n_mc]
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CU_ROOT = Path(os.environ.get("CHLORIDE_UNCERTAINTIES_ROOT", ROOT.parent / "chloride_uncertainties"))
REGIS = Path(os.environ.get("REGIS_NC", r"C:\GISDATA\REGIS\REGIS.nc"))
OUT_DATA = ROOT / "data/4-output/direct_ecbulk_chloride"
OUT_REP = ROOT / "reports/direct_ecbulk_chloride"
FIT_JSON = OUT_DATA / "direct_ecbulk_chloride_fit.json"
N_MC = 200
SEED = 4321
MIN_YEAR = 1980


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ccr = load_module("compare_cl_relations", CU_ROOT / "src/4-analyze/compare_cl_relations.py")
bds = load_module("build_dataset", HERE / "build_dataset.py")
fdr = load_module("fit_direct_ecbulk_chloride", HERE / "fit_direct_ecbulk_chloride.py")
h2c = ccr.h2c
CACHE = ccr.CACHE_DIR

RELS = ("chain_linear", "chain_regional", "direct", "direct_fa")
LABEL = {"chain_linear": "chain, linear (previous)", "chain_regional": "chain, area relation (current)",
         "direct": "direct, generic", "direct_fa": "direct, formation-aware"}
SHORT = {"chain_linear": "CL", "chain_regional": "CR", "direct": "DG", "direct_fa": "DF"}
COL = {"chain_linear": "#6b6a64", "chain_regional": "#eb6834", "direct": "#2a78d6", "direct_fa": "#1baf7a"}
INK, INK2, GRID = ccr.INK, ccr.INK2, ccr.GRID
SCHEMES = {"detail": ccr.CL_CLASSES, "coarse": ccr.CL_CLASSES_COARSE}


# ---------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------


def load_relations() -> tuple[dict, dict]:
    fit = json.loads(FIT_JSON.read_text())
    p = fit["recommended"]["params"]
    generic = {"model": "shiftedpower", **{k: p[k] for k in ("a", "b", "e0", "tau", "s0", "s1", "s2", "nu")},
               "name": "direct_generic"}
    groups = fit["formation_aware"]["groups"]
    rel = {"generic": generic}
    for key, g in groups.items():
        rel[key] = dict(generic, a=g["a"], e0=g["e0"], name=key)
    return rel, groups


def regis_units(df: pd.DataFrame) -> pd.Series:
    """REGIS unit at the midpoint of each HEM layer."""
    pts = pd.DataFrame({"x": df["x"].to_numpy(), "y": df["y"].to_numpy(),
                        "z": 0.5 * (df["top"].to_numpy() + df["bottom"].to_numpy())})
    with xr.open_dataset(REGIS) as ds:
        u = bds.classify_regis(ds, pts)
    return pd.Series(u["regis_layer"].astype(object).where(u["regis_layer"].notna(), None).to_numpy(),
                     index=df.index)


def simulate_ecb(df: pd.DataFrame, n_mc=N_MC, seed=SEED) -> np.ndarray:
    """Bulk EC [mS/cm] realisations from the HEM resistivity uncertainty (as hem2chloride)."""
    np.random.seed(seed)
    ecb = np.empty((len(df), n_mc), dtype=np.float32)
    mu, sd = df["sharp_rho_mean"].to_numpy(), df["sharp_rho_std"].to_numpy()
    for i in range(n_mc):
        rho = h2c.log_normal_discrete(mu, sd, ccr.RHO_CLASSES, ccr.RHO_CLASSES_BNDS)
        ecb[:, i] = 10.0 / np.maximum(rho, 1e-3)
    return ecb


def direct_stats(ecb: np.ndarray, keys: pd.Series, rel: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    gen = ccr.chloride_stats(ecb, rel["generic"])
    fa = pd.DataFrame(index=range(len(ecb)), columns=gen.columns, dtype=float)
    for key, idx in pd.Series(range(len(keys)), index=keys.to_numpy()).groupby(level=0).indices.items():
        rows = np.asarray(idx)
        st = ccr.chloride_stats(ecb[rows], rel.get(key, rel["generic"]))
        fa.iloc[rows] = st.to_numpy()
    return gen, fa


def assemble(base: pd.DataFrame, ecb: np.ndarray, rel: dict, groups: dict) -> pd.DataFrame:
    res = base.reset_index(drop=True).copy()
    for r_old, r_new in (("linear", "chain_linear"), ("regional", "chain_regional")):
        cols = [c for c in res.columns if c.startswith(f"{r_old}_")]
        res = res.rename(columns={c: c.replace(f"{r_old}_", f"{r_new}_", 1) for c in cols})
    res["regis_unit"] = regis_units(res)
    res["relation_key"] = res["regis_unit"].map(lambda u: fdr.relation_for(u, groups))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res["ecb_q50"] = np.nanmedian(ecb, axis=1)
    gen, fa = direct_stats(ecb, res["relation_key"], rel)
    for name, st in (("direct", gen), ("direct_fa", fa)):
        res[[f"{name}_{c}" for c in st.columns]] = st.to_numpy()
    return res


# ---------------------------------------------------------------------
# comparison on the test lines
# ---------------------------------------------------------------------


def compare_area(area: str, rel: dict, groups: dict, n_mc=N_MC) -> pd.DataFrame:
    ab = ccr.AREAS[area]["abbr"]
    cache = OUT_DATA / f"{ab}_lines_direct.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    chain = pd.read_parquet(CACHE / f"{ab}_cl.parquet")
    hem = pd.read_parquet(CACHE / f"{ab}_hem.parquet")[["index", "layer", "sharp_rho_std"]]
    base = chain.merge(hem, on=["index", "layer"], how="left")
    ecb = simulate_ecb(base, n_mc)
    res = assemble(base, ecb, rel, groups)
    res.to_parquet(cache)
    return res


def class_change(res, a, b, q, bounds):
    ca, cb = ccr.cl_class(res[f"{a}_cl_q{q}"], bounds), ccr.cl_class(res[f"{b}_cl_q{q}"], bounds)
    return {"any": float(np.mean(ca != cb)), "lower": float(np.mean(cb < ca)), "higher": float(np.mean(cb > ca))}


def summarise_lines(res: pd.DataFrame) -> dict:
    out = {"n_layers": int(len(res)), "n_soundings": int(res["index"].nunique()),
           "relation_level_share": res["relation_key"].str.split(":").str[0].value_counts(normalize=True)
           .round(3).to_dict(),
           "top_units": res["regis_unit"].value_counts(normalize=True).head(8).round(3).to_dict()}
    for scheme, bounds in SCHEMES.items():
        for ref in ("chain_linear", "chain_regional"):
            for r in ("direct", "direct_fa"):
                out[f"class_change_{scheme}_{r}_vs_{ref}"] = {
                    f"q{q}": class_change(res, ref, r, q, bounds) for q in ccr.CLASS_Q}
        out[f"class_change_{scheme}_direct_fa_vs_direct"] = {
            f"q{q}": class_change(res, "direct", "direct_fa", q, bounds) for q in ccr.CLASS_Q}
        out[f"class_share_q50_{scheme}"] = {
            r: np.bincount(ccr.cl_class(res[f"{r}_cl_q50"], bounds).astype(int),
                           minlength=len(bounds) + 1).tolist() for r in RELS}
    for r in RELS:
        lg = {q: np.log10(np.maximum(res[f"{r}_cl_q{q}"], 10.0)) for q in (5, 25, 75, 95)}
        out[f"log10_iqr_{r}"] = float(np.nanmedian(lg[75] - lg[25]))
        out[f"log10_width90_{r}"] = float(np.nanmedian(lg[95] - lg[5]))
        out[f"median_cl_q50_{r}"] = float(np.nanmedian(res[f"{r}_cl_q50"]))
    lr = np.log10(np.maximum(res["direct_fa_cl_q50"], 10.0) / np.maximum(res["chain_regional_cl_q50"], 10.0))
    out["log10_ratio_direct_fa_vs_chain_regional"] = {
        "median": float(lr.median()), "p10": float(lr.quantile(0.1)), "p90": float(lr.quantile(0.9))}
    for c in ccr.EXCEED:
        for r in ("direct", "direct_fa"):
            d = res[f"{r}_p_ge{int(c)}"] - res[f"chain_regional_p_ge{int(c)}"]
            out[f"dP_ge{int(c)}_{r}_vs_chain_regional"] = {"mean": float(d.mean()), "mean_abs": float(d.abs().mean())}
        zc = ccr.interface_elevation(res, "chain_regional_cl_q50", c)
        zd = ccr.interface_elevation(res, "direct_fa_cl_q50", c)
        both = zc.notna() & zd.notna()
        out[f"interface{int(c)}_direct_fa_vs_chain_regional"] = {
            "share_chain": float(zc.notna().mean()), "share_direct_fa": float(zd.notna().mean()),
            "median_shift_m": float((zd - zc)[both].median()) if both.any() else None,
            "mean_abs_shift_m": float((zd - zc)[both].abs().mean()) if both.any() else None}
    return out


def rank_lines(res: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"line_no": res["line_no"]})
    for q in ccr.CLASS_Q:
        d[f"change_q{q}"] = (ccr.cl_class(res[f"chain_regional_cl_q{q}"]) !=
                             ccr.cl_class(res[f"direct_fa_cl_q{q}"]))
    r = d.groupby("line_no").mean()
    r["class_change"] = r.mean(axis=1)
    return r.sort_values("class_change", ascending=False)


# ---------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------


def validation_area(area: str, rel: dict, groups: dict, n_mc=N_MC) -> pd.DataFrame:
    ab = ccr.AREAS[area]["abbr"]
    cache = OUT_DATA / f"{ab}_val_direct.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    base = pd.read_parquet(CACHE / f"{ab}_val_cl.parquet")
    ecb = simulate_ecb(base, n_mc, seed=SEED + 1)
    res = assemble(base, ecb, rel, groups)
    res.to_parquet(cache)
    return res


def _cluster_ci(x: np.ndarray, groups: np.ndarray, rng, n=2000):
    ug = np.unique(groups)
    idx = {u: np.flatnonzero(groups == u) for u in ug}
    bs = [x[np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])].mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import binomtest

    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0


def validation_scores(res: pd.DataFrame, area: str, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    obs = res["cl_meas"].to_numpy()
    grp = res["sounding"].to_numpy()
    rows = []
    hits = {}
    for r in RELS:
        row = {"area": area, "relation": r, "n": len(res), "n_soundings": int(len(np.unique(grp)))}
        for scheme, bounds in SCHEMES.items():
            cobs = ccr.cl_class(obs, bounds)
            h = ccr.cl_class(res[f"{r}_cl_q50"], bounds) == cobs
            hits[(r, scheme)] = h
            row[f"{scheme}_hit_q50"] = float(h.mean())
            row[f"{scheme}_hit_q50_ci"] = _cluster_ci(h.astype(float), grp, rng)
            c25, c75 = ccr.cl_class(res[f"{r}_cl_q25"], bounds), ccr.cl_class(res[f"{r}_cl_q75"], bounds)
            row[f"{scheme}_within_q25_q75"] = float(np.mean((cobs >= c25) & (cobs <= c75)))
            row[f"{scheme}_within1_q50"] = float(np.mean(np.abs(ccr.cl_class(res[f"{r}_cl_q50"], bounds) - cobs) <= 1))
        for lev, lo, hi in ((50, 25, 75), (90, 5, 95)):
            inside = (obs >= res[f"{r}_cl_q{lo}"].to_numpy()) & (obs <= res[f"{r}_cl_q{hi}"].to_numpy())
            hits[(r, f"cov{lev}")] = inside
            row[f"coverage_{lev}"] = float(inside.mean())
            row[f"coverage_{lev}_ci"] = _cluster_ci(inside.astype(float), grp, rng)
        for c in ccr.EXCEED:
            row[f"brier_{int(c)}"] = float(np.mean((res[f"{r}_p_ge{int(c)}"] - (obs >= c)) ** 2))
        err = np.log10(np.maximum(res[f"{r}_cl_q50"], 10.0)) - np.log10(np.maximum(obs, 10.0))
        row["median_log10_error"] = float(np.median(err))
        row["mean_abs_log10_error"] = float(np.mean(np.abs(err)))
        rows.append(row)
    out = pd.DataFrame(rows)
    # paired tests against the current chain (area relation)
    for key in ("coarse", "detail", "cov50", "cov90"):
        out[f"p_mcnemar_{key}_vs_chain_regional"] = [
            _mcnemar(hits[("chain_regional", key)], hits[(r, key)]) if r != "chain_regional" else np.nan
            for r in RELS]
    return out


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------


def fig_line(area, res, line_no, path):
    d = res[res.line_no == line_no].copy()
    dist = ccr._distance(d)
    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True, constrained_layout=True)
    cmap = ListedColormap(ccr.CL_COL)
    bounds = [0] + ccr.CL_LEV + [ccr.MAX_CL]
    norm = BoundaryNorm(bounds, cmap.N)
    for ax, r in zip(axes[:2], ("chain_regional", "direct_fa")):
        pc = ccr._section(ax, d, dist, d[f"{r}_cl_q50"], cmap, norm)
        ax.set_title(f"Median Cl, {LABEL[r]}", loc="left", fontsize=10, color=INK)
    cb = fig.colorbar(pc, ax=axes[:2], shrink=0.9, aspect=25, pad=0.01, spacing="uniform", ticks=bounds[1:-1])
    cb.ax.set_yticklabels([f"{v:,.0f}" for v in bounds[1:-1]])
    cb.set_label("Cl [mg/l]", fontsize=9)
    cb.ax.tick_params(labelsize=7)
    dcmap = ListedColormap([c for _, c in ccr.DIFF_CATS])
    dnorm = BoundaryNorm(np.arange(len(ccr.DIFF_CATS) + 1) - 0.5, dcmap.N)
    for ax, q in zip(axes[2:], ccr.CLASS_Q):
        cat = ccr.class_difference(d[f"chain_regional_cl_q{q}"], d[f"direct_fa_cl_q{q}"])
        ccr._section(ax, d, dist, cat, dcmap, dnorm)
        ax.set_title(f"Class of Q{q}: direct formation-aware vs chain (area relation)  "
                     f"({np.mean(cat != 3):.0%} of layers change)", loc="left", fontsize=10, color=INK)
    for ax in axes:
        ax.set_ylabel("z [m NAP]", fontsize=9)
        ccr._style(ax)
    axes[-1].set_xlabel("distance along line [km] (gaps shortened)", fontsize=9)
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in ccr.DIFF_CATS]
    fig.legend(handles=handles, loc="outside lower center", ncol=7, fontsize=8, frameon=False,
               title="detail class with the direct conversion, compared with the chain", title_fontsize=8)
    fig.suptitle(f"{ccr.AREAS[area]['label']}, line {line_no}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_overview(results: dict, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True)
    for j, (area, res) in enumerate(results.items()):
        ax = axes[0, j]
        for k, r in enumerate(RELS):
            cls = ccr.cl_class(res[f"{r}_cl_q50"])
            bottom = 0.0
            for c, name in enumerate(ccr.CLASS_NAMES):
                v = float(np.mean(cls == c))
                ax.bar(k, v, bottom=bottom, color=ccr.CLASS_COL[name], edgecolor="white", lw=0.6, width=0.8,
                       label=name if k == 0 else None)
                bottom += v
        ax.set_xticks(range(len(RELS)), [SHORT[r] for r in RELS], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("share of HEM layers", fontsize=9)
        ax.set_title(f"{ccr.AREAS[area]['label']}: class of median Cl", loc="left", fontsize=10, color=INK)
        if j == 2:
            h, l = ax.get_legend_handles_labels()
            ax.legend(h[::-1], l[::-1], fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1, 1),
                      title="Cl [mg/l]", title_fontsize=8)
        ccr._style(ax)
        ax = axes[1, j]
        bins = np.logspace(-1.2, 1.3, 26)
        b = pd.cut(res["ecb_q50"], bins)
        for r in RELS:
            m = res.groupby(b, observed=True)[f"{r}_cl_q50"].median()
            n = res.groupby(b, observed=True).size()
            m = m[n >= 30]
            ax.plot([iv.mid for iv in m.index], np.maximum(m.values, 1), color=COL[r], lw=1.8, label=LABEL[r])
        ax.set(xscale="log", yscale="log")
        ax.set_ylim(1, 3e4)
        ax.set_xlabel("median HEM bulk EC [mS/cm]", fontsize=9)
        ax.set_ylabel("median of the layer medians, Cl [mg/l]", fontsize=9)
        ax.grid(color=GRID, lw=0.6)
        if j == 0:
            ax.legend(fontsize=7.5, frameon=False)
        ccr._style(ax)
    fig.suptitle("CL chain + previous linear relation, CR chain + area relation, DG direct generic, "
                 "DF direct formation-aware", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_validation(scores: pd.DataFrame, val: dict, path: Path):
    areas = list(val)
    fig, axes = plt.subplots(3, len(areas), figsize=(4.6 * len(areas), 10), constrained_layout=True, squeeze=False)
    for j, area in enumerate(areas):
        s = scores[scores.area == area].set_index("relation")
        ax = axes[0, j]
        for i, (m, lab) in enumerate((("coarse_hit_q50", "coarse"), ("detail_hit_q50", "detail"))):
            for k, r in enumerate(RELS):
                x = i + (k - 1.5) * 0.2
                v = s.loc[r, m]
                lo, hi = s.loc[r, f"{m}_ci"]
                ax.bar(x, v, width=0.18, color=COL[r], label=LABEL[r] if i == 0 else None)
                ax.plot([x, x], [lo, hi], color=INK, lw=0.9)
        ax.set_xticks([0, 1], ["coarse classes", "detail classes"], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("median in measured class", fontsize=9)
        ax.set_title(f"{ccr.AREAS[area]['label']}, n = {int(s['n'].iloc[0])} "
                     f"({int(s['n_soundings'].iloc[0])} soundings)", loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        if j == 0:
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        ccr._style(ax)
        ax = axes[1, j]
        for i, lev in enumerate((50, 90)):
            for k, r in enumerate(RELS):
                x = i + (k - 1.5) * 0.2
                lo, hi = s.loc[r, f"coverage_{lev}_ci"]
                ax.bar(x, s.loc[r, f"coverage_{lev}"], width=0.18, color=COL[r])
                ax.plot([x, x], [lo, hi], color=INK, lw=0.9)
            ax.plot([i - 0.45, i + 0.45], [lev / 100] * 2, color=INK, lw=1.2, ls=(0, (3, 2)))
        ax.set_xticks([0, 1], ["50% interval", "90% interval"], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("share inside (95% CI)", fontsize=9)
        ax.set_title("Interval coverage (dashed: nominal)", loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ccr._style(ax)
        ax = axes[2, j]
        res = val[area]
        cobs = ccr.cl_class(res["cl_meas"], ccr.CL_CLASSES_COARSE)
        errs = [("too fresh", "#2a78d6"), ("measured class", "#c8c7c1"), ("too saline", "#eb6834")]
        for k, r in enumerate(RELS):
            e = np.sign(ccr.cl_class(res[f"{r}_cl_q50"], ccr.CL_CLASSES_COARSE) - cobs)
            bottom = 0.0
            for v_, (lab, colr) in zip((-1, 0, 1), errs):
                frac = float(np.mean(e == v_))
                ax.bar(k, frac, bottom=bottom, color=colr, width=0.7, edgecolor="white",
                       label=lab if k == 0 else None)
                if frac > 0.06:
                    ax.text(k, bottom + frac / 2, f"{frac:.0%}", ha="center", va="center", fontsize=7.5, color=INK)
                bottom += frac
        ax.set_xticks(range(len(RELS)), [SHORT[r] for r in RELS], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title("Median vs measured coarse class", loc="left", fontsize=10, color=INK)
        if j == len(areas) - 1:
            h, l = ax.get_legend_handles_labels()
            ax.legend(h[::-1], l[::-1], fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
        ccr._style(ax)
    fig.suptitle(f"Validation (measurements >= {MIN_YEAR}); error bars: 95% cluster-bootstrap CI over soundings",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------


def main(n_mc=N_MC, n_plot_lines=1):
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REP.mkdir(parents=True, exist_ok=True)
    rel, groups = load_relations()
    results, summary = {}, {}
    for area in ccr.AREAS:
        print(f"*** lines {area}", flush=True)
        res = compare_area(area, rel, groups, n_mc)
        results[area] = res
        summary[area] = summarise_lines(res)
        rk = rank_lines(res)
        rk.to_csv(OUT_REP / f"line_ranking_{ccr.AREAS[area]['abbr']}.csv")
        for ln in rk.index[:n_plot_lines]:
            fig_line(area, res, int(ln), OUT_REP / f"section_{ccr.AREAS[area]['abbr']}_line{int(ln)}.png")
    fig_overview(results, OUT_REP / "fig_overview.png")

    val, sc = {}, []
    for area in ccr.AREAS:
        print(f"*** validation {area}", flush=True)
        v = validation_area(area, rel, groups, n_mc)
        v = v[pd.to_datetime(v["datetime"], errors="coerce").dt.year >= MIN_YEAR].reset_index(drop=True)
        val[area] = v
        sc.append(validation_scores(v, area))
    scores = pd.concat(sc, ignore_index=True)
    scores.to_csv(OUT_REP / "validation_table.csv", index=False)
    fig_validation(scores, val, OUT_REP / "fig_validation.png")
    summary["validation"] = scores.to_dict(orient="records")
    summary["settings"] = {"n_mc": n_mc, "min_year": MIN_YEAR, "seed": SEED,
                           "chain_results": str(CACHE), "regis": str(REGIS)}
    (OUT_REP / "comparison_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    cols = ["area", "relation", "n", "coarse_hit_q50", "detail_hit_q50", "coverage_50", "coverage_90",
            "brier_150", "brier_1000", "median_log10_error", "mean_abs_log10_error",
            "p_mcnemar_coarse_vs_chain_regional", "p_mcnemar_detail_vs_chain_regional",
            "p_mcnemar_cov50_vs_chain_regional"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(scores[cols])
    return results, summary


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_MC)
