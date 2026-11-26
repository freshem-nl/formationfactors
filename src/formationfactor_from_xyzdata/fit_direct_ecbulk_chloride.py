"""Test avenue: a direct relation between bulk EC and chloride (bypassing formation
factor and surface conductivity).

Fits a heteroscedastic relation Cl = f(ECbulk) to the collocated chloride - borehole/
CPT conductivity pairs of ``build_dataset.py``
(data/3-input/formationfactor_from_xyzdata/chloride_geophysics_regis.csv), using the
model library of the chloride_uncertainties repository (fit_ec_chloride.py, read only):

    ln Cl ~ a + b ln(tau softplus((ECb - e0) / tau)) + sigma(ECb) T(nu),
    ln sigma = s0 + s1 u + s2 u^2,  u = ln ECb

ECbulk [mS/cm] = 10 * conductivity [S/m] at in-situ temperature, i.e. without a
temperature correction, the same quantity hem2chloride derives from the HEM
resistivity (10 / rho), so the relation can be applied to HEM directly.

Chloride samples at or near the validation locations of the HEM comparison
(compare_direct_vs_hem2chloride.py) are held out of the fit.

This is an exploratory test, not meant for incorporation in hem2chloride.

Run from the repository root:
    python src/formationfactor_from_xyzdata/fit_direct_ecbulk_chloride.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
CU_ROOT = Path(os.environ.get("CHLORIDE_UNCERTAINTIES_ROOT", ROOT.parent / "chloride_uncertainties"))
INPUT = ROOT / "data/3-input/formationfactor_from_xyzdata/chloride_geophysics_regis.csv"
OUT_DATA = ROOT / "data/4-output/direct_ecbulk_chloride"
OUT_REP = ROOT / "reports/direct_ecbulk_chloride"
FIT_JSON = OUT_DATA / "direct_ecbulk_chloride_fit.json"
VAL_CACHE = CU_ROOT / "data/2-interim/cl_relation_testset"  # validation points of the HEM comparison

MAX_COND_S_M = 2.5  # bulk conductivity above that of sea-water saturated sediment: implausible
HOLDOUT_RADIUS_M = 500.0  # hold out chloride samples within this distance of a validation point
N_BOOT = 200
SEED = 1

C = {"sand": "#2a78d6", "clay": "#eb6834", "complex": "#1baf7a", "other": "#6b6a64"}
INK, INK2, MUTED, GRID = "#1f1f1d", "#3a3a37", "#9b9a94", "#e6e5e0"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fec = load_module("fit_ec_chloride", CU_ROOT / "src/4-analyze/fit_ec_chloride.py")


def lithology_group(code) -> str:
    """Sand / clay / complex unit from the REGIS layer code (diagnostic only)."""
    if not isinstance(code, str):
        return "other"
    if re.search(r"z\d*$", code):
        return "sand"
    if re.search(r"k\d*$", code):
        return "clay"
    if re.search(r"c$", code):
        return "complex"
    return "other"


def validation_points() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(VAL_CACHE.glob("*_val_points.parquet"))]
    if not frames:
        raise FileNotFoundError(f"no validation points in {VAL_CACHE}; run compare_cl_relations.py there")
    return pd.concat(frames, ignore_index=True)


def load_pairs() -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(INPUT, dtype={"locationid": str, "locationcode": str, "log_id": str})
    info = {"input_pairs": len(d)}
    d = d[(d["conductivity_s_m"] > 0) & (d["conductivity_s_m"] <= MAX_COND_S_M) & (d["chloride_g_l"] > 0)]
    info["after_plausibility"] = len(d)
    d = d.assign(
        ecb_ms_cm=10.0 * d["conductivity_s_m"],
        cl_mg_l=1000.0 * d["chloride_g_l"],
        nitg=d["log_source"] + ":" + d["log_id"].astype(str),  # cluster = log (bootstrap unit)
        well=d["locationcode"].str[:8],
        litho=d["chloride_regis_layer"].map(lithology_group),
        year=pd.to_datetime(d["datetime"], errors="coerce").dt.year,
    )
    # hold out the validation wells and everything within HOLDOUT_RADIUS_M of them
    v = validation_points()
    near = cKDTree(v[["x_meas", "y_meas"]].to_numpy()).query(
        d[["chloride_x_rd_m", "chloride_y_rd_m"]].to_numpy(), k=1)[0] <= HOLDOUT_RADIUS_M
    same = d["well"].isin(v["locationcode"].str[:8]) | d["locationid"].isin(v["locationid"].astype(str))
    info["held_out_validation"] = int((near | same).sum())
    d = d[~(near | same)].reset_index(drop=True)
    info["fit_pairs"] = len(d)
    info["fit_logs"] = int(d["nitg"].nunique())
    info["fit_wells"] = int(d["well"].nunique())
    return d, info


def fit_models(d: pd.DataFrame) -> dict:
    out = {}
    for mean in ("loglinear", "shiftedpower"):
        for var in ("constant", "quadratic"):
            f = fec.fit_heteroscedastic(d["ecb_ms_cm"], d["cl_mg_l"], mean=mean, dist="t", var=var,
                                        wells=d["nitg"])
            out[f"{mean}_t_{var}"] = f
    return out


def plot_fit(d: pd.DataFrame, fit, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    x = np.logspace(np.log10(d.ecb_ms_cm.min()), np.log10(d.ecb_ms_cm.max()), 200)
    for ax, logy in zip(axes, (True, False)):
        for g in ("sand", "complex", "clay", "other"):
            s = d[d.litho == g]
            ax.scatter(s.ecb_ms_cm, s.cl_mg_l, s=7, alpha=0.45, color=C[g], lw=0, label=f"{g} (n = {len(s)})",
                       rasterized=True)
        for lo, hi, a in ((0.05, 0.95, 0.12), (0.25, 0.75, 0.22)):
            ax.fill_between(x, fit.quantile(x, lo), fit.quantile(x, hi), color=INK, alpha=a, lw=0,
                            label=f"{round(100 * (hi - lo))}% interval")
        ax.plot(x, fit.predict(x), color=INK, lw=1.8, label="median")
        ax.set_xscale("log")
        ax.set_yscale("log" if logy else "linear")
        if not logy:
            ax.set_ylim(0, 5000)
            ax.set_xlim(0.3, 20)
        ax.set_xlabel("bulk EC (in situ) [mS/cm]", fontsize=9)
        ax.set_ylabel("Cl [mg/l]", fontsize=9)
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK2)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left", markerscale=2.5)
    axes[0].set_title("Direct relation, log scale", loc="left", fontsize=10, color=INK)
    axes[1].set_title("Linear chloride axis, 0.3-20 mS/cm", loc="left", fontsize=10, color=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Formation-aware version
# ---------------------------------------------------------------------

N_MIN_PAIRS = 50  # minimum collocated pairs for a unit / formation relation ...
N_MIN_LOGS = 10  # ... from at least this many logs
N_MIN_BRACKISH = 15  # pairs with Cl >= 150 mg/l needed to also fit the background shift e0
E0_BOUNDS = (0.0, 5.0)


def formation_of(unit) -> str | None:
    """REGIS formation code: unit code without its lithological suffix (z2, k1, c)."""
    if not isinstance(unit, str):
        return None
    return re.sub(r"([zk]\d*|c)$", "", unit) or None


def fit_group(d: pd.DataFrame, base) -> tuple[dict, list[str]]:
    """Group-specific level a (and background shift e0 when there is brackish data),
    with the shape (b, tau), scale and nu of the generic relation ``base``."""
    from scipy import optimize, stats

    x, z = d.ecb_ms_cm.to_numpy(), np.log(d.cl_mg_l.to_numpy())
    p = dict(base.params)
    free = ["a", "e0"] if (d.cl_mg_l >= 150).sum() >= N_MIN_BRACKISH else ["a"]
    sc = base.scale(x)

    def nll(theta):
        q = dict(p, **dict(zip(free, theta)))
        loc = fec._shifted_power([q["a"], q["b"], q["e0"], np.log(q["tau"])], x)
        return -stats.t.logpdf((z - loc) / sc, q["nu"]).sum()

    bounds = [(None, None)] + ([E0_BOUNDS] if "e0" in free else [])
    r = optimize.minimize(nll, [p[k] for k in free], method="L-BFGS-B", bounds=bounds)
    p.update(dict(zip(free, r.x)))
    return p, free


def fit_formation_aware(d: pd.DataFrame, base) -> dict:
    """Relations per REGIS unit and formation where the data suffice; see ``relation_for``."""
    d = d.assign(formation=d["chloride_regis_layer"].map(formation_of))
    groups = {}
    for level, col in (("unit", "chloride_regis_layer"), ("formation", "formation")):
        for key, g in d.groupby(col):
            if len(g) >= N_MIN_PAIRS and g.nitg.nunique() >= N_MIN_LOGS:
                params, free = fit_group(g, base)
                groups[f"{level}:{key}"] = {
                    "level": level, "key": key, "n": int(len(g)), "n_logs": int(g.nitg.nunique()),
                    "n_cl_ge_150": int((g.cl_mg_l >= 150).sum()), "free": free,
                    "a": float(params["a"]), "e0": float(params["e0"]),
                }
    return groups


def relation_for(unit, groups: dict) -> str:
    """Key of the relation for a REGIS unit: unit -> formation -> generic."""
    if isinstance(unit, str) and f"unit:{unit}" in groups:
        return f"unit:{unit}"
    f = formation_of(unit)
    if f is not None and f"formation:{f}" in groups:
        return f"formation:{f}"
    return "generic"


def group_fit(base, g: dict | None):
    """FitResult of a group (generic parameters with the group's a and e0)."""
    import dataclasses

    if g is None:
        return base
    return dataclasses.replace(base, params=dict(base.params, a=g["a"], e0=g["e0"]))


def cross_validate(d: pd.DataFrame, k=5, seed=SEED) -> pd.DataFrame:
    """Grouped (by log) k-fold CV: log predictive density of generic vs formation-aware."""
    rng = np.random.default_rng(seed)
    logs = d.nitg.unique()
    fold = dict(zip(logs, rng.permutation(len(logs)) % k))
    f_idx = d.nitg.map(fold).to_numpy()
    rows = []
    for i in range(k):
        tr, te = d[f_idx != i], d[f_idx == i]
        base = fec.fit_heteroscedastic(tr.ecb_ms_cm, tr.cl_mg_l, "shiftedpower", "t", "quadratic", se=False)
        groups = fit_formation_aware(tr, base)
        keys = te["chloride_regis_layer"].map(lambda u: relation_for(u, groups))
        lp_gen = base.logpdf(te.ecb_ms_cm.to_numpy(), te.cl_mg_l.to_numpy())
        lp_fa = np.empty(len(te))
        for key, idx in te.groupby(keys.to_numpy()).indices.items():
            f = group_fit(base, groups.get(key))
            lp_fa[idx] = f.logpdf(te.ecb_ms_cm.to_numpy()[idx], te.cl_mg_l.to_numpy()[idx])
        rows.append(pd.DataFrame({"fold": i, "litho": te.litho.to_numpy(), "key": keys.to_numpy(),
                                  "lp_generic": lp_gen, "lp_formation": lp_fa,
                                  "nitg": te.nitg.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def residual_table(d: pd.DataFrame, fit) -> pd.DataFrame:
    """Median residual (ln Cl - location) per lithology group and bulk-EC range."""
    r = np.log(d.cl_mg_l) - fit.location(d.ecb_ms_cm.to_numpy())
    b = pd.cut(d.ecb_ms_cm, [0, 0.5, 1, 2, 5, 100])
    t = pd.DataFrame({"litho": d.litho, "ecb": b, "res": r}).groupby(["litho", "ecb"], observed=True)["res"]
    return pd.DataFrame({"n": t.size(), "median_residual_ln": t.median().round(3)}).reset_index()


def main():
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REP.mkdir(parents=True, exist_ok=True)
    d, info = load_pairs()
    print(info)
    fits = fit_models(d)
    aic = {k: round(f.aic, 1) for k, f in fits.items()}
    print("AIC:", aic)
    rec = fits["shiftedpower_t_quadratic"]
    boot = fec.bootstrap_by_well(d.rename(columns={"ecb_ms_cm": "ec25_ms_cm"}), rec, n_boot=N_BOOT, seed=SEED)
    res_tab = residual_table(d, rec)
    res_tab.to_csv(OUT_REP / "residuals_by_lithology.csv", index=False)
    plot_fit(d, rec, OUT_REP / "fig_direct_relation.png")
    # coverage of the predictive intervals per bulk-EC range
    cov = fec.coverage_table(rec, d.rename(columns={"ecb_ms_cm": "ec25_ms_cm"}),
                             bins=[0, 0.5, 1, 2, 5, 100])
    cov.to_csv(OUT_REP / "coverage_direct_relation.csv")
    # formation-aware version
    groups = fit_formation_aware(d, rec)
    keys = d["chloride_regis_layer"].map(lambda u: relation_for(u, groups))
    cv = cross_validate(d)
    dl = cv.lp_formation - cv.lp_generic
    per_log = cv.assign(dl=dl).groupby("nitg")["dl"].sum()
    cv_sum = {
        "mean_lpd_generic": float(cv.lp_generic.mean()), "mean_lpd_formation": float(cv.lp_formation.mean()),
        "delta_mean": float(dl.mean()),
        # cluster-robust SE of the mean difference (logs as clusters)
        "delta_se": float(np.sqrt(len(per_log) / (len(per_log) - 1) * ((per_log - per_log.sum() / len(dl) *
                          cv.groupby("nitg").size()) ** 2).sum()) / len(dl)),
        "by_litho": cv.assign(dl=dl).groupby("litho")["dl"].agg(["size", "mean"]).round(4).to_dict("index"),
        "share_fallback_generic": float((cv.key == "generic").mean()),
    }
    print("formation-aware groups:", len(groups), " CV:", {k: v for k, v in cv_sum.items() if k != "by_litho"})
    print(pd.DataFrame(cv_sum["by_litho"]).T)
    out = {
        "data": info,
        "formation_aware": {
            "rule": f"REGIS unit, else formation, if >= {N_MIN_PAIRS} pairs from >= {N_MIN_LOGS} logs, "
                    f"else generic; a free, e0 free if >= {N_MIN_BRACKISH} pairs with Cl >= 150 mg/l",
            "groups": groups,
            "share_pairs": keys.str.split(":").str[0].value_counts(normalize=True).round(3).to_dict(),
            "cv": cv_sum,
        },
        "settings": {"max_conductivity_s_m": MAX_COND_S_M, "holdout_radius_m": HOLDOUT_RADIUS_M,
                     "x": "ECbulk = 10 * conductivity [S/m], in situ, mS/cm", "n_boot": N_BOOT},
        "aic": aic,
        "recommended": {"name": "shiftedpower_t_quadratic", **rec.to_dict(),
                        "boot_lo": boot.quantile(0.025).to_dict(), "boot_hi": boot.quantile(0.975).to_dict()},
        "lithology_counts": d.litho.value_counts().to_dict(),
        "share_cl_ge_150": float((d.cl_mg_l >= 150).mean()),
        "share_cl_ge_1000": float((d.cl_mg_l >= 1000).mean()),
    }
    FIT_JSON.write_text(json.dumps(out, indent=1, default=float))
    (OUT_REP / "direct_relation_fit.json").write_text(json.dumps(out, indent=1, default=float))
    p = rec.params
    print({k: round(p[k], 3) for k in ("a", "b", "e0", "tau", "s0", "s1", "s2", "nu")})
    print(res_tab.to_string(index=False))
    print(cov)


if __name__ == "__main__":
    main()
