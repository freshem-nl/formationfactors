"""Fit and plot ground conductivity against groundwater chloride concentration."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import linregress

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "3-input" / "formationfactor_from_xyzdata" / "chloride_geophysics_regis.csv"
OUTPUT = ROOT / "reports" / "figures" / "formationfactor_from_xyzdata"
X, Y = "chloride_g_l", "conductivity_s_m"
GROUP = ["log_source", "log_id"]
COLORS = {"LAS": "#157789", "ECPT": "#b34e27"}


def coefficients(sums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, sx, sy, sxx, sxy = sums.T
    denom = n * sxx - sx**2
    slope = np.divide(n * sxy - sx * sy, denom, out=np.full_like(denom, np.nan), where=denom > 0)
    return (sy - slope * sx) / n, slope


def regress(data: pd.DataFrame, rng: np.random.Generator) -> tuple[dict, np.ndarray, np.ndarray]:
    x, y = data[X].to_numpy(), data[Y].to_numpy()
    fit = linregress(x, y)
    sums = data.assign(n=1, xx=x**2, xy=x*y).groupby(GROUP)[["n", X, Y, "xx", "xy"]].sum()
    if len(sums) < 3:
        raise ValueError("At least three logs are needed for bootstrap and held-out evaluation.")

    # Keep the number of logs per source fixed, not the number of rows per log.
    bootstrap_sums = np.zeros((5000, sums.shape[1]))
    for _, source in sums.groupby(level="log_source"):
        n = len(source)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=len(bootstrap_sums))
        bootstrap_sums += weights @ source.to_numpy()
    intercepts, slopes = coefficients(bootstrap_sums)
    valid = np.isfinite(slopes)
    if valid.sum() < 0.99 * len(bootstrap_sums):
        raise ValueError("Too many bootstrap samples have no variation in chloride.")
    intercepts, slopes = intercepts[valid], slopes[valid]

    # Evaluate each observation using a model trained without any rows from its log.
    loo_intercepts, loo_slopes = coefficients(sums.sum().to_numpy() - sums.to_numpy())
    heldout = pd.DataFrame({"intercept": loo_intercepts, "slope": loo_slopes}, index=sums.index)
    heldout = heldout.reindex(pd.MultiIndex.from_frame(data[GROUP]))
    errors = y - (heldout["intercept"].to_numpy() + heldout["slope"].to_numpy() * x)
    if not np.isfinite(errors).all():
        raise ValueError("A leave-one-log-out fit has no variation in chloride.")
    lo_b, hi_b = np.quantile(slopes, [0.025, 0.975])
    lo_a, hi_a = np.quantile(intercepts, [0.025, 0.975])
    stats = {
        "n_pairs": len(data), "n_logs": len(sums),
        "intercept_s_m": fit.intercept, "intercept_ci_low_s_m": lo_a, "intercept_ci_high_s_m": hi_a,
        "slope_s_m_per_g_l": fit.slope, "slope_ci_low": lo_b, "slope_ci_high": hi_b,
        "r_squared": fit.rvalue**2,
        "heldout_mae_s_m": np.abs(errors).mean(),
        "heldout_rmse_s_m": np.sqrt(np.mean(errors**2)),
        "heldout_90pct_absolute_error_s_m": np.quantile(np.abs(errors), 0.9),
        "bootstrap_draws_used": int(valid.sum()),
    }
    return stats, intercepts, slopes


def plot_panel(ax, data, result, title, xmax, ymax=None):
    stats, intercepts, slopes = result
    for source, rows in data.groupby("log_source"):
        ax.scatter(rows[X], rows[Y], s=10 if source == "LAS" else 23,
                   color=COLORS[source], alpha=0.28 if source == "LAS" else 0.7,
                   edgecolors="none", label=source)
    grid = np.linspace(data[X].min(), data[X].max(), 200)
    low, high = np.quantile(intercepts[:, None] + slopes[:, None] * grid, [0.025, 0.975], axis=0)
    ax.fill_between(grid, low, high, color="#4d6077", alpha=0.23, label="95% confidence band (mean)")
    ax.plot(grid, stats["intercept_s_m"] + stats["slope_s_m_per_g_l"] * grid,
            color="#1e293b", lw=1.6, label="Linear regression")
    ax.set(title=title, xlim=(0, xmax))
    if ymax is not None:
        ax.set_ylim(bottom=min(0.0, float(low.min())), top=ymax)
    ax.grid(alpha=0.20)
    ax.set_axisbelow(True)
    label = f"{stats['n_pairs']:,} pairs | {stats['n_logs']} logs | $R^2$ = {stats['r_squared']:.3f}"
    if ymax is not None:
        label += f"\n{(data[Y] > ymax).sum()} observations above plot range"
    ax.text(0.98, 0.04, label, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = pd.read_csv(args.input, dtype={"log_id": str, "same_regis_layer": "boolean"})
    if (
        not np.isfinite(data[[X, Y]]).all().all() or not (data[X].ge(0) & data[Y].gt(0)).all()
        or data[GROUP].isna().any().any() or set(data["log_source"]) != set(COLORS)
    ):
        raise ValueError("Expected finite nonnegative chloride, positive conductivity and LAS/ECPT log IDs.")

    strict = data["same_regis_layer"].fillna(False) & data["screen_gap_m"].eq(0) & data["xy_distance_m"].le(25)
    subsets = {"Pooled": data, "LAS": data[data["log_source"].eq("LAS")],
               "ECPT": data[data["log_source"].eq("ECPT")], "Strict match": data.loc[strict]}
    rng = np.random.default_rng(42)
    fits = {name: regress(subset, rng) for name, subset in subsets.items()}
    summary = pd.DataFrame({name: result[0] for name, result in fits.items()}).T
    summary = summary.astype({"n_pairs": int, "n_logs": int, "bootstrap_draws_used": int})

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    fig.suptitle("Ground conductivity versus groundwater chloride", fontsize=17, weight="bold")
    fig.supxlabel("Groundwater chloride (g/L)")
    fig.supylabel("Ground conductivity (S/m)")
    xmax = data[X].max() * 1.035
    plot_panel(axes[0, 0], data, fits["Pooled"], "Pooled | full range", xmax)
    plot_panel(axes[0, 1], data, fits["Pooled"], "Pooled | zoom (same fit)", xmax, ymax=1)
    plot_panel(axes[1, 0], subsets["LAS"], fits["LAS"], "LAS only", xmax)
    plot_panel(axes[1, 1], subsets["ECPT"], fits["ECPT"], "ECPT only", xmax)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "chloride_conductivity_regression.png", dpi=180)
    plt.close(fig)
    summary.to_csv(args.output / "chloride_conductivity_regression.csv", index_label="subset")
    print(summary.to_string(float_format=lambda value: f"{value:.5g}"))


if __name__ == "__main__":
    main()
