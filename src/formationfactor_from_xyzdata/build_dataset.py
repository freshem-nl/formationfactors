"""Pair chloride screens with electrical log samples and REGIS II units."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / r"data\1-external\geophys_data"
CHLORIDE = Path(
    r"C:\Users\delsman\Nextcloud\Team4-Chloride\Validation_dataset"
    r"\FM_validation_dataset_chloride_final.csv"
)
REGIS = Path(r"C:\GISDATA\REGIS\REGIS.nc")
CHLORIDE_COLUMNS = {
    "x": "chloride_x_rd_m", "y": "chloride_y_rd_m", "z": "chloride_z_nap_m",
    "filter_top": "screen_top_nap_m", "filter_bottom": "screen_bottom_nap_m",
    "value": "chloride_g_l", "regis_layer": "input_regis_layer",
    "regis_description": "input_regis_description",
}
COMMON_LOG_COLUMNS = {
    "depth_m": "log_depth_m", "ec_sm": "conductivity_s_m",
    "notes": "log_notes", "notes_long": "log_notes_long",
}
LOG_COLUMNS = {
    "ECPT": COMMON_LOG_COLUMNS | {
        "x_rd": "log_x_rd_m", "y_rd": "log_y_rd_m", "ecpt_id": "log_id",
        "surface_level_m": "log_surface_nap_m",
    },
    "LAS": COMMON_LOG_COLUMNS | {
        "x": "log_x_rd_m", "y": "log_y_rd_m", "las_id": "log_id",
        "surface_level": "log_surface_nap_m", "z": "log_z_nap_m",
        "curve_name": "log_curve", "file": "log_file", "well_name": "log_well_name",
    },
}
SAMPLE = "chloride_csv_row"
SCREEN = ["screen_bottom_nap_m", "screen_top_nap_m", "chloride_z_nap_m"]
CL_XY = ["chloride_x_rd_m", "chloride_y_rd_m"]
LOG_XY = ["log_x_rd_m", "log_y_rd_m"]
RANK = [
    "screen_gap_m", "xy_distance_m", "sample_z_difference_m", "log_csv_row",
]


def read_chloride(path: Path) -> pd.DataFrame:
    columns = ["locationid", "locationcode", "datetime", "parametercode", "unitabbrev", "owner"]
    frame = pd.read_csv(path, usecols=[*columns, *CHLORIDE_COLUMNS],
                        dtype={"locationid": str, "locationcode": str})
    frame = frame.rename(columns=CHLORIDE_COLUMNS)
    frame[SAMPLE] = frame.index + 2
    valid = (
        frame["parametercode"].eq("Cl") & frame["unitabbrev"].str.lower().eq("g/l")
        & np.isfinite(frame[[*CL_XY, *SCREEN, "chloride_g_l"]]).all(axis=1)
        & frame["chloride_g_l"].ge(0)
        & frame["chloride_z_nap_m"].between(frame[SCREEN[0]], frame[SCREEN[1]])
    )
    if not valid.all():
        raise ValueError(f"{(~valid).sum()} invalid chloride records (units, values or screen geometry)")
    return frame


def best(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(RANK).drop_duplicates(SAMPLE)


def match_logs(
    chloride: pd.DataFrame, path: Path, source: str, radius: float, tolerance: float,
    chunk_size: int = 100_000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = LOG_COLUMNS[source]
    log_fields = list(dict.fromkeys([*columns.values(), "log_z_nap_m", "log_csv_row"]))
    winners = pd.DataFrame(columns=[SAMPLE, *log_fields, *RANK[:-1]])
    tree = cKDTree(chloride[CL_XY].to_numpy())
    stats = {"input": 0, "invalid": 0}
    for raw in pd.read_csv(path, usecols=list(columns), dtype={
        k: str for k, v in columns.items()
        if v in {"log_id", "log_notes", "log_notes_long", "log_curve", "log_file", "log_well_name"}
    }, chunksize=chunk_size):
        stats["input"] += len(raw)
        log = raw.rename(columns=columns)
        log["log_csv_row"] = log.index + 2
        if source == "ECPT":
            log["log_z_nap_m"] = log["log_surface_nap_m"] - log["log_depth_m"]
        valid = (
            np.isfinite(log[[*LOG_XY, "log_z_nap_m", "log_depth_m", "conductivity_s_m"]]).all(axis=1)
            & log["conductivity_s_m"].gt(0) & log["log_depth_m"].ge(0)
            & log["log_id"].notna()
        )
        stats["invalid"] += int((~valid).sum())
        log = log.loc[valid]
        locations = log[LOG_XY].drop_duplicates().reset_index(drop=True)
        candidates = tree.query_ball_point(locations.to_numpy(), r=radius)
        lengths = [len(indices) for indices in candidates]
        if not any(lengths):
            continue
        links = locations.iloc[np.repeat(np.arange(len(locations)), lengths)].reset_index(drop=True)
        screens = chloride.iloc[np.concatenate([i for i in candidates if i])].reset_index(drop=True)
        links = pd.concat([links, screens[[SAMPLE, *SCREEN]]], axis=1)
        links["xy_distance_m"] = np.hypot(
            links[LOG_XY[0]] - screens[CL_XY[0]], links[LOG_XY[1]] - screens[CL_XY[1]],
        )
        pairs = log.merge(links, on=LOG_XY)
        z = pairs["log_z_nap_m"]
        pairs["screen_gap_m"] = np.maximum(0.0, np.maximum(pairs[SCREEN[0]] - z, z - pairs[SCREEN[1]]))
        pairs["sample_z_difference_m"] = (z - pairs[SCREEN[2]]).abs()
        eligible = pairs.loc[pairs["screen_gap_m"].le(tolerance), winners.columns]
        if not eligible.empty:
            batch = best(eligible)
            winners = batch if winners.empty else best(pd.concat([winners, batch], ignore_index=True))
    winners["log_source"] = source
    return winners, stats


def cell_indices(bounds: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (
        bounds.ndim != 2 or bounds.shape[1] != 2 or np.any(bounds[:, 0] >= bounds[:, 1])
        or np.any(bounds[1:, 0] != bounds[:-1, 1])
    ):
        raise ValueError("Expected ascending contiguous REGIS cell bounds")
    indices = np.searchsorted(bounds[:, 1], values, side="right")
    inside = (values >= bounds[0, 0]) & (values < bounds[-1, 1])
    return indices, inside


def layer_at_z(
    tops: np.ndarray, bottoms: np.ndarray, layers: np.ndarray, z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(tops) & np.isfinite(bottoms) & (bottoms != -9999) & (tops != -9999)
    valid &= (tops > bottoms) & (layers != "mv")
    contains = valid & (bottoms < z[:, None]) & (z[:, None] <= tops)
    count = contains.sum(axis=1)
    return np.where(count == 1, contains.argmax(axis=1), -1), count


def classify_regis(ds: xr.Dataset, points: pd.DataFrame) -> pd.DataFrame:
    """Sample containing CF grid cells using the explicit bounds, not x/y as centres."""
    for name in ("top", "bottom"):
        if ds[name].attrs.get("units") != "m-NAP" or set(ds[name].dims) != {"layer", "y", "x"}:
            raise ValueError(f"Unexpected REGIS geometry for {name}; expected layer/y/x in m-NAP")
    if str(ds["crs"].attrs.get("epsg_code", "")).strip('"') != "EPSG:28992":
        raise ValueError("Expected REGIS in EPSG:28992")
    xb, yb = ds["x_bounds"].values, ds["y_bounds"].values
    ix, inside_x = cell_indices(xb, points["x"].to_numpy())
    iy, inside_y = cell_indices(yb, points["y"].to_numpy())
    inside = inside_x & inside_y
    layers = ds["layer"].values.astype(str)
    out = pd.DataFrame(index=np.arange(len(points)))
    out["regis_cell_x_rd_m"] = np.where(inside, xb[np.minimum(ix, len(xb)-1)].mean(axis=1), np.nan)
    out["regis_cell_y_rd_m"] = np.where(inside, yb[np.minimum(iy, len(yb)-1)].mean(axis=1), np.nan)
    out["regis_status"] = np.where(inside, "no_layer_at_elevation", "outside_grid")
    out["regis_layer"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out[["regis_top_nap_m", "regis_bottom_nap_m"]] = np.nan
    if "screen_top" in points:
        out["screen_overlap_fraction"] = np.nan
    # Read each intersecting y strip once; on-disk chunks span the full x axis.
    y_chunk = ds["top"].encoding.get("preferred_chunks", {}).get("y", 23)
    for band in np.unique(iy[inside] // y_chunk):
        rows = np.flatnonzero(inside & (iy // y_chunk == band))
        x0, x1 = ix[rows].min(), ix[rows].max() + 1
        y0, y1 = iy[rows].min(), iy[rows].max() + 1
        subset = ds[["top", "bottom"]].isel(x=slice(x0, x1), y=slice(y0, y1)).load()
        tops, bottoms = (
            subset[var].transpose("y", "x", "layer").values[iy[rows] - y0, ix[rows] - x0]
            for var in ("top", "bottom")
        )
        chosen, count = layer_at_z(tops, bottoms, layers, points["z"].to_numpy()[rows])
        out.loc[rows[count > 1], "regis_status"] = "ambiguous_layers"
        ok = np.flatnonzero(chosen >= 0)
        matched = rows[ok]
        top, bottom = tops[ok, chosen[ok]], bottoms[ok, chosen[ok]]
        out.loc[matched, "regis_status"] = "matched"
        out.loc[matched, "regis_layer"] = layers[chosen[ok]]
        out.loc[matched, "regis_top_nap_m"] = top
        out.loc[matched, "regis_bottom_nap_m"] = bottom
        if "screen_top" in points:
            screen_top = points["screen_top"].to_numpy()[matched]
            screen_bottom = points["screen_bottom"].to_numpy()[matched]
            length = screen_top - screen_bottom
            overlap = np.maximum(0, np.minimum(screen_top, top) - np.maximum(screen_bottom, bottom))
            out.loc[matched, "screen_overlap_fraction"] = np.divide(
                overlap, length, out=np.ones_like(length), where=length > 0,
            )
    return out


def add_regis(pairs: pd.DataFrame, ds: xr.Dataset) -> pd.DataFrame:
    chloride = pairs[[*CL_XY, SCREEN[2], SCREEN[1], SCREEN[0]]].copy()
    chloride.columns = ["x", "y", "z", "screen_top", "screen_bottom"]
    logs = pairs[[*LOG_XY, "log_z_nap_m"]].copy()
    logs.columns = ["x", "y", "z"]
    points = pd.concat([chloride, logs], ignore_index=True)
    units = classify_regis(ds, points)
    cl_units = units.iloc[:len(pairs)].add_prefix("chloride_")
    log_units = units.iloc[len(pairs):].drop(columns="screen_overlap_fraction")
    log_units = log_units.reset_index(drop=True).add_prefix("log_")
    result = pd.concat([pairs, cl_units, log_units], axis=1)
    result["same_regis_layer"] = result["chloride_regis_layer"].eq(result["log_regis_layer"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chloride", type=Path, default=CHLORIDE)
    parser.add_argument("--ecpt", type=Path, default=EXTERNAL / "all_ecpt_data_clean.csv")
    parser.add_argument("--las", type=Path, default=EXTERNAL / "all_las_data_clean.csv")
    parser.add_argument("--regis", type=Path, default=REGIS)
    parser.add_argument("--radius-m", type=float, default=250.0)
    parser.add_argument("--vertical-tolerance-m", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT / r"data\3-input\formationfactor_from_xyzdata")
    args = parser.parse_args()
    for value in (args.radius_m, args.vertical_tolerance_m):
        if not np.isfinite(value) or value < 0:
            parser.error("Matching thresholds must be finite and nonnegative.")

    chloride = read_chloride(args.chloride)
    matches = []
    stats = {}
    for source, path in (("ECPT", args.ecpt), ("LAS", args.las)):
        matched, stats[source] = match_logs(chloride, path, source, args.radius_m, args.vertical_tolerance_m)
        matches.append(matched)
        print(f"{source}: {len(matched):,} matched chloride rows; {stats[source]}", flush=True)
    paired = chloride.merge(pd.concat(matches, ignore_index=True), on=SAMPLE, how="inner")
    paired = paired.sort_values([SAMPLE, "log_source"]).reset_index(drop=True)
    paired["resistivity_ohm_m"] = 1.0 / paired["conductivity_s_m"].astype(float)
    paired["vertical_match"] = np.where(paired["screen_gap_m"].eq(0), "inside_screen", "within_tolerance")
    with xr.open_dataset(args.regis) as ds:
        result = add_regis(paired, ds)
        model = ds.attrs.get("project")
    summary = {
        "inputs": {name: str(getattr(args, name)) for name in ("chloride", "ecpt", "las", "regis")},
        "regis_model": model,
        "horizontal_crs": "EPSG:28992",
        "vertical_datum": "NAP",
        "radius_m": args.radius_m,
        "vertical_tolerance_m": args.vertical_tolerance_m,
        "chloride_input_rows": len(chloride),
        "chloride_matched_rows": int(result[SAMPLE].nunique()),
        "matched_bbox_rd_m": [float(f(result[c])) for f in (min, max) for c in CL_XY] if len(result) else None,
        "matched_screen_range_nap_m": [float(result[SCREEN[0]].min()), float(result[SCREEN[1]].max())]
        if len(result) else None,
        "log_input_counts": stats,
        "pair_counts": result.groupby(["log_source", "vertical_match"]).size().unstack(fill_value=0).to_dict("index"),
        "regis_chloride_status": result["chloride_regis_status"].value_counts().to_dict(),
        "regis_log_status": result["log_regis_status"].value_counts().to_dict(),
        "pairs_same_regis_layer": int(result["same_regis_layer"].sum()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "chloride_geophysics_regis.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
