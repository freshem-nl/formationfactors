#!/usr/bin/env python
"""
Simplified SkyTEM section plotter.

Features
--------
- Works with the Aarhus Workbench export that contains LINE_NO, RHO_* and DEP_TOP_* columns.
- Draws both the raw layered model (filled between layer top/bottom) and an interpolated section.
- Optional salinity panel derived from pore-water conductivity via a configurable formation factor.
- Configurable colour scale (map, min/max range, linear or log scaling) and quantity (conductivity or resistivity).
- Supports limiting the plotted depth and exporting one image per line straight from the CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter
import re


def natural_sort_key(text: str) -> list:
    """Sort helper that keeps numeric suffixes in order (e.g. RHO_1, RHO_10)."""
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"([0-9]+)", text)]


def bulk_to_pore_conductivity(ec_bulk_s_per_m: np.ndarray, formation_factor: float = 3.2) -> np.ndarray:
    """Convert bulk conductivity (S/m) to pore-water conductivity (S/m) using sigma_pore = sigma_bulk * F."""
    ec_bulk = np.asarray(ec_bulk_s_per_m, dtype=float)
    return np.where(ec_bulk > 0, ec_bulk * formation_factor, np.nan)


def classify_salinity_from_conductivity(ec_pore_s_per_m: np.ndarray) -> np.ndarray:
    """Return 0=fresh, 1=brackish, 2=saline based on pore-water conductivity thresholds."""
    ec = np.asarray(ec_pore_s_per_m, dtype=float)
    classes = np.full(ec.shape, np.nan)
    valid = ~np.isnan(ec) & (ec > 0)
    classes[np.logical_and(valid, ec <= 0.18)] = 0
    classes[np.logical_and(valid, (ec > 0.18) & (ec <= 1.8))] = 1
    classes[np.logical_and(valid, ec > 1.8)] = 2
    return classes


def sanitize_filename_component(text: Optional[str]) -> str:
    """Make text safe for filenames on Windows."""
    if text is None:
        return ""
    invalid = set('<>:"/\\|?*')
    cleaned = "".join(ch if ch not in invalid and ord(ch) >= 32 else "_" for ch in str(text))
    cleaned = " ".join(cleaned.split())
    return cleaned.strip(".")


def read_skytem_xyz(file_path: Path) -> pd.DataFrame:
    """Parse the SkyTEM inversion export, handling AGS (/ LINE_NO) and #HEADERS styles."""
    file_path = Path(file_path)
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    header_line = None
    data_lines: list[str] = []

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("/ LINE_NO"):
            header_line = line[1:].strip()
            for tail in lines[idx + 1 :]:
                stripped = tail.strip()
                if stripped and not stripped.startswith("/") and not stripped.startswith("#"):
                    data_lines.append(stripped)
            break

        if line.upper().startswith("#HEADERS"):
            header_line = line.replace("#HEADERS", "").strip()
            continue

        if header_line and line.upper().startswith("#DATA"):
            data_lines.append(line.replace("#DATA", "").strip())
            continue

        if header_line and not line.startswith("/") and not line.startswith("#"):
            data_lines.append(line)

    if header_line is None:
        for idx, raw_line in enumerate(lines):
            stripped = raw_line.strip().lstrip("/").lstrip("#").strip()
            if "LINE_NO" in stripped and ("RHO_" in stripped or "SIGMA_" in stripped):
                header_line = stripped
                for tail in lines[idx + 1 :]:
                    entry = tail.strip()
                    if entry and not entry.startswith("/") and not entry.startswith("#"):
                        data_lines.append(entry)
                break

    if header_line is None:
        raise ValueError("Unable to find column headers containing LINE_NO and RHO_/SIGMA_ information.")

    columns = [col.strip() for col in header_line.split() if col.strip()]
    rows: list[list[str]] = []
    for entry in data_lines:
        values = entry.split()
        if len(values) == len(columns):
            rows.append(values)

    if not rows:
        raise ValueError(f"No data rows found in {file_path}")

    df = pd.DataFrame(rows, columns=columns)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([-9999, 9999], np.nan)

    if "UTMX" in df.columns and "X" not in df.columns:
        df["X"] = df["UTMX"]
    if "UTMY" in df.columns and "Y" not in df.columns:
        df["Y"] = df["UTMY"]

    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError("Input file must contain X/Y or UTMX/UTMY coordinates.")

    return df


def compute_line_length(line_df: pd.DataFrame) -> np.ndarray:
    """Return cumulative distance along the line based on X/Y coordinates."""
    coords = line_df[["X", "Y"]].to_numpy(dtype=float)
    dx = np.diff(coords[:, 0], prepend=coords[0, 0])
    dy = np.diff(coords[:, 1], prepend=coords[0, 1])
    step = np.sqrt(dx**2 + dy**2)
    length = np.cumsum(step)
    length -= length.min()
    return length


def compute_layer_bounds(depth_top_cols: np.ndarray, doi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Treat DEP_TOP_* columns as depths below the surface (m). Ensure each layer top increases with depth
    and derive layer bottoms from the next top (using DOI for the final layer).
    """
    tops = np.asarray(depth_top_cols, dtype=float).copy()
    n_soundings, n_layers = tops.shape

    if n_layers == 0:
        return tops, tops

    tops[:, 0] = np.where(np.isfinite(tops[:, 0]), tops[:, 0], 0.0)

    diffs = np.diff(tops, axis=1)
    typical_thickness = np.nanmedian(np.where(diffs > 0, diffs, np.nan), axis=1)
    typical_thickness = np.where(np.isfinite(typical_thickness) & (typical_thickness > 0), typical_thickness, 10.0)

    for j in range(1, n_layers):
        prev = tops[:, j - 1]
        current = tops[:, j]
        missing = ~np.isfinite(current)
        current[missing] = prev[missing] + typical_thickness[missing]
        not_deeper = current <= prev
        current[not_deeper] = prev[not_deeper] + np.maximum(typical_thickness[not_deeper], 1.0)
        tops[:, j] = current

    bottoms = np.empty_like(tops)
    if n_layers > 1:
        bottoms[:, :-1] = tops[:, 1:]

    if n_layers > 1:
        prev_thickness = tops[:, -1] - tops[:, -2]
    else:
        prev_thickness = typical_thickness
    if n_layers > 2:
        ratios = np.full_like(diffs[:, 1:], np.nan)
        prev_pos = diffs[:, :-1] > 0
        curr_pos = diffs[:, 1:] > 0
        mask = prev_pos & curr_pos
        ratios[mask] = diffs[:, 1:][mask] / diffs[:, :-1][mask]
        z_expansion_factor = np.nanmedian(ratios[np.isfinite(ratios)])
    else:
        z_expansion_factor = np.nan
    if not np.isfinite(z_expansion_factor) or z_expansion_factor <= 0:
        z_expansion_factor = 1.2
    last_thickness = prev_thickness * z_expansion_factor
    last_thickness = np.where(np.isfinite(last_thickness) & (last_thickness > 0), last_thickness, typical_thickness)
    bottoms[:, -1] = tops[:, -1] + last_thickness

    if np.any(np.isfinite(doi)):
        doi_depth = doi
        valid = np.isfinite(doi_depth)
        bottoms[valid, -1] = np.minimum(bottoms[valid, -1], doi_depth[valid])

    too_shallow = bottoms <= tops
    bottoms[too_shallow] = tops[too_shallow] + 0.5

    return tops, bottoms


def _compute_profile_length_from_xyz_df(xyz_df: pd.DataFrame) -> np.ndarray:
    """
    Compute cumulative distance along the line for a raw SkyTEM XYZ DataFrame.

    Prefers TxX/TxY if present, otherwise falls back to generic X/Y or index.
    """
    if {"TxX", "TxY"}.issubset(xyz_df.columns):
        coords = xyz_df[["TxX", "TxY"]].rename(columns={"TxX": "X", "TxY": "Y"})
    elif {"X", "Y"}.issubset(xyz_df.columns):
        coords = xyz_df[["X", "Y"]]
    elif {"UTMX", "UTMY"}.issubset(xyz_df.columns):
        coords = xyz_df[["UTMX", "UTMY"]].rename(columns={"UTMX": "X", "UTMY": "Y"})
    else:
        # Fallback: simple index-based coordinate
        idx = np.arange(len(xyz_df), dtype=float)
        return idx - idx.min()

    return compute_line_length(coords.reset_index(drop=True))


def plot_gate_usage_from_xyz(
    xyz_path: Path,
    line_number: Optional[int] = None,
    max_stations: Optional[int] = None,
) -> plt.Figure:
    """
    Plot which LM/HM gates are used in the inversion based on LMZ_Flag*/HMZ_Flag* columns.

    - **Black** points: gates with Flag == 1 and non-zero data (used).
    - **Grey** points: gates that exist but are not used (Flag != 1, zero, or NaN).

    The x-axis is distance along the line (from TxX/TxY where available),
    the y-axis is gate number.
    """
    xyz_path = Path(xyz_path)
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    # SkyTEM XYZ exports are whitespace-separated; use a regex separator.
    # NOTE: use r"\s+" (single backslash) here – using "\\s+" would be interpreted
    # as a literal backslash + "s" and break column parsing.
    df = pd.read_csv(xyz_path, header=0, sep=r"\s+", engine="python")
    # Remove possible leading "/" from column names (Workbench-style headers)
    df.columns = df.columns.str.replace("/", "", regex=False)

    if line_number is not None:
        if "Line" in df.columns:
            mask = df["Line"] == int(line_number)
            if mask.any():
                df = df[mask].copy()
            else:
                # If requested line ID does not exist in this XYZ, fall back to all
                # soundings instead of raising – this keeps the caller logic simple.
                print(
                    f"WARNING: Line {line_number} not found in {xyz_path.name}; "
                    f"plotting all available soundings instead."
                )
        else:
            # No explicit Line column – fall back to using all rows as one "line"
            # This keeps the helper robust for XYZ formats without a Line field.
            print(
                f"WARNING: XYZ file {xyz_path.name} has no 'Line' column; "
                f"plotting all soundings as a single line."
            )

    df = df.reset_index(drop=True)
    if max_stations is not None and max_stations > 0:
        df = df.iloc[: int(max_stations), :].reset_index(drop=True)

    n_stations = len(df)
    if n_stations == 0:
        raise ValueError("No soundings found after filtering.")

    # Compute along-line distance and sort by it
    lengths = _compute_profile_length_from_xyz_df(df)
    order = np.argsort(lengths)
    lengths = lengths[order]
    df = df.iloc[order].reset_index(drop=True)

    def _build_gate_usage_matrices(
        df_in: pd.DataFrame,
        val_prefix: str,
        flag_prefix: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (gate_numbers, used_mask, has_data_mask).

        used_mask  : Flag == 1 and finite, non-zero value.
        has_data_mask : finite value present in the column (before flag masking).
        """
        cols = list(df_in.columns)
        val_cols = [c for c in cols if c.startswith(val_prefix)]
        if not val_cols:
            return np.array([], dtype=int), np.zeros((n_stations, 0), dtype=bool), np.zeros(
                (n_stations, 0), dtype=bool
            )

        # Parse gate numbers from column names, e.g. LMZ_D1 → 1, HMZ_D10 → 10
        gate_numbers: list[int] = []
        col_gate_map: dict[int, str] = {}
        pattern = re.compile(rf"^{re.escape(val_prefix)}(\d+)$")
        for name in val_cols:
            m = pattern.match(name)
            if not m:
                continue
            gno = int(m.group(1))
            gate_numbers.append(gno)
            col_gate_map[gno] = name

        if not gate_numbers:
            return np.array([], dtype=int), np.zeros((n_stations, 0), dtype=bool), np.zeros(
                (n_stations, 0), dtype=bool
            )

        gate_numbers_sorted = sorted(set(gate_numbers))
        n_gates = len(gate_numbers_sorted)
        used = np.zeros((n_stations, n_gates), dtype=bool)
        has_data = np.zeros_like(used)

        gate_idx_map = {g: i for i, g in enumerate(gate_numbers_sorted)}

        for gno in gate_numbers_sorted:
            val_col = col_gate_map.get(gno)
            if val_col is None:
                continue
            j = gate_idx_map[gno]
            vals = pd.to_numeric(df_in[val_col], errors="coerce")
            has_data[:, j] = vals.notna().to_numpy()

            flag_col = f"{flag_prefix}{gno}"
            if flag_col in df_in.columns:
                flags = pd.to_numeric(df_in[flag_col], errors="coerce")
                flag_ok = flags == 1
            else:
                # If no flag column exists, treat all gates as unflagged (used if data present)
                flag_ok = pd.Series(True, index=df_in.index)

            used_gate = flag_ok & vals.notna() & (vals != 0.0)
            used[:, j] = used_gate.to_numpy()

        return np.asarray(gate_numbers_sorted, dtype=int), used, has_data

    lm_gate_numbers, lm_used, lm_has_data = _build_gate_usage_matrices(df, "LMZ_D", "LMZ_Flag")
    hm_gate_numbers, hm_used, hm_has_data = _build_gate_usage_matrices(df, "HMZ_D", "HMZ_Flag")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(18, 6), gridspec_kw={"height_ratios": [1, 1]})
    ax_lm, ax_hm = axes

    def _plot_panel(
        ax,
        gate_numbers: np.ndarray,
        used_mask: np.ndarray,
        has_data_mask: np.ndarray,
        title: str,
    ) -> None:
        if gate_numbers.size == 0:
            ax.text(
                0.5,
                0.5,
                "No gate columns found",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_yticks([])
            ax.set_ylabel("Gate #")
            ax.set_title(title, fontsize=13, pad=6)
            ax.grid(True, color="0.9", linestyle="--", linewidth=0.5)
            return

        for j, gno in enumerate(gate_numbers):
            # Used gates (black)
            used_idx = used_mask[:, j]
            if np.any(used_idx):
                x_used = lengths[used_idx]
                y_used = np.full_like(x_used, gno, dtype=float)
                ax.plot(x_used, y_used, "k.", markersize=2.0)

            # Gates with data but not used (grey)
            flagged_idx = has_data_mask[:, j] & ~used_mask[:, j]
            if np.any(flagged_idx):
                x_flag = lengths[flagged_idx]
                y_flag = np.full_like(x_flag, gno, dtype=float)
                ax.plot(x_flag, y_flag, ".", color="0.75", markersize=2.0)

        ax.set_yticks(gate_numbers)
        ax.set_ylim(gate_numbers.max() + 0.5, gate_numbers.min() - 0.5)
        ax.set_ylabel("Gate #", fontsize=11)
        ax.set_title(title, fontsize=13, pad=6)
        ax.grid(True, color="0.9", linestyle="--", linewidth=0.5)

    _plot_panel(ax_lm, lm_gate_numbers, lm_used, lm_has_data, "LM gate usage (black = used, grey = flagged)")
    _plot_panel(ax_hm, hm_gate_numbers, hm_used, hm_has_data, "HM gate usage (black = used, grey = flagged)")

    ax_hm.set_xlabel("Distance along line (m)", fontsize=12)
    if lengths.size > 0:
        ax_hm.set_xlim(lengths.min(), lengths.max())

    fig.suptitle("SkyTEM gate usage by line (LM/HM)", fontsize=16, y=0.96)
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])

    return fig


def detect_gap_areas(lengths: np.ndarray, max_gap: float = 30.0) -> np.ndarray:
    """
    Detect gap areas where spacing between consecutive points exceeds max_gap.
    Returns a boolean mask: True for valid areas, False for gap areas.
    Also returns gap bounds for masking interpolated grid.
    """
    lengths = np.asarray(lengths, dtype=float)
    n = lengths.size
    if n <= 1:
        return np.ones(n, dtype=bool), []
    
    # Calculate spacing between consecutive points
    spacing = np.diff(lengths)
    
    # Identify gaps > max_gap
    gap_mask = spacing > max_gap
    valid_mask = np.ones(n, dtype=bool)
    
    # Mark points adjacent to gaps as invalid
    # If gap between i and i+1, mark both i and i+1 as in gap area
    gap_indices = np.where(gap_mask)[0]
    for idx in gap_indices:
        if idx < n:
            valid_mask[idx] = False
        if idx + 1 < n:
            valid_mask[idx + 1] = False
    
    # Collect gap bounds for interpolated grid masking
    gap_bounds = []
    for idx in gap_indices:
        if idx < n - 1:
            gap_start = lengths[idx]
            gap_end = lengths[idx + 1]
            gap_bounds.append((gap_start, gap_end))
    
    return valid_mask, gap_bounds


def compute_station_bounds(lengths: np.ndarray) -> np.ndarray:
    """Return left/right bounds for each sounding to draw vertical blocks."""
    lengths = np.asarray(lengths, dtype=float)
    n = lengths.size
    if n == 1:
        half = 25.0
        return np.array([[lengths[0] - half, lengths[0] + half]])

    midpoints = (lengths[1:] + lengths[:-1]) / 2.0
    bounds = np.zeros((n, 2))
    spacing_first = lengths[1] - lengths[0]
    spacing_last = lengths[-1] - lengths[-2]

    bounds[0, 0] = lengths[0] - spacing_first / 2.0
    bounds[0, 1] = midpoints[0]
    bounds[-1, 0] = midpoints[-1]
    bounds[-1, 1] = lengths[-1] + spacing_last / 2.0

    if n > 2:
        bounds[1:-1, 0] = midpoints[:-1]
        bounds[1:-1, 1] = midpoints[1:]

    return bounds


def prepare_simpeg_arrays(df: pd.DataFrame) -> dict:
    """
    Prepare arrays for plotting from a SimPEG stacked CSV.

    Expects columns: X, Y, Ztop, Zbot, sigma, misfit, altitude, DEM.
    One row per layer.
    """
    required = ["X", "Y", "Ztop", "Zbot", "sigma"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"SimPEG CSV must contain column '{col}'.")

    # Ensure numeric
    for col in df.columns:
        if col not in ["X", "Y"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Group by station (X, Y)
    groups = df.groupby(["X", "Y"], sort=False)
    station_records: list[dict] = []
    max_layers = 0

    for (x, y), g in groups:
        g = g.copy()
        dem = g["DEM"].iloc[0] if "DEM" in g.columns else 0.0
        alt = g["altitude"].iloc[0] if "altitude" in g.columns else 0.0
        misfit = g["misfit"].iloc[0] if "misfit" in g.columns else np.nan

        # Convert elevations to depths below surface (DEM)
        depth_top = dem - g["Ztop"].to_numpy(dtype=float)
        depth_bot = dem - g["Zbot"].to_numpy(dtype=float)

        # Sort by depth
        order = np.argsort(depth_top)
        depth_top = depth_top[order]
        depth_bot = depth_bot[order]
        sigma = g["sigma"].to_numpy(dtype=float)[order]

        rho = np.where(sigma > 0, 1.0 / sigma, np.nan)

        max_layers = max(max_layers, len(depth_top))

        station_records.append(
            {
                "X": float(x),
                "Y": float(y),
                "DEM": float(dem),
                "altitude": float(alt),
                "misfit": float(misfit),
                "depth_top": depth_top,
                "depth_bottom": depth_bot,
                "rho": rho,
            }
        )

    n_stations = len(station_records)
    if n_stations == 0:
        raise ValueError("No stations found in SimPEG CSV.")

    depth_top_arr = np.full((n_stations, max_layers), np.nan, dtype=float)
    depth_bottom_arr = np.full_like(depth_top_arr, np.nan)
    rho_arr = np.full_like(depth_top_arr, np.nan)

    xs = np.empty(n_stations, dtype=float)
    ys = np.empty(n_stations, dtype=float)
    dems = np.empty(n_stations, dtype=float)
    alts = np.empty(n_stations, dtype=float)
    misfits = np.empty(n_stations, dtype=float)

    for i, rec in enumerate(station_records):
        nl = len(rec["depth_top"])
        depth_top_arr[i, :nl] = rec["depth_top"]
        depth_bottom_arr[i, :nl] = rec["depth_bottom"]
        rho_arr[i, :nl] = rec["rho"]

        xs[i] = rec["X"]
        ys[i] = rec["Y"]
        dems[i] = rec["DEM"]
        alts[i] = rec["altitude"]
        misfits[i] = rec["misfit"]

    # Compute along-line lengths and sort by distance
    df_stn = pd.DataFrame({"X": xs, "Y": ys})
    lengths = compute_line_length(df_stn)
    order = np.argsort(lengths)
    lengths = lengths[order]
    depth_top_arr = depth_top_arr[order]
    depth_bottom_arr = depth_bottom_arr[order]
    rho_arr = rho_arr[order]
    dems = dems[order]
    alts = alts[order]
    misfits = misfits[order]

    return {
        "lengths": lengths,
        "rho": rho_arr,
        "depth_top": depth_top_arr,
        "depth_bottom": depth_bottom_arr,
        "surface": dems,
        "altitude": alts,
        "misfit": misfits,
    }


def plot_simpeg_section(
    df_sim: pd.DataFrame,
    *,
    cmap_name: str = "RdYlBu_r",
    scale: str = "log",
    unit: str = "conductivity",
    value_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    max_depth: Optional[float] = None,
    grid_length: int = 400,
    grid_depth: int = 400,
    plot_salinity: bool = False,
    formation_factor: float = 3.2,
    title: Optional[str] = None,
    smooth_sigma: float = 0.0,
    show_interpolated: bool = True,
):
    """Plot a section from a SimPEG stacked CSV (no DOI, no LINE_NO)."""
    arrays = prepare_simpeg_arrays(df_sim)
    lengths = arrays["lengths"]
    rho = arrays["rho"]
    depth_top = arrays["depth_top"]
    depth_bottom = arrays["depth_bottom"]
    surface = np.asarray(arrays["surface"], dtype=float)
    altitude = np.asarray(arrays["altitude"], dtype=float)
    residuals = np.asarray(arrays["misfit"], dtype=float)

    # Normalize lengths
    lengths = lengths - lengths.min()

    # Conductivity / resistivity
    ec = np.where(rho > 0, 1.0 / rho, np.nan)
    if unit == "conductivity":
        values_matrix = ec
        bar_label = "Conductivity (S/m)"
    elif unit == "resistivity":
        values_matrix = rho
        bar_label = "Resistivity (ohm m)"
    else:
        raise ValueError("unit must be 'conductivity' or 'resistivity'.")

    if np.any(~np.isfinite(surface)):
        fallback = float(np.nanmedian(surface[np.isfinite(surface)])) if np.any(np.isfinite(surface)) else 0.0
        surface = np.where(np.isfinite(surface), surface, fallback)

    finite_depths = depth_bottom[np.isfinite(depth_bottom)]
    if finite_depths.size == 0:
        raise ValueError("No finite depth information available.")

    if max_depth is None:
        plot_depth = float(np.nanmax(finite_depths))
    else:
        plot_depth = float(max_depth)
    plot_depth = max(plot_depth, 5.0)

    depth_bottom_clipped = np.minimum(depth_bottom, plot_depth)
    layer_top_elev = surface[:, None] - depth_top
    layer_bottom_elev = surface[:, None] - depth_bottom_clipped
    y_max = float(np.nanmax(surface)) + 5.0
    y_min = float(np.nanmin(surface - plot_depth)) - 5.0

    vmin, vmax = derive_value_range(values_matrix, scale, value_range)
    cmap = plt.get_cmap(cmap_name)
    if scale == "log":
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Layout: terrain + altitude + residual + raw (+ optional interpolated and salinity)
    if show_interpolated:
        subplot_count = 5 + (1 if plot_salinity else 0)
        height_ratios = [0.8, 0.8, 0.6, 2.0, 2.0] + ([1.2] if plot_salinity else [])
    else:
        # No interpolated panel → give extra space to the raw scatter panel
        subplot_count = 4 + (1 if plot_salinity else 0)
        height_ratios = [0.8, 0.8, 0.6, 3.0] + ([1.2] if plot_salinity else [])

    fig = plt.figure(figsize=(20, 16 + (3 if plot_salinity else 0)))
    gs = plt.GridSpec(subplot_count, 1, height_ratios=height_ratios, hspace=0.15)

    ax_terrain = fig.add_subplot(gs[0])
    ax_alt = fig.add_subplot(gs[1])
    ax_res = fig.add_subplot(gs[2])
    ax_scatter = fig.add_subplot(gs[3])  # Raw data points
    ax_interp = fig.add_subplot(gs[4]) if show_interpolated else None  # Interpolated/smoothed

    # If we are not showing the interpolated section, the salinity axis (if requested)
    # occupies the next row; otherwise it comes after the interpolated panel.
    sal_index = 4 if (plot_salinity and not show_interpolated) else 5
    ax_sal = fig.add_subplot(gs[sal_index]) if (plot_salinity and subplot_count > 4) else None

    axes = [ax_terrain, ax_alt, ax_res, ax_scatter]
    if ax_interp is not None:
        axes.append(ax_interp)
    if ax_sal is not None:
        axes.append(ax_sal)

    # Share x-axis (all panels share the terrain axis limits)
    for ax in axes[1:]:
        ax.sharex(ax_terrain)

    # Terrain from DEM
    profile_length = lengths
    terrain_series = pd.Series(surface)
    ax_terrain.plot(profile_length, terrain_series, "k-", linewidth=1.2)
    terrain_min = float(np.nanmin(terrain_series))
    terrain_max = float(np.nanmax(terrain_series))
    terrain_range = terrain_max - terrain_min
    if terrain_range > 0:
        ax_terrain.set_ylim(terrain_min - terrain_range * 0.05, terrain_max + terrain_range * 0.05)
    else:
        ax_terrain.set_ylim(terrain_min - 5, terrain_max + 5)
    ax_terrain.set_ylabel("Terrain elevation (m)", fontsize=12)
    ax_terrain.grid(True, color="0.85", linestyle="--", linewidth=0.6)
    ax_terrain.set_title("Terrain elevation", fontsize=13, pad=6)
    ax_terrain.tick_params(labelbottom=False)

    # Altitude
    altitude_series = pd.Series(altitude)
    altitude_series = altitude_series.interpolate(limit_direction="both")
    ax_alt.plot(profile_length, altitude_series, "k-", linewidth=1.2)
    ax_alt.set_ylim(0, float(np.nanmax(altitude_series)) * 1.05)
    ax_alt.set_ylabel("Altitude (m agl)", fontsize=12)
    ax_alt.grid(True, color="0.85", linestyle="--", linewidth=0.6)
    ax_alt.set_title("Flight altitude", fontsize=13, pad=6)
    ax_alt.tick_params(labelbottom=False)

    # Residual (misfit ratio)
    ax_res.scatter(profile_length, residuals, c=residuals, cmap="RdYlGn_r", s=18, alpha=0.8)
    ax_res.axhline(1.0, color="green", linestyle="-", linewidth=1.0, alpha=0.7)
    ax_res.set_ylabel("Residual", fontsize=12)
    ax_res.set_title("Misfit (RMS / target)", fontsize=13, pad=6)
    if np.any(np.isfinite(residuals)):
        res_max = np.nanpercentile(residuals, 95)
        ax_res.set_ylim(0.5, max(2.5, res_max * 1.1))
    ax_res.grid(True, color="0.85", linestyle="--", linewidth=0.6)
    ax_res.tick_params(labelbottom=False)

    # Optional interpolated grid for smooth background
    mesh = None
    if show_interpolated:
        # Create fine depth grid
        depth_axis = np.linspace(0, plot_depth, grid_depth)
        length_axis = np.linspace(lengths.min(), lengths.max(), grid_length)

        # Create grid filled with layer values
        grid_values = np.full((grid_depth, grid_length), np.nan, dtype=float)
        surface_interp = np.interp(length_axis, lengths, surface)

        # For each grid column, interpolate layer boundaries and values from nearby stations
        # then fill each layer from top to bottom
        for col_idx, length_val in enumerate(length_axis):
            # Find nearest stations for horizontal interpolation
            distances = np.abs(lengths - length_val)
            # Use linear interpolation between nearest stations
            if len(lengths) == 1:
                # Single station case
                i = 0
                station_depths_top = depth_top[i, :]
                station_depths_bot = depth_bottom_clipped[i, :]
                station_values = values_matrix[i, :]
            else:
                # Find two nearest stations
                sorted_indices = np.argsort(distances)
                i1, i2 = sorted_indices[0], sorted_indices[1]

                # If exactly on a station, use that station
                if distances[i1] < 1e-6:
                    i = i1
                    station_depths_top = depth_top[i, :]
                    station_depths_bot = depth_bottom_clipped[i, :]
                    station_values = values_matrix[i, :]
                else:
                    # Interpolate between two nearest stations
                    d1, d2 = distances[i1], distances[i2]
                    total_dist = d1 + d2
                    if total_dist > 0:
                        w1 = d2 / total_dist  # Weight for station 1 (closer = higher weight)
                        w2 = d1 / total_dist  # Weight for station 2
                    else:
                        w1, w2 = 1.0, 0.0

                    # Interpolate layer boundaries and values
                    station_depths_top = w1 * depth_top[i1, :] + w2 * depth_top[i2, :]
                    station_depths_bot = w1 * depth_bottom_clipped[i1, :] + w2 * depth_bottom_clipped[i2, :]
                    station_values = w1 * values_matrix[i1, :] + w2 * values_matrix[i2, :]

            # Fill each layer's depth range with its interpolated value
            for layer_idx in range(len(station_depths_top)):
                if not (
                    np.isfinite(station_depths_top[layer_idx])
                    and np.isfinite(station_depths_bot[layer_idx])
                    and np.isfinite(station_values[layer_idx])
                ):
                    continue

                top_d = station_depths_top[layer_idx]
                bot_d = station_depths_bot[layer_idx]
                val = station_values[layer_idx]

                # Find depth grid indices for this layer
                depth_mask = (depth_axis >= top_d) & (depth_axis <= bot_d)
                if np.any(depth_mask):
                    grid_values[depth_mask, col_idx] = val

        # Convert depth grid to elevation grid
        length_grid, depth_grid = np.meshgrid(length_axis, depth_axis)
        elevation_grid = surface_interp - depth_grid

        # Mask areas beyond column depth limits
        column_depth_limit = np.nanmax(depth_bottom_clipped, axis=1)
        column_depth_limit = np.where(np.isfinite(column_depth_limit), column_depth_limit, 0.0)
        depth_limit_grid = np.interp(length_axis, lengths, column_depth_limit)
        mask_invalid = depth_grid > depth_limit_grid
        grid_values = np.where(mask_invalid, np.nan, grid_values)

    # Scatter plot of actual layer data points (raw data)
    layer_mid_depth = 0.5 * (depth_top + depth_bottom_clipped)
    layer_mid_elev = surface[:, None] - layer_mid_depth
    mask_valid = np.isfinite(values_matrix) & np.isfinite(layer_mid_elev) & (layer_mid_depth <= plot_depth)
    lengths_scatter = np.repeat(lengths, values_matrix.shape[1])[mask_valid.ravel()]
    elev_scatter = layer_mid_elev.ravel()[mask_valid.ravel()]
    values_scatter = values_matrix.ravel()[mask_valid.ravel()]
    
    scatter = ax_scatter.scatter(
        lengths_scatter,
        elev_scatter,
        c=values_scatter,
        cmap=cmap,
        norm=norm,
        s=15,
        alpha=0.8,
        edgecolors='k',
        linewidths=0.3,
    )
    
    if show_interpolated:
        # Plot interpolated background
        mesh = ax_interp.pcolormesh(
            length_grid,
            elevation_grid,
            np.ma.masked_invalid(grid_values),
            shading="auto",
            cmap=cmap,
            norm=norm,
        )

    x_min, x_max = lengths.min(), lengths.max()
    axes_xlim = [ax_terrain, ax_alt, ax_res, ax_scatter]
    if ax_interp is not None:
        axes_xlim.append(ax_interp)
    if ax_sal is not None:
        axes_xlim.append(ax_sal)
    for ax in axes_xlim:
        ax.set_xlim(x_min, x_max)

    axes_to_format = [ax_scatter]
    if ax_interp is not None:
        axes_to_format.append(ax_interp)
    if plot_salinity and ax_sal is not None:
        axes_to_format.append(ax_sal)
    for ax in axes_to_format:
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Elevation (m)", fontsize=12)
        ax.grid(True, color="0.85", linestyle="--", linewidth=0.6)
        ax.plot(lengths, surface, "k-", linewidth=1.0)
    
    ax_scatter.set_title("Raw layer data points", fontsize=13, pad=6)
    ax_scatter.tick_params(labelbottom=False)
    if show_interpolated and ax_interp is not None:
        ax_interp.set_title("Interpolated section (terrain referenced)", fontsize=13, pad=6)

    if plot_salinity and ax_sal is not None:
        ec_pore = bulk_to_pore_conductivity(ec, formation_factor)
        salinity_classes = classify_salinity_from_conductivity(ec_pore)
        sal_cmap = ListedColormap(["#1f78b4", "#ffb703", "#d73027"])
        sal_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], sal_cmap.N)
        # Compute station bounds for salinity panel
        sal_station_bounds = compute_station_bounds(lengths)
        sal_valid_mask = np.ones(len(lengths), dtype=bool)
        sal_collection = draw_layer_blocks(
            ax_sal,
            sal_station_bounds,
            layer_top_elev,
            layer_bottom_elev,
            salinity_classes,
            sal_cmap,
            sal_norm,
            y_min,
            y_max,
            valid_mask=sal_valid_mask,
        )
        ax_sal.set_title("Salinity classification (fresh/brackish/saline)", fontsize=14, pad=8)
        cbar_sal = fig.colorbar(
            sal_collection,
            ax=ax_sal,
            orientation="vertical",
            pad=0.02,
            fraction=0.04,
        )
        cbar_sal.set_ticks([0, 1, 2])
        cbar_sal.set_ticklabels(["Fresh", "Brackish", "Saline"])
        cbar_sal.ax.tick_params(labelsize=10)

    bottom_axis = ax_sal if (plot_salinity and ax_sal is not None) else (ax_interp if ax_interp is not None else ax_scatter)
    bottom_axis.set_xlabel("Distance along line (m)", fontsize=12)

    overall_title = title or "SimPEG SkyTEM line"
    fig.suptitle(overall_title, fontsize=16, y=0.98)

    plt.subplots_adjust(hspace=0.15, top=0.95, right=0.87, left=0.08)

    sm = mesh if (show_interpolated and mesh is not None) else scatter
    cbar_axes = [ax_scatter]
    if ax_interp is not None:
        cbar_axes.append(ax_interp)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=cbar_axes,
        pad=0.02,
        fraction=0.04,
    )
    cbar.set_label(bar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    return fig, axes, sm


def draw_layer_blocks(
    ax,
    station_bounds: np.ndarray,
    layer_tops: np.ndarray,
    layer_bottoms: np.ndarray,
    values: np.ndarray,
    cmap,
    norm,
    y_min: float,
    y_max: float,
    valid_mask: Optional[np.ndarray] = None,
    doi_elevations: Optional[np.ndarray] = None,
    alpha_below_doi: float = 0.3,
):
    """Fill each layer cell with a rectangle coloured by its value.
    
    Args:
        doi_elevations: Array of DOI elevations (one per sounding). If provided,
            layers below DOI will have reduced alpha.
        alpha_below_doi: Alpha value for layers below DOI (default 0.3).
    """
    patches: list[Rectangle] = []
    patch_values: list[float] = []
    patch_alphas: list[float] = []
    n_soundings, n_layers = values.shape
    
    if valid_mask is None:
        valid_mask = np.ones(n_soundings, dtype=bool)

    for i in range(n_soundings):
        # Skip if in gap area
        if not valid_mask[i]:
            continue
            
        left, right = station_bounds[i]
        width = right - left
        if np.isnan(width) or width <= 0:
            continue
        
        # Get DOI elevation for this sounding (if available)
        doi_elev = doi_elevations[i] if doi_elevations is not None and i < len(doi_elevations) else None
        
        for j in range(n_layers):
            val = values[i, j]
            top = layer_tops[i, j]
            bottom = layer_bottoms[i, j]
            if not np.isfinite(val) or not np.isfinite(top) or not np.isfinite(bottom):
                continue
            top_clipped = np.clip(top, y_min, y_max)
            bottom_clipped = np.clip(bottom, y_min, y_max)
            if bottom_clipped >= top_clipped:
                continue
            
            # Determine alpha: reduce if below DOI
            alpha = 1.0
            if doi_elev is not None and np.isfinite(doi_elev):
                # If the layer is entirely below DOI, use reduced alpha
                # Use the midpoint of the layer to determine if it's below DOI
                layer_mid_elev = (top_clipped + bottom_clipped) / 2.0
                if layer_mid_elev < doi_elev:
                    alpha = alpha_below_doi
            
            rect = Rectangle((left, bottom_clipped), width, top_clipped - bottom_clipped)
            patches.append(rect)
            patch_values.append(val)
            patch_alphas.append(alpha)

    if not patches:
        return None

    # Create separate collections for above and below DOI to handle different alpha values
    if doi_elevations is not None and any(a < 1.0 for a in patch_alphas):
        # Split patches into above and below DOI
        patches_above = [p for p, a in zip(patches, patch_alphas) if a >= 1.0]
        patches_below = [p for p, a in zip(patches, patch_alphas) if a < 1.0]
        values_above = [v for v, a in zip(patch_values, patch_alphas) if a >= 1.0]
        values_below = [v for v, a in zip(patch_values, patch_alphas) if a < 1.0]
        
        collection = None
        if patches_above:
            coll_above = PatchCollection(patches_above, cmap=cmap, norm=norm, linewidths=0, alpha=1.0)
            coll_above.set_array(np.asarray(values_above))
            ax.add_collection(coll_above)
            collection = coll_above
        if patches_below:
            coll_below = PatchCollection(patches_below, cmap=cmap, norm=norm, linewidths=0, alpha=alpha_below_doi)
            coll_below.set_array(np.asarray(values_below))
            ax.add_collection(coll_below)
            if collection is None:
                collection = coll_below
    else:
        collection = PatchCollection(patches, cmap=cmap, norm=norm, linewidths=0, alpha=1.0)
        collection.set_array(np.asarray(patch_values))
        ax.add_collection(collection)
    
    return collection


def build_interpolated_grid(
    lengths: np.ndarray,
    layer_mid_depth: np.ndarray,
    values: np.ndarray,
    max_depth: float,
    grid_length: int,
    grid_depth: int,
    method: str = "rbf",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate values onto a regular (length, depth) mesh.
    
    Uses RBF interpolation by default for smoother results with scattered data.
    Falls back to griddata if RBF fails.
    """
    mask = np.isfinite(values) & np.isfinite(layer_mid_depth) & (layer_mid_depth <= max_depth)
    if not np.any(mask):
        raise ValueError("No valid cells available for interpolation.")

    repeat_lengths = np.repeat(lengths, layer_mid_depth.shape[1])
    points = np.column_stack((repeat_lengths[mask.ravel()], layer_mid_depth.ravel()[mask.ravel()]))
    samples = values.ravel()[mask.ravel()]

    length_axis = np.linspace(lengths.min(), lengths.max(), grid_length)
    depth_axis = np.linspace(0, max_depth, grid_depth)
    grid_L, grid_D = np.meshgrid(length_axis, depth_axis)
    grid_points = np.column_stack((grid_L.ravel(), grid_D.ravel()))

    if method == "rbf" and len(points) >= 3:
        # Use RBF interpolation for smoother results with scattered data
        try:
            # Normalize coordinates for better RBF performance
            length_range = lengths.max() - lengths.min()
            depth_range = max_depth
            if length_range > 0 and depth_range > 0:
                points_norm = points.copy()
                points_norm[:, 0] = (points[:, 0] - lengths.min()) / length_range
                points_norm[:, 1] = points[:, 1] / depth_range
                grid_points_norm = grid_points.copy()
                grid_points_norm[:, 0] = (grid_points[:, 0] - lengths.min()) / length_range
                grid_points_norm[:, 1] = grid_points[:, 1] / depth_range
                
                # Use thin-plate spline kernel (smooth, no parameters to tune)
                rbf = RBFInterpolator(points_norm, samples, kernel="thin_plate_spline", smoothing=0.0)
                grid_flat = rbf(grid_points_norm)
                grid_combined = grid_flat.reshape(grid_L.shape)
            else:
                raise ValueError("Invalid coordinate ranges")
        except Exception:
            # Fall back to griddata if RBF fails
            grid_primary = griddata(points, samples, (grid_L, grid_D), method="cubic")
            grid_nearest = griddata(points, samples, (grid_L, grid_D), method="nearest")
            grid_combined = np.where(np.isnan(grid_primary), grid_nearest, grid_primary)
    else:
        # Use griddata with cubic interpolation (smoother than linear)
        grid_primary = griddata(points, samples, (grid_L, grid_D), method="cubic")
        grid_nearest = griddata(points, samples, (grid_L, grid_D), method="nearest")
        grid_combined = np.where(np.isnan(grid_primary), grid_nearest, grid_primary)

    return length_axis, depth_axis, grid_combined


def derive_value_range(
    values: np.ndarray,
    scale: str,
    provided_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Tuple[float, float]:
    """Determine vmin/vmax either from user input or percentiles."""
    if provided_range and all(v is not None for v in provided_range):
        vmin, vmax = float(provided_range[0]), float(provided_range[1])
        if vmin >= vmax:
            raise ValueError("value-min must be smaller than value-max.")
        return vmin, vmax

    flat = values[np.isfinite(values)]
    if scale == "log":
        flat = flat[flat > 0]
    if flat.size == 0:
        raise ValueError("No finite values available to derive colour range.")

    lower = np.percentile(flat, 2)
    upper = np.percentile(flat, 98)
    if scale == "log" and lower <= 0:
        lower = flat.min()
    if upper <= lower:
        upper = lower * 1.1

    return float(lower), float(upper)


def prepare_line_arrays(df: pd.DataFrame, line_number: int) -> dict:
    """Extract arrays needed for plotting one line."""
    line_df = df[df["LINE_NO"] == line_number].copy()
    if line_df.empty:
        raise ValueError(f"Line {line_number} not found.")

    rho_cols = sorted([c for c in line_df.columns if c.upper().startswith("RHO_") and not c.upper().startswith("RHO_STD")], key=natural_sort_key)
    rho_std_cols = sorted([c for c in line_df.columns if c.upper().startswith("RHO_STD_")], key=natural_sort_key)
    sigma_cols = sorted([c for c in line_df.columns if c.upper().startswith("SIGMA_") and not c.upper().startswith("SIGMA_STD")], key=natural_sort_key)
    sigma_std_cols = sorted([c for c in line_df.columns if c.upper().startswith("SIGMA_STD_")], key=natural_sort_key)
    depth_cols = sorted([c for c in line_df.columns if c.upper().startswith("DEP_TOP_")], key=natural_sort_key)

    if not depth_cols:
        raise ValueError("No DEP_TOP_* columns found; cannot determine layer geometry.")

    if rho_cols:
        rho_values = line_df[rho_cols].to_numpy(dtype=float)
        if rho_std_cols and len(rho_std_cols) == len(rho_cols):
            rho_std_values = line_df[rho_std_cols].to_numpy(dtype=float)
        else:
            rho_std_values = None
    elif sigma_cols:
        sigma_values = line_df[sigma_cols].to_numpy(dtype=float)
        rho_values = np.where(sigma_values > 0, 1.0 / sigma_values, np.nan)
        if sigma_std_cols and len(sigma_std_cols) == len(sigma_cols):
            sigma_std_values = line_df[sigma_std_cols].to_numpy(dtype=float)
            rho_std_values = np.where(sigma_values > 0, sigma_std_values / (sigma_values**2), np.nan)
        else:
            rho_std_values = None
    else:
        raise ValueError("No RHO_* or SIGMA_* columns found.")
    
    has_std = rho_std_values is not None

    depth_bottom_cols = line_df[depth_cols].to_numpy(dtype=float)

    if rho_values.shape != depth_bottom_cols.shape:
        n_layers = min(rho_values.shape[1], depth_bottom_cols.shape[1])
        rho_values = rho_values[:, :n_layers]
        depth_bottom_cols = depth_bottom_cols[:, :n_layers]

    doi_col = None
    for candidate in ("DOI_CONSERVATIVE", "DOI_STANDARD"):
        if candidate in line_df.columns:
            doi_col = candidate
            break
    doi_values = line_df[doi_col].to_numpy(dtype=float) if doi_col else np.full(len(line_df), np.nan)

    line_df = line_df.reset_index(drop=True)
    lengths = compute_line_length(line_df)
    order = np.argsort(lengths)
    line_df = line_df.loc[order].reset_index(drop=True)
    lengths = lengths[order]
    rho_values = rho_values[order]
    depth_bottom_cols = depth_bottom_cols[order]
    doi_values = doi_values[order]

    surface_col = None
    for candidate in ("ELEVATION", "SURFACE", "TOPO", "GROUND_ELEVATION"):
        if candidate in line_df.columns:
            surface_col = candidate
            break
    if surface_col:
        surface = line_df[surface_col].astype(float).to_numpy()
    else:
        surface = np.zeros(len(line_df), dtype=float)
    surface = surface[order]

    # DEP_TOP_* are always treated as depths below surface (like plot_SkyTEM_new_format.py)
    # No conversion needed - use them directly as depths, convert to elevations only for plotting
    layer_tops, layer_bottoms = compute_layer_bounds(depth_bottom_cols, doi_values)

    return {
        "lengths": lengths,
        "rho": rho_values,
        "rho_std": rho_std_values if has_std else None,
        "depth_top": layer_tops,
        "depth_bottom": layer_bottoms,
        "doi": doi_values,
        "surface": surface,
    }


def plot_skytem_section(
    df: pd.DataFrame,
    line_number: int,
    *,
    cmap_name: str = "RdYlBu_r",
    scale: str = "log",
    unit: str = "conductivity",
    value_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    max_depth: Optional[float] = None,
    grid_length: int = 400,
    grid_depth: int = 400,
    plot_salinity: bool = False,
    formation_factor: float = 3.2,
    plot_std_dev: bool = False,
    title: Optional[str] = None,
    show_interpolated: bool = True,  # kept for backwards-compatibility but ignored
    show_terrain: bool = False,
    show_altitude: bool = False,
    show_residual: bool = False,
):
    """
    Create a SkyTEM section plot.

    By default this shows a single panel with the raw layered model. Optional
    small panels for terrain, altitude and residual can be enabled via
    show_terrain / show_altitude / show_residual.
    """
    arrays = prepare_line_arrays(df, line_number)
    lengths = arrays["lengths"]
    rho = arrays["rho"]
    depth_top = arrays["depth_top"]
    depth_bottom = arrays["depth_bottom"]
    doi = arrays["doi"]
    surface = np.asarray(arrays["surface"], dtype=float)

    # Normalise profile distance
    lengths = lengths - lengths.min()

    # Convert to chosen quantity
    ec = np.where(rho > 0, 1.0 / rho, np.nan)
    if unit == "conductivity":
        values_matrix = ec
        bar_label = "Conductivity (S/m)"
    elif unit == "resistivity":
        values_matrix = rho
        bar_label = "Resistivity (ohm m)"
    else:
        raise ValueError("unit must be 'conductivity' or 'resistivity'.")

    if np.any(~np.isfinite(surface)):
        fallback = float(np.nanmedian(surface[np.isfinite(surface)])) if np.any(np.isfinite(surface)) else 0.0
        surface = np.where(np.isfinite(surface), surface, fallback)

    finite_depths = depth_bottom[np.isfinite(depth_bottom)]
    if finite_depths.size == 0:
        raise ValueError("No finite depth information available.")

    if max_depth is None:
        plot_depth = float(np.nanmax(finite_depths))
    else:
        plot_depth = float(max_depth)
    plot_depth = max(plot_depth, 5.0)

    doi_depth = np.where(np.isfinite(doi), doi, np.nan)
    if np.any(np.isfinite(doi_depth)):
        max_doi_depth = float(np.nanmax(doi_depth))
        plot_depth = max(plot_depth, max_doi_depth)

    # Convert depths to elevations for plotting (elevation = surface - depth)
    depth_bottom_clipped = np.minimum(depth_bottom, plot_depth)
    layer_top_elev = surface[:, None] - depth_top
    layer_bottom_elev = surface[:, None] - depth_bottom_clipped
    y_max = float(np.nanmax(surface)) + 5.0
    y_min = float(np.nanmin(surface - plot_depth)) - 5.0

    # Colour scale
    vmin, vmax = derive_value_range(values_matrix, scale, value_range)
    cmap = plt.get_cmap(cmap_name)
    if scale == "log":
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Optional per-station info for terrain / altitude / residual panels
    need_aux_panels = show_terrain or show_altitude or show_residual
    terrain_series = None
    altitude_series = None
    residuals = None
    profile_length = None

    if need_aux_panels:
        line_df = df[df["LINE_NO"] == line_number].copy()
        if line_df.empty:
            raise ValueError(f"Could not find raw records for line {line_number}")
        line_df = line_df.reset_index(drop=True)
        line_df["length"] = compute_line_length(line_df)
        line_df["length"] -= line_df["length"].min()
        line_df = line_df.sort_values("length").reset_index(drop=True)
        profile_length = line_df["length"].to_numpy()

        # Terrain
        if show_terrain:
            surface_col = None
            for candidate in ("ELEVATION", "SURFACE", "TOPO", "GROUND_ELEVATION"):
                if candidate in line_df.columns:
                    surface_col = candidate
                    break
            if surface_col:
                terrain_series = pd.to_numeric(line_df[surface_col], errors="coerce")
                terrain_series = terrain_series.interpolate(limit_direction="both")
            else:
                terrain_series = pd.Series(np.zeros(len(line_df)))

        # Altitude
        if show_altitude:
            altitude_col = None
            for candidate in ["ALTITUDE_A-PRIORI_[m]", "ALTITUDE_[m]", "ALTITUDE", "ALT"]:
                if candidate in line_df.columns:
                    altitude_col = candidate
                    break
            if altitude_col:
                altitude_series = pd.to_numeric(line_df[altitude_col], errors="coerce")
                altitude_series = altitude_series.interpolate(limit_direction="both")
            else:
                altitude_series = pd.Series(np.full(len(line_df), 50.0))

        # Residual
        if show_residual:
            residual_col = None
            for candidate in ["RESDATA", "RESTOTAL"]:
                if candidate in line_df.columns:
                    residual_col = candidate
                    break
            if residual_col is None:
                residual_col = "RES_DUMMY"
                line_df[residual_col] = 1.0
            residuals = pd.to_numeric(line_df[residual_col], errors="coerce")

    # Figure layout: optional small top panels + main raw panel
    n_top = int(show_terrain) + int(show_altitude) + int(show_residual)
    if n_top == 0:
        fig, ax_raw = plt.subplots(figsize=(20, 6))
        axes_top: list[plt.Axes] = []
    else:
        height_top = 0.8
        height_raw = 3.0
        height_ratios = [height_top] * n_top + [height_raw]
        fig = plt.figure(figsize=(20, 6 + 2 * n_top))
        gs = plt.GridSpec(n_top + 1, 1, height_ratios=height_ratios, hspace=0.08)

        axes_top = []
        row = 0
        if show_terrain:
            ax_terr = fig.add_subplot(gs[row])
            axes_top.append(ax_terr)
            row += 1
        if show_altitude:
            ax_alt = fig.add_subplot(gs[row])
            axes_top.append(ax_alt)
            row += 1
        if show_residual:
            ax_res = fig.add_subplot(gs[row])
            axes_top.append(ax_res)
            row += 1
        ax_raw = fig.add_subplot(gs[row])

        # Share x across all panels
        for ax in axes_top:
            ax.sharex(ax_raw)

    # Detect gaps and compute per-station bounds
    valid_mask, gap_bounds = detect_gap_areas(lengths, max_gap=30.0)
    if gap_bounds:
        print(f"  -> Detected {len(gap_bounds)} gap(s) >30m, masking data in gap areas")
    station_bounds = compute_station_bounds(lengths)

    # DOI elevations for transparency below DOI
    doi_elevations = None
    if np.any(np.isfinite(doi_depth)):
        doi_elevations = surface - doi_depth

    raw_collection = draw_layer_blocks(
        ax_raw,
        station_bounds,
        layer_top_elev,
        layer_bottom_elev,
        values_matrix,
        cmap,
        norm,
        y_min,
        y_max,
        valid_mask=valid_mask,
        doi_elevations=doi_elevations,
        alpha_below_doi=0.3,
    )

    # DOI line
    doi_mask = np.isfinite(doi_elev := surface - doi_depth)
    if np.any(doi_mask):
        ax_raw.plot(lengths[doi_mask], doi_elev[doi_mask], "k--", linewidth=1.2, label="DOI")
        ax_raw.legend(loc="upper right", fontsize=10, frameon=False)

    # Axes formatting for raw panel
    ax_raw.set_xlim(lengths.min(), lengths.max())
    ax_raw.set_ylim(y_min, y_max)
    ax_raw.set_xlabel("Distance along line (m)", fontsize=12)
    ax_raw.set_ylabel("Elevation (m)", fontsize=12)
    ax_raw.grid(True, color="0.85", linestyle="--", linewidth=0.6)
    ax_raw.plot(lengths, surface, "k-", linewidth=1.0)
    ax_raw.set_title("Raw layered model (filled by layer thickness)", fontsize=14, pad=8)

    # Optional top panels
    if need_aux_panels and profile_length is not None:
        row = 0
        if show_terrain and terrain_series is not None:
            ax_terr = axes_top[row]
            ax_terr.plot(profile_length, terrain_series, "k-", linewidth=1.0)
            tmin = float(np.nanmin(terrain_series))
            tmax = float(np.nanmax(terrain_series))
            trng = tmax - tmin
            if trng > 0:
                ax_terr.set_ylim(tmin - trng * 0.05, tmax + trng * 0.05)
            else:
                ax_terr.set_ylim(tmin - 5, tmax + 5)
            ax_terr.set_ylabel("Terrain (m)", fontsize=10)
            ax_terr.grid(True, color="0.9", linestyle="--", linewidth=0.5)
            ax_terr.tick_params(labelbottom=False)
            row += 1

        if show_altitude and altitude_series is not None:
            ax_alt = axes_top[row]
            ax_alt.plot(profile_length, altitude_series, "k-", linewidth=1.0)
            ax_alt.set_ylim(0, float(np.nanmax(altitude_series)) * 1.05)
            ax_alt.set_ylabel("Alt (m agl)", fontsize=10)
            ax_alt.grid(True, color="0.9", linestyle="--", linewidth=0.5)
            ax_alt.tick_params(labelbottom=False)
            row += 1

        if show_residual and residuals is not None:
            ax_res = axes_top[row]
            ax_res.scatter(
                profile_length,
                residuals,
                c=residuals,
                cmap="RdYlGn_r",
                s=12,
                alpha=0.8,
            )
            ax_res.axhline(1.0, color="green", linestyle="-", linewidth=1.0, alpha=0.7)
            ax_res.set_ylabel("Residual", fontsize=10)
            if np.any(np.isfinite(residuals)):
                res_max = np.nanpercentile(residuals, 95)
                ax_res.set_ylim(0.5, max(2.5, res_max * 1.1))
            ax_res.grid(True, color="0.9", linestyle="--", linewidth=0.5)
            ax_res.tick_params(labelbottom=False)

    overall_title = title or f"SkyTEM line {line_number}"
    fig.suptitle(overall_title, fontsize=16, y=0.96)
    plt.subplots_adjust(top=0.9, right=0.87, left=0.08, hspace=0.08 if n_top > 0 else 0.15)

    # Colourbar
    sm = raw_collection
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_raw,
        pad=0.02,
        fraction=0.04,
    )
    cbar.set_label(bar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    return fig, [ax_raw], sm


def plot_doi_boxplot(
    df: pd.DataFrame,
    line_numbers: Optional[Sequence[int]] = None,
    *,
    use_conservative: bool = False,
) -> plt.Figure:
    """
    Create a box plot summarising DOI depths per line.

    Parameters
    ----------
    df : DataFrame
        Aarhus-style inversion result with at least LINE_NO and a DOI column.
    line_numbers : optional sequence of int
        If given, restrict analysis to these LINE_NO values. Otherwise all lines
        present in the DataFrame are used.
    use_conservative : bool, default False
        If True, prefer DOI_CONSERVATIVE over DOI_STANDARD.
    """
    if "LINE_NO" not in df.columns:
        raise ValueError("DOI boxplot requires a LINE_NO column in the input DataFrame.")

    doi_candidates: List[str]
    if use_conservative:
        doi_candidates = ["DOI_CONSERVATIVE", "DOI_STANDARD"]
    else:
        doi_candidates = ["DOI_STANDARD", "DOI_CONSERVATIVE"]

    doi_col = None
    for c in doi_candidates:
        if c in df.columns:
            doi_col = c
            break
    if doi_col is None:
        raise ValueError("No DOI_STANDARD or DOI_CONSERVATIVE column found for DOI analysis.")

    work = df.copy()
    if line_numbers is not None:
        line_set = {int(ln) for ln in line_numbers}
        work = work[work["LINE_NO"].isin(line_set)].copy()
        if work.empty:
            raise ValueError(f"No rows found for requested line(s): {sorted(line_set)}")

    # Build per-line DOI arrays
    grouped = work.groupby("LINE_NO")
    lines: List[int] = []
    data: List[np.ndarray] = []
    for ln, g in grouped:
        vals = pd.to_numeric(g[doi_col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        lines.append(int(ln))
        data.append(vals)

    if not data:
        raise ValueError("No finite DOI values available for boxplot.")

    # Sort by line number for stable layout
    order = np.argsort(lines)
    lines = [lines[i] for i in order]
    data = [data[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(
        data,
        labels=[str(ln) for ln in lines],
        patch_artist=True,
        showfliers=True,
    )

    # Simple styling
    for patch in bp["boxes"]:
        patch.set_facecolor("#a6cee3")
    for median in bp["medians"]:
        median.set_color("k")
        median.set_linewidth(1.5)

    ax.set_xlabel("Line number (LINE_NO)", fontsize=11)
    ax.set_ylabel("DOI depth (m)", fontsize=11)
    ax.set_title(
        f"DOI depth distribution per line ({doi_col})",
        fontsize=13,
        pad=8,
    )
    ax.grid(True, color="0.9", linestyle="--", linewidth=0.6, axis="y")

    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SkyTEM sections with raw and interpolated panels.")
    parser.add_argument("file", type=Path, help="SkyTEM inversion file or SimPEG stacked CSV.")
    parser.add_argument(
        "--input-type",
        choices=["aarhus", "simpeg"],
        default="aarhus",
        help="Type of input file: 'aarhus' (Workbench export) or 'simpeg' (stacked CSV from SimPEG).",
    )
    parser.add_argument("--line", type=int, nargs="+", help="Line numbers to plot (Aarhus mode, default: all).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG outputs.")
    parser.add_argument("--cmap", default="RdYlBu_r", help="Matplotlib colormap name.")
    parser.add_argument("--scale", choices=["linear", "log"], default="linear", help="Colour scale type (linear or log).")
    parser.add_argument("--unit", choices=["conductivity", "resistivity"], default="conductivity")
    parser.add_argument("--value-min", type=float, default=None, help="Minimum value for colour scale (auto-detected if not specified).")
    parser.add_argument("--value-max", type=float, default=None, help="Maximum value for colour scale (auto-detected if not specified).")
    parser.add_argument("--max-depth", type=float, default=55.0, help="Clip plot to this depth (m).")
    parser.add_argument("--grid-length", type=int, default=400, help="Number of interpolation samples along line.")
    parser.add_argument("--grid-depth", type=int, default=400, help="Number of interpolation samples with depth.")
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma for interpolated grid (0=no smoothing, try 2-5 for smoother visualization).",
    )
    parser.add_argument("--plot-salinity", action="store_true", help="Add salinity classification panel.")
    parser.add_argument("--formation-factor", type=float, default=3.2, help="Formation factor for salinity conversion.")
    parser.add_argument("--title", default=None, help="Custom figure title.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    parser.add_argument("--no-save", action="store_true", help="Do not write PNG files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_type == "aarhus":
        df = read_skytem_xyz(args.file)
        print(f"Loaded {len(df)} soundings from {args.file.name}")

        lines: Iterable[int]
        if args.line:
            lines = args.line
        else:
            lines = sorted(int(val) for val in df["LINE_NO"].dropna().unique())
            print(f"No --line specified; plotting all {len(lines)} lines: {lines}")

        for line_number in lines:
            print(f"Plotting SkyTEM line {line_number} ...")
            fig, axes, _ = plot_skytem_section(
                df,
                line_number,
                cmap_name=args.cmap,
                scale=args.scale,
                unit=args.unit,
                value_range=(args.value_min, args.value_max),
                max_depth=args.max_depth,
                grid_length=args.grid_length,
                grid_depth=args.grid_depth,
                plot_salinity=args.plot_salinity,
                formation_factor=args.formation_factor,
                title=args.title,
            )

            if not args.no_save:
                stem = sanitize_filename_component(args.file.stem)
                suffix = f"{args.unit}"
                if args.plot_salinity:
                    suffix += "_sal"
                filename = output_dir / f"{stem}_line_{line_number}_{suffix}.png"
                fig.savefig(filename, dpi=150, bbox_inches="tight")
                print(f"  -> saved {filename}")

            if args.show:
                plt.show()
            plt.close(fig)
    else:
        # SimPEG stacked CSV mode: one line per file, with stacked layers
        df_sim = pd.read_csv(args.file)
        print(f"Loaded {len(df_sim)} layer rows from {args.file.name} (SimPEG stacked CSV)")

        fig, axes, _ = plot_simpeg_section(
            df_sim,
            cmap_name=args.cmap,
            scale=args.scale,
            unit=args.unit,
            value_range=(args.value_min, args.value_max),
            max_depth=args.max_depth,
            grid_length=args.grid_length,
            grid_depth=args.grid_depth,
            plot_salinity=args.plot_salinity,
            formation_factor=args.formation_factor,
            title=args.title,
            smooth_sigma=args.smooth_sigma,
        )

        if not args.no_save:
            stem = sanitize_filename_component(args.file.stem)
            suffix = f"{args.unit}_simpeg"
            if args.plot_salinity:
                suffix += "_sal"
            filename = output_dir / f"{stem}_{suffix}.png"
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"  -> saved {filename}")

        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()

