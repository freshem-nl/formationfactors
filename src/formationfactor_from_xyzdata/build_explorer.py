"""Build a self-contained, offline groundwater/bulk EC explorer at 25 degrees C."""

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "3-input" / "formationfactor_from_xyzdata" / "chloride_geophysics_regis.csv"
OUTPUT = ROOT / "reports" / "figures" / "formationfactor_from_xyzdata" / "explorer.html"
NUMERIC = (
    "chloride_g_l", "conductivity_s_m", "xy_distance_m", "screen_gap_m",
    "chloride_z_nap_m", "log_z_nap_m",
)
OPTIONAL_NUMERIC = (
    "chloride_x_rd_m", "chloride_y_rd_m", "log_x_rd_m", "log_y_rd_m",
    "screen_top_nap_m", "screen_bottom_nap_m", "log_depth_m", "log_surface_nap_m",
    "sample_z_difference_m", "resistivity_ohm_m", "chloride_screen_overlap_fraction",
    "chloride_regis_cell_x_rd_m", "chloride_regis_cell_y_rd_m",
    "log_regis_cell_x_rd_m", "log_regis_cell_y_rd_m",
    "chloride_regis_top_nap_m", "chloride_regis_bottom_nap_m",
    "log_regis_top_nap_m", "log_regis_bottom_nap_m",
)


def prepare_data(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {*NUMERIC, "chloride_csv_row", "log_source", "log_id",
                    "chloride_regis_layer", "log_regis_layer", "input_regis_layer",
                    "input_regis_description", "same_regis_layer", "locationcode"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("The matched dataset is empty.")
    descriptions: dict[str, set[str]] = {}
    for row in rows:
        if row["input_regis_layer"] and row["input_regis_description"]:
            descriptions.setdefault(row["input_regis_layer"], set()).add(row["input_regis_description"])
    ids = set()
    for row in rows:
        for column in NUMERIC:
            row[column] = float(row[column])
            if not math.isfinite(row[column]):
                raise ValueError(f"Nonfinite {column} in chloride row {row['chloride_csv_row']}")
        for column in OPTIONAL_NUMERIC:
            if column not in row:
                continue
            row[column] = float(row[column]) if row[column] else None
            if row[column] is not None and not math.isfinite(row[column]):
                raise ValueError(f"Nonfinite {column} in chloride row {row['chloride_csv_row']}")
        if row["chloride_g_l"] < 0 or row["conductivity_s_m"] <= 0:
            raise ValueError("Expected nonnegative chloride and positive conductivity.")
        row["ec_bulk_11_ms_cm"] = 10.0 * row["conductivity_s_m"]
        row["ec_bulk_25_ms_cm"] = 1.36 * row["ec_bulk_11_ms_cm"]
        row["ec_groundwater_25_ms_cm"] = (1000.0 * row["chloride_g_l"] + 450.0) / 360.0
        row["ec_groundwater_11_ms_cm"] = row["ec_groundwater_25_ms_cm"] / 1.36
        if row["log_source"] not in {"LAS", "ECPT"} or not row["log_id"]:
            raise ValueError("Expected a LAS/ECPT source and a log ID.")
        row["pair_id"] = f"{row['chloride_csv_row']}:{row['log_source']}"
        if row["pair_id"] in ids:
            raise ValueError(f"Duplicate pair ID: {row['pair_id']}")
        ids.add(row["pair_id"])
        for prefix in ("chloride", "log"):
            code = row[f"{prefix}_regis_layer"]
            names = descriptions.get(code, set())
            description = next(iter(names)) if len(names) == 1 else ""
            row[f"{prefix}_formation"] = (
                description.split(",")[0] if description
                else f"Unmapped unit: {code}" if code else "No REGIS assignment"
            )
    return rows


def build(input_path: Path, output_path: Path) -> None:
    rows = prepare_data(input_path)
    template = (HERE / "explorer.html").read_text(encoding="utf-8")
    # Prevent source text from closing the application/json script element.
    payload = json.dumps(rows, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    javascript = (HERE / "explorer.js").read_text(encoding="utf-8")
    if "</script" in javascript.lower():
        raise ValueError("JavaScript must not contain a closing script tag.")
    html = template.replace("__DATA_JSON__", payload).replace(
        '<script src="explorer.js"></script>', f"<script>\n{javascript}\n</script>",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path} ({len(rows):,} pairs, {output_path.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.input, args.output)


if __name__ == "__main__":
    main()
