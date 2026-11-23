import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from build_dataset import classify_regis, match_logs


class DatasetTests(unittest.TestCase):
    def test_match_prefers_inside_across_chunks_and_limits_fallback(self):
        chloride = pd.DataFrame({
            "chloride_csv_row": [2, 3, 4, 5],
            "chloride_x_rd_m": [0.0, 0.0, 0.0, 251.0],
            "chloride_y_rd_m": [0.0, 0.0, 0.0, 0.0],
            "screen_bottom_nap_m": [-5.0, -2.0, -9.0, -5.0],
            "screen_top_nap_m": [-4.0, -1.0, -8.0, -4.0],
            "chloride_z_nap_m": [-4.5, -1.5, -8.5, -4.5],
        })
        ecpt = pd.DataFrame({
            "ecpt_id": ["near", "near", "near", "inside"],
            "x_rd": [0.0, 0.0, 0.0, -250.0], "y_rd": 0.0, "surface_level_m": 2.0,
            "depth_m": [5.9, 5.0, 10.1, 6.5], "ec_sm": [0.1, 0.2, 0.0, 0.3],
            "notes": "", "notes_long": "",
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ecpt.csv"
            ecpt.to_csv(path, index=False)
            matches, stats = match_logs(chloride, path, "ECPT", 250, 1, chunk_size=2)
        matches = matches.set_index("chloride_csv_row").sort_index()
        self.assertEqual(list(matches.index), [2, 3])
        self.assertEqual(matches["log_id"].tolist(), ["inside", "near"])
        self.assertEqual(matches["log_z_nap_m"].tolist(), [-4.5, -3.0])
        self.assertEqual(matches["screen_gap_m"].tolist(), [0.0, 1.0])
        self.assertEqual(matches["log_csv_row"].tolist(), [5, 3])
        self.assertEqual(stats, {"input": 4, "invalid": 1})

    def test_regis_uses_bounds_and_masks_nodata(self):
        dims = ("layer", "y", "x")
        ds = xr.Dataset(
            {
                "top": (dims, np.array([[[0, 0]], [[0, -9999]], [[-10, -9999]]],
                                       dtype=np.float32), {"units": "m-NAP"}),
                "bottom": (dims, np.array([[[-9999, -9999]], [[-10, -9999]], [[-20, -9999]]],
                                          dtype=np.float32), {"units": "m-NAP"}),
                "x_bounds": (("x", "bounds"), [[0, 100], [100, 200]]),
                "y_bounds": (("y", "bounds"), [[0, 100]]),
                "crs": ((), 0, {"epsg_code": '"EPSG:28992"'}),
            },
            coords={"x": [0, 100], "y": [0], "layer": [b"mv", b"upper", b"lower"]},
        )
        points = pd.DataFrame({
            "x": [99.0, 99.0, 100.0, 200.0], "y": 10.0, "z": [-5.0, -10.0, -5.0, -5.0],
            "screen_bottom": -12.0, "screen_top": -2.0,
        })
        units = classify_regis(ds, points)
        self.assertEqual(units["regis_status"].tolist(),
                         ["matched", "matched", "no_layer_at_elevation", "outside_grid"])
        self.assertEqual(units["regis_layer"].iloc[:2].tolist(), ["upper", "lower"])
        self.assertEqual(units["regis_cell_x_rd_m"].iloc[:3].tolist(), [50.0, 50.0, 150.0])
        self.assertTrue(pd.isna(units["regis_layer"].iloc[2]))
        self.assertAlmostEqual(units["screen_overlap_fraction"].iloc[0], 0.8)
        self.assertEqual(units["regis_bottom_nap_m"].iloc[0], -10.0)
        self.assertTrue(pd.api.types.is_float_dtype(units["regis_bottom_nap_m"]))
