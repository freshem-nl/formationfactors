import csv
import json
import tempfile
import unittest
from pathlib import Path

from build_explorer import build, prepare_data


class ExplorerTests(unittest.TestCase):
    def write_data(self, directory, **overrides):
        row = {
            "chloride_g_l": "1", "conductivity_s_m": "0.05", "xy_distance_m": "2",
            "screen_gap_m": "0", "chloride_z_nap_m": "-20", "log_z_nap_m": "-20",
            "chloride_csv_row": "2", "log_source": "LAS", "log_id": "B01",
            "chloride_regis_layer": "NUBXz1", "log_regis_layer": "NUBXz2",
            "input_regis_layer": "NUBXz1", "same_regis_layer": "False",
            "input_regis_description": "Formatie van Boxtel, eerste zandige eenheid",
            "locationcode": "B01",
        } | overrides
        path = Path(directory) / "input.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_assignment_metadata_and_stable_id(self):
        with tempfile.TemporaryDirectory() as directory:
            row = prepare_data(self.write_data(directory))[0]
        self.assertEqual(row["pair_id"], "2:LAS")
        self.assertEqual(row["chloride_formation"], "Formatie van Boxtel")
        self.assertEqual(row["log_formation"], "Unmapped unit: NUBXz2")
        self.assertEqual(row["chloride_g_l"], 1.0)

    def test_invalid_data_is_reported(self):
        for overrides in ({"conductivity_s_m": "0"}, {"chloride_g_l": "nan"},
                          {"log_source": "unknown"}, {"log_id": ""}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    prepare_data(self.write_data(directory, **overrides))

    def test_ec_units_temperature_and_inverse_relation(self):
        for chloride, expected_ec25 in (("0", 1.25), ("1", 1450 / 360), ("1.25", 1700 / 360), ("5", 5450 / 360)):
            with self.subTest(chloride=chloride), tempfile.TemporaryDirectory() as directory:
                row = prepare_data(self.write_data(directory, chloride_g_l=chloride))[0]
                self.assertAlmostEqual(row["ec_bulk_11_ms_cm"], 0.5)
                self.assertAlmostEqual(row["ec_bulk_25_ms_cm"], 0.68)
                self.assertAlmostEqual(row["ec_bulk_25_ms_cm"], 1.36 * row["ec_bulk_11_ms_cm"])
                self.assertAlmostEqual(row["ec_groundwater_25_ms_cm"], expected_ec25)
                self.assertAlmostEqual(row["ec_groundwater_11_ms_cm"], expected_ec25 / 1.36)
                self.assertAlmostEqual(
                    360 * (1.36 * row["ec_groundwater_11_ms_cm"]) - 450,
                    1000 * float(chloride),
                )
                self.assertGreater(row["ec_groundwater_11_ms_cm"], 0)
                self.assertEqual(row["chloride_g_l"], float(chloride))

    def test_numeric_exports_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_data(directory, screen_bottom_nap_m="-21", log_regis_top_nap_m="")
            row = prepare_data(path)[0]
            self.assertEqual(row["screen_bottom_nap_m"], -21.0)
            self.assertIsNone(row["log_regis_top_nap_m"])
            lines = path.read_text(encoding="utf-8").splitlines()
            with path.open("a", encoding="utf-8", newline="") as stream:
                stream.write(lines[1] + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate pair ID"):
                prepare_data(path)

    def test_standalone_html_escapes_source_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_data(directory, locationcode="</script><script>alert(1)</script>")
            output = Path(directory) / "explorer.html"
            build(path, output)
            html = output.read_text(encoding="utf-8")
        self.assertNotIn('<script src=', html)
        self.assertNotIn("__DATA_JSON__", html)
        self.assertLess(html.index('id="parameter-table"'), html.index('id="fit-table"'))
        payload = html.split('<script id="dataset" type="application/json">')[1].split("</script>")[0]
        self.assertNotIn("<", payload)
        self.assertEqual(json.loads(payload)[0]["locationcode"], "</script><script>alert(1)</script>")
