import math
import unittest

from pyrecest.evaluation.group_results_by_filter import group_results_by_filter


class TestGroupResultsByFilter(unittest.TestCase):
    def test_mixed_parameters_do_not_crash_sorting(self):
        rows = [
            {"name": "pf", "parameter": "b", "score": 3.0},
            {"name": "pf", "parameter": 1, "score": 1.0},
            {"name": "pf", "parameter": None, "score": 0.0},
        ]

        grouped = group_results_by_filter(rows)

        self.assertEqual(grouped["pf"]["parameter"], [None, 1, "b"])
        self.assertEqual(grouped["pf"]["score"], [0.0, 1.0, 3.0])

    def test_nan_parameter_does_not_disrupt_numeric_order(self):
        rows = [
            {"name": "pf", "parameter": 1.0, "score": "one"},
            {"name": "pf", "parameter": float("nan"), "score": "nan"},
            {"name": "pf", "parameter": 0.0, "score": "zero"},
            {"name": "pf", "parameter": "b", "score": "category"},
            {"name": "pf", "parameter": None, "score": "none"},
        ]

        grouped = group_results_by_filter(rows)["pf"]

        self.assertEqual(grouped["score"], ["none", "zero", "one", "nan", "category"])
        self.assertTrue(math.isnan(grouped["parameter"][3]))


if __name__ == "__main__":
    unittest.main()
