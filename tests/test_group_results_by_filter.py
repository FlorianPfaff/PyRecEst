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

    def test_heterogeneous_metric_keys_stay_aligned(self):
        rows = [
            {"name": "pf", "parameter": 3, "std": 0.3},
            {"name": "pf", "parameter": 1, "score": 1.0},
            {"name": "pf", "parameter": 2, "score": 2.0, "std": 0.2},
        ]

        grouped = group_results_by_filter(rows)

        self.assertEqual(grouped["pf"]["parameter"], [1, 2, 3])
        self.assertEqual(grouped["pf"]["score"], [1.0, 2.0, None])
        self.assertEqual(grouped["pf"]["std"], [None, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
