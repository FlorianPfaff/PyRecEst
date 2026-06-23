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


if __name__ == "__main__":
    unittest.main()
