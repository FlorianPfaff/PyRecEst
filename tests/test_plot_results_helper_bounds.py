import unittest

from pyrecest.evaluation.plot_results import get_min_max_param


class PlotResultsHelperBoundsTest(unittest.TestCase):
    def test_fallback_without_numeric_parameters(self):
        summaries = [
            {"name": "kf", "parameter": None},
            {"name": "wn", "parameter": None},
        ]

        self.assertEqual(get_min_max_param(summaries), (0.0, 1.0))

    def test_numeric_parameter_bounds(self):
        summaries = [
            {"name": "kf", "parameter": None},
            {"name": "pf", "parameter": 100},
            {"name": "pf", "parameter": 10},
        ]

        self.assertEqual(get_min_max_param(summaries), (10, 100))


if __name__ == "__main__":
    unittest.main()
