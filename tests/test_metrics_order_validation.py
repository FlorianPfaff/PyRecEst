import unittest

import numpy as np

from pyrecest.utils.metrics import gospa_distance, ospa_distance


class TestFiniteSetMetricOrderValidation(unittest.TestCase):
    def test_set_distance_order_rejects_nonfinite_values(self):
        for metric in (ospa_distance, gospa_distance):
            for order in (np.nan, np.inf, -np.inf):
                with self.subTest(metric=metric.__name__, order=order):
                    with self.assertRaisesRegex(
                        ValueError,
                        "order must be finite and at least 1",
                    ):
                        metric([[0.0]], [[0.0]], cutoff=1.0, order=order)


if __name__ == "__main__":
    unittest.main()
