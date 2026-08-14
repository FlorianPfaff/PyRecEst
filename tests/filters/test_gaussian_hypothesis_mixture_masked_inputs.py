import unittest

import numpy as np
from pyrecest.filters import (
    WeightedGaussianHypothesis,
    normalize_log_weights,
)


class GaussianHypothesisMaskedInputTest(unittest.TestCase):
    def test_masked_hypothesis_fields_are_rejected(self):
        cases = (
            (
                "mean",
                lambda: WeightedGaussianHypothesis(
                    np.ma.array([0.0], mask=[True]),
                    np.array([[1.0]]),
                ),
            ),
            (
                "covariance",
                lambda: WeightedGaussianHypothesis(
                    np.array([0.0]),
                    np.ma.array([[1.0]], mask=[[True]]),
                ),
            ),
            (
                "log_weight",
                lambda: WeightedGaussianHypothesis(
                    np.array([0.0]),
                    np.array([[1.0]]),
                    log_weight=np.ma.array(0.0, mask=True),
                ),
            ),
        )

        for field_name, construct in cases:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    construct()

    def test_masked_log_weight_vectors_are_rejected(self):
        invalid_values = (
            np.ma.array([0.0, 1.0], mask=[False, True]),
            [0.0, np.ma.masked],
            np.array([0.0, np.ma.masked], dtype=object),
        )

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "log_weights"):
                    normalize_log_weights(values)

    def test_clear_mask_wrappers_remain_supported(self):
        hypothesis = WeightedGaussianHypothesis(
            np.ma.array([0.0], mask=[False]),
            np.ma.array([[1.0]], mask=[[False]]),
            log_weight=np.ma.array(0.0, mask=False),
        )
        weights = normalize_log_weights(
            np.ma.array([0.0, 0.0], mask=[False, False])
        )

        np.testing.assert_array_equal(hypothesis.mean, np.array([0.0]))
        np.testing.assert_array_equal(hypothesis.covariance, np.array([[1.0]]))
        self.assertEqual(hypothesis.log_weight, 0.0)
        np.testing.assert_allclose(weights, np.array([0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
