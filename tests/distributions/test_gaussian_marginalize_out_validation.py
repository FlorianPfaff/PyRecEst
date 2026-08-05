import unittest

import numpy as np

from pyrecest.backend import array, to_numpy
from pyrecest.distributions import GaussianDistribution


class GaussianMarginalizeOutValidationTest(unittest.TestCase):
    def setUp(self):
        self.distribution = GaussianDistribution(
            array([1.0, 2.0]),
            array([[1.1, 0.4], [0.4, 0.9]]),
        )

    def test_rejects_duplicate_dimensions(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.distribution.marginalize_out([0, 0])

    def test_rejects_empty_marginal(self):
        with self.assertRaisesRegex(ValueError, "leave at least one dimension"):
            self.distribution.marginalize_out([0, 1])

    def test_rejects_temporal_dimensions(self):
        temporal_dimensions = (
            np.timedelta64(0, "ns"),
            np.timedelta64(0, "us"),
            np.datetime64("1970-01-01T00:00:00.000000000", "ns"),
            np.array(np.timedelta64(0, "ns")),
            np.array(np.datetime64("1970-01-01T00:00:00.000000000", "ns")),
            [np.timedelta64(0, "ns")],
        )

        for dimensions in temporal_dimensions:
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(ValueError, "integer indices"):
                    self.distribution.marginalize_out(dimensions)

    def test_preserves_numpy_integer_dimensions(self):
        marginal = self.distribution.marginalize_out(np.int64(0))
        self.assertEqual(marginal.dim, 1)

    def test_accepts_zero_dimensional_integer_array_dimensions(self):
        marginal = self.distribution.marginalize_out(np.array(0, dtype=np.int64))

        self.assertEqual(marginal.dim, 1)
        np.testing.assert_allclose(to_numpy(marginal.mu), np.array([2.0]))
        np.testing.assert_allclose(to_numpy(marginal.C), np.array([[0.9]]))

    def test_rejects_zero_dimensional_noninteger_or_masked_arrays(self):
        invalid_dimensions = (
            np.array(0.5),
            np.array("0"),
            np.ma.array(0, mask=True),
        )

        for dimensions in invalid_dimensions:
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(ValueError, "integer indices"):
                    self.distribution.marginalize_out(dimensions)


if __name__ == "__main__":
    unittest.main()
