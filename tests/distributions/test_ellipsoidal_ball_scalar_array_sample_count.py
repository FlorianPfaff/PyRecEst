import unittest

import numpy as np

from pyrecest.backend import array, diag
from pyrecest.distributions import EllipsoidalBallUniformDistribution


class TestEllipsoidalBallScalarArraySampleCount(unittest.TestCase):
    def setUp(self):
        self.distribution = EllipsoidalBallUniformDistribution(
            array([0.0, 0.0]), diag(array([1.0, 1.0]))
        )

    def test_accepts_zero_dimensional_integer_array(self):
        samples = self.distribution.sample(np.array(3, dtype=np.int64))

        self.assertEqual(samples.shape, (3, 2))

    def test_accepts_clear_mask_integer_scalar(self):
        samples = self.distribution.sample(np.ma.array(2, mask=False))

        self.assertEqual(samples.shape, (2, 2))

    def test_rejects_non_scalar_integer_array(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.distribution.sample(np.array([3], dtype=np.int64))

    def test_rejects_masked_integer_scalar(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.distribution.sample(np.ma.array(3, mask=True))


if __name__ == "__main__":
    unittest.main()
