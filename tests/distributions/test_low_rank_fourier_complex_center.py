"""Regression coverage for low-rank identity Fourier normalization."""

import unittest

import numpy as np
import pyrecest.backend
from pyrecest.distributions.hypertorus.low_rank_hypertoroidal_fourier_distribution import (
    LowRankHypertoroidalFourierDistribution,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",  # pylint: disable=no-member
    reason="Low-rank Fourier prototype is NumPy-only",
)
class TestLowRankFourierComplexCenter(unittest.TestCase):
    def test_identity_normalization_rejects_complex_center_coefficient(self):
        coefficients = np.zeros(3, dtype=np.complex128)
        coefficients[1] = 1.0 + 1.0j

        with self.assertRaisesRegex(
            ValueError, "Center coefficient must be real-valued"
        ):
            LowRankHypertoroidalFourierDistribution.from_dense(coefficients)


if __name__ == "__main__":
    unittest.main()
