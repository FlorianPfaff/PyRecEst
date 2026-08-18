import unittest

import numpy.testing as npt
import pyrecest.backend

from pyrecest.backend import array
from pyrecest.distributions.circle.piecewise_constant_distribution import (
    PiecewiseConstantDistribution,
)
from pyrecest.filters.piecewise_constant_filter import PiecewiseConstantFilter


@unittest.skipUnless(
    pyrecest.backend.__backend_name__ == "numpy",  # pylint: disable=no-member
    "SciPy numerical integration regression is NumPy-only.",
)
class TestPiecewiseConstantFilterCircularWrapping(unittest.TestCase):
    def setUp(self):
        self.uniform_noise = PiecewiseConstantDistribution(array([1.0]))

    def test_system_matrix_wraps_additive_model_outputs(self):
        matrix = PiecewiseConstantFilter.calculate_system_matrix_numerically(
            2,
            lambda x, w: x + w,
            self.uniform_noise,
        )

        npt.assert_allclose(matrix, 0.5, atol=2e-4)
        npt.assert_allclose(matrix.sum(axis=0), 1.0, atol=2e-4)

    def test_measurement_matrix_wraps_additive_model_outputs(self):
        matrix = PiecewiseConstantFilter.calculate_measurement_matrix_numerically(
            2,
            2,
            lambda x, v: x + v,
            self.uniform_noise,
        )

        npt.assert_allclose(matrix, 0.5, atol=2e-4)
        npt.assert_allclose(matrix.sum(axis=0), 1.0, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
