import unittest

import numpy as np

import pyrecest.backend
from pyrecest.distributions.circle.wrapped_cauchy_distribution import (
    WrappedCauchyDistribution,
)


class WrappedCauchyParameterScalarShapeTest(unittest.TestCase):
    def test_normalizes_one_element_parameter_arrays_to_scalars(self):
        distribution = WrappedCauchyDistribution(
            np.array([0.7]),
            np.array([0.5]),
        )

        for value in (
            distribution.mu,
            distribution.gamma,
            distribution.trigonometric_moment(1),
        ):
            with self.subTest(value=value):
                converted = np.asarray(pyrecest.backend.to_numpy(value))
                self.assertEqual(converted.shape, ())


if __name__ == "__main__":
    unittest.main()
