import unittest

import numpy as np

import pyrecest.backend
from pyrecest.distributions.circle.wrapped_laplace_distribution import (
    WrappedLaplaceDistribution,
)


class WrappedLaplaceParameterScalarShapeTest(unittest.TestCase):
    def test_normalizes_one_element_parameter_arrays_to_scalars(self):
        distribution = WrappedLaplaceDistribution(
            np.array([2.0]),
            np.array([1.3]),
        )

        for value in (
            distribution.lambda_,
            distribution.kappa,
            distribution.pdf(1.0),
            distribution.trigonometric_moment(1),
        ):
            with self.subTest(value=value):
                converted = np.asarray(pyrecest.backend.to_numpy(value))
                self.assertEqual(converted.shape, ())


if __name__ == "__main__":
    unittest.main()
