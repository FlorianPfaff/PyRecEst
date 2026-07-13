import unittest

import numpy as np
import pyrecest.backend

from pyrecest.distributions.circle.wrapped_cauchy_distribution import (
    WrappedCauchyDistribution,
)


class WrappedCauchyScalarParameterShapeTest(unittest.TestCase):
    def test_normalizes_one_element_parameter_arrays_to_scalars(self):
        distribution = WrappedCauchyDistribution(
            np.array([0.7]),
            np.array([0.5]),
        )

        values = {
            "mu": distribution.mu,
            "gamma": distribution.gamma,
            "trigonometric_moment": distribution.trigonometric_moment(1),
        }
        for name, value in values.items():
            with self.subTest(name=name):
                converted = np.asarray(pyrecest.backend.to_numpy(value))
                self.assertEqual(converted.shape, ())

    def test_one_element_parameters_match_scalar_parameters(self):
        array_parameters = WrappedCauchyDistribution(
            np.array([0.7]),
            np.array([0.5]),
        )
        scalar_parameters = WrappedCauchyDistribution(0.7, 0.5)

        np.testing.assert_allclose(
            pyrecest.backend.to_numpy(array_parameters.trigonometric_moment(2)),
            pyrecest.backend.to_numpy(scalar_parameters.trigonometric_moment(2)),
        )
        np.testing.assert_allclose(
            pyrecest.backend.to_numpy(array_parameters.pdf([0.1, 0.7, 1.2])),
            pyrecest.backend.to_numpy(scalar_parameters.pdf([0.1, 0.7, 1.2])),
        )


if __name__ == "__main__":
    unittest.main()
