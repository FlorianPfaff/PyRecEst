import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture


class GaussianMixtureParameterShapeTest(unittest.TestCase):
    def test_rejects_missing_component_covariance(self):
        means = array([0.0, 2.0])
        covariance_matrices = array([[[1.0]]])

        with self.assertRaisesRegex(
            ValueError,
            "covariance_matrices must have shape",
        ):
            GaussianMixture.mixture_parameters_to_gaussian_parameters(
                means,
                covariance_matrices,
                array([0.25, 0.75]),
            )

    def test_rejects_covariance_without_component_axis(self):
        means = array([[0.0, 0.0], [1.0, 1.0]])
        covariance_matrices = array([[1.0, 0.0], [0.0, 1.0]])

        with self.assertRaisesRegex(
            ValueError,
            "covariance_matrices must have shape",
        ):
            GaussianMixture.mixture_parameters_to_gaussian_parameters(
                means,
                covariance_matrices,
                array([0.5, 0.5]),
            )

    def test_matching_component_shapes_preserve_moment_matching(self):
        mean, covariance = (
            GaussianMixture.mixture_parameters_to_gaussian_parameters(
                array([0.0, 2.0]),
                array([[[1.0, 3.0]]]),
                array([0.25, 0.75]),
            )
        )

        npt.assert_allclose(to_numpy(mean), np.array([1.5]))
        npt.assert_allclose(to_numpy(covariance), np.array([[3.25]]))


if __name__ == "__main__":
    unittest.main()
