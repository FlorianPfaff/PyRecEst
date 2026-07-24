import unittest

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)


class GaussianMixtureConstructorTest(unittest.TestCase):
    def test_rejects_empty_component_list(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            GaussianMixture([], array([]))

    def test_rejects_non_gaussian_components(self):
        component = LinearDiracDistribution(array([0.0]), array([1.0]))

        with self.assertRaisesRegex(ValueError, "GaussianDistribution"):
            GaussianMixture([component], array([1.0]))

    def test_parameter_conversion_accepts_sequence_weights(self):
        means = array([[0.0], [2.0]])
        covariance_matrices = array([[[1.0, 3.0]]])

        for weights in ([0.25, 0.75], (0.25, 0.75)):
            with self.subTest(weight_type=type(weights).__name__):
                mean, covariance = (
                    GaussianMixture.mixture_parameters_to_gaussian_parameters(
                        means, covariance_matrices, weights
                    )
                )

                np.testing.assert_allclose(to_numpy(mean), [1.5])
                np.testing.assert_allclose(to_numpy(covariance), [[3.25]])


if __name__ == "__main__":
    unittest.main()
