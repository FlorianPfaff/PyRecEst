import unittest
import warnings

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture


class MixtureWeightOverflowTest(unittest.TestCase):
    def test_normalizes_extreme_finite_weights_without_overflow(self):
        backend_dtype = to_numpy(array([1.0])).dtype
        max_weight = np.finfo(backend_dtype).max
        components = [
            GaussianDistribution(array([0.0]), array([[1.0]])),
            GaussianDistribution(array([1.0]), array([[1.0]])),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mixture = GaussianMixture(
                components,
                array([max_weight, max_weight / 2.0]),
            )

        npt.assert_allclose(to_numpy(mixture.w), np.array([2.0 / 3.0, 1.0 / 3.0]))


if __name__ == "__main__":
    unittest.main()
