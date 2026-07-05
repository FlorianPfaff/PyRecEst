import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture


class GaussianMixtureSetMeanValidationTest(unittest.TestCase):
    def test_set_mean_accepts_python_sequence(self):
        gm1 = GaussianDistribution(array([0.0, 1.0]), diag(array([1.0, 2.0])))
        gm2 = GaussianDistribution(array([2.0, 3.0]), diag(array([3.0, 4.0])))
        gmix = GaussianMixture([gm1, gm2], array([0.25, 0.75]))

        shifted = gmix.set_mean([10.0, -2.0])

        npt.assert_allclose(shifted.mean(), array([10.0, -2.0]))
        npt.assert_allclose(gmix.mean(), array([1.5, 2.5]))
        npt.assert_allclose(shifted.w, gmix.w)
        npt.assert_allclose(shifted.covariance(), gmix.covariance())

    def test_set_mean_rejects_wrong_shape(self):
        gm1 = GaussianDistribution(array([0.0, 1.0]), diag(array([1.0, 2.0])))
        gm2 = GaussianDistribution(array([2.0, 3.0]), diag(array([3.0, 4.0])))
        gmix = GaussianMixture([gm1, gm2], array([0.25, 0.75]))

        for invalid_mean in (1.0, [1.0], [[1.0, 2.0]], [1.0, 2.0, 3.0]):
            with self.subTest(invalid_mean=invalid_mean):
                with self.assertRaisesRegex(ValueError, "new_mean"):
                    gmix.set_mean(invalid_mean)


if __name__ == "__main__":
    unittest.main()
