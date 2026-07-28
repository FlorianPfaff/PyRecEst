import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)


class LinearDiracSupportValidationTest(unittest.TestCase):
    def test_constructor_rejects_invalid_support_values(self):
        invalid_locations = (
            (array([1.0 + 2.0j, 3.0 + 0.0j]), "real"),
            (array([1.0, np.nan]), "finite"),
            (array([1.0, np.inf]), "finite"),
        )

        for locations, message in invalid_locations:
            with self.subTest(locations=locations):
                with self.assertRaisesRegex(ValueError, message):
                    LinearDiracDistribution(locations)

    def test_constructor_rejects_rank_three_support(self):
        with self.assertRaisesRegex(ValueError, "scalar, 1D array, or 2D array"):
            LinearDiracDistribution(array([[[0.0], [1.0]]]))

    def test_set_mean_rejects_invalid_values_without_mutating_distribution(self):
        dist = LinearDiracDistribution(
            array([[0.0, 0.0], [2.0, 4.0]]), array([0.25, 0.75])
        )
        original_locations = np.asarray(to_numpy(dist.d)).copy()
        invalid_means = (
            (array([1.0 + 2.0j, 3.0 + 0.0j]), "real"),
            (array([1.0, np.nan]), "finite"),
            (array([1.0, np.inf]), "finite"),
        )

        for new_mean, message in invalid_means:
            with self.subTest(new_mean=new_mean):
                with self.assertRaisesRegex(ValueError, message):
                    dist.set_mean(new_mean)
                npt.assert_allclose(to_numpy(dist.d), original_locations)

    def test_moment_helper_rejects_invalid_samples(self):
        invalid_samples = (
            (array([1.0 + 2.0j, 3.0 + 0.0j]), "real"),
            (array([1.0, np.nan]), "finite"),
            (array([1.0, np.inf]), "finite"),
        )

        for samples, message in invalid_samples:
            with self.subTest(samples=samples):
                with self.assertRaisesRegex(ValueError, message):
                    LinearDiracDistribution.weighted_samples_to_mean_and_cov(samples)


if __name__ == "__main__":
    unittest.main()
