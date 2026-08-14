import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, to_numpy
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture


class GaussianMixtureComplexMeanTest(unittest.TestCase):
    def test_set_mean_rejects_complex_target_without_mutating_components(self):
        component_1 = GaussianDistribution(array([0.0, 1.0]), diag(array([1.0, 2.0])))
        component_2 = GaussianDistribution(array([2.0, 3.0]), diag(array([3.0, 4.0])))
        mixture = GaussianMixture([component_1, component_2], array([0.25, 0.75]))
        original_means = [to_numpy(dist.mu).copy() for dist in mixture.dists]

        invalid_means = (
            [10.0 + 2.0j, -2.0],
            np.array([10.0, -2.0 - 3.0j], dtype=np.complex64),
        )
        for invalid_mean in invalid_means:
            with self.subTest(invalid_mean=invalid_mean):
                with self.assertRaisesRegex(
                    ValueError, "new_mean must contain only real values"
                ):
                    mixture.set_mean(invalid_mean)

        for dist, original_mean in zip(mixture.dists, original_means):
            npt.assert_allclose(to_numpy(dist.mu), original_mean)
            self.assertFalse(np.iscomplexobj(to_numpy(dist.mu)))


if __name__ == "__main__":
    unittest.main()
