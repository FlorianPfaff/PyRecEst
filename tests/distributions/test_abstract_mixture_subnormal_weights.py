import unittest

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, eye, float64
from pyrecest.distributions.hypertorus.hypertoroidal_mixture import HypertoroidalMixture
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)


class AbstractMixtureSubnormalWeightsTest(unittest.TestCase):
    def test_subnormal_positive_weights_preserve_components_and_ratios(self):
        component = ToroidalWrappedNormalDistribution(array([1.0, 0.0]), eye(2))
        tiny_weight = np.finfo(np.float64).tiny / 1024.0

        mixture = HypertoroidalMixture(
            [component, component.shift(array([1.0, 1.0]))],
            array([2.0 * tiny_weight, tiny_weight], dtype=float64),
        )

        self.assertEqual(len(mixture.dists), 2)
        self.assertTrue(
            allclose(
                mixture.w,
                array([2.0 / 3.0, 1.0 / 3.0], dtype=float64),
            )
        )


if __name__ == "__main__":
    unittest.main()
