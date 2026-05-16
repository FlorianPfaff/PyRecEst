import unittest
from unittest.mock import patch

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)


class DeterministicOneDimensionalDistribution(AbstractManifoldSpecificDistribution):
    def __init__(self):
        super().__init__(dim=1)

    @property
    def input_dim(self):
        return 1

    def get_manifold_size(self):
        return 1.0

    def pdf(self, xs):
        value = float(to_numpy(xs).squeeze())
        if value == 0.0:
            return array(1.0)
        if value == 1.0:
            return array(0.1)
        return array(value)

    def mean(self):
        return array([0.0])


class AbstractManifoldSpecificDistributionTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="This regression test targets the non-JAX MH implementation.",
    )
    def test_sample_metropolis_hastings_records_rejections(self):
        distribution = DeterministicOneDimensionalDistribution()
        proposals = iter([array([1.0]), array([2.0]), array([3.0])])

        def proposal(_):
            return next(proposals)

        with patch(
            "pyrecest.distributions.abstract_manifold_specific_distribution.random.rand",
            return_value=0.5,
        ):
            samples = distribution.sample_metropolis_hastings(
                n=2, burn_in=0, skipping=1, proposal=proposal, start_point=array([0.0])
            )

        npt.assert_allclose(to_numpy(samples), [0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
