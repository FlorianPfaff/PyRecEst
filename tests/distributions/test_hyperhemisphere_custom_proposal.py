import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ones, sum, zeros
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_uniform_distribution import (
    HyperhemisphericalUniformDistribution,
)


class TestHyperhemisphereCustomProposal(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="The JAX custom-proposal signature is backend-specific.",
    )
    def test_custom_proposal_uses_default_start_point(self):
        distribution = HyperhemisphericalUniformDistribution(2)
        proposal_point = array([[0.0, 0.0, 1.0]])

        def proposal(_):
            return proposal_point

        samples = distribution.sample_metropolis_hastings(
            4,
            burn_in=1,
            skipping=1,
            proposal=proposal,
        )

        self.assertEqual(samples.shape, (4, distribution.input_dim))
        npt.assert_allclose(sum(samples**2, axis=1), ones(4), rtol=1e-10)
        npt.assert_array_less(-samples[:, -1], zeros(samples.shape[0]))


if __name__ == "__main__":
    unittest.main()
