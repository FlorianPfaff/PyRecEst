import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import ceil, linspace, pi
from pyrecest.distributions import CircularFourierDistribution, VonMisesDistribution


class TestCircularFourierLinspaceGridRegression(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_from_distribution_uses_exactly_requested_odd_grid_size(self):
        dist = VonMisesDistribution(2.5, 1.5)
        fd = CircularFourierDistribution.from_distribution(
            dist, n=999, transformation="identity"
        )
        self.assertEqual(fd.n, 999)
        self.assertEqual(fd.c.shape[0], ceil(999 / 2.0))
        xs = linspace(0.0, 2.0 * pi, 32, endpoint=False)
        npt.assert_allclose(fd.pdf(xs), dist.pdf(xs), rtol=2e-3, atol=5e-5)
