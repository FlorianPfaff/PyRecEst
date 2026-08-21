import unittest

import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.distributions.cart_prod.hypercylindrical_state_space_subdivision_gaussian_distribution import (
    HypercylindricalStateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.circle.circular_dirac_distribution import (
    CircularDiracDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)


class TestHypercylindricalStateSpaceSubdivisionGaussianDistribution(unittest.TestCase):
    def test_mode_uses_joint_peak_not_periodic_marginal_weight(self):
        periodic = CircularDiracDistribution(array([0.25, 1.25]), array([0.6, 0.4]))
        broad = GaussianDistribution(array([0.0]), array([[100.0]]))
        narrow = GaussianDistribution(array([5.0]), array([[0.01]]))
        distribution = HypercylindricalStateSpaceSubdivisionGaussianDistribution(
            periodic, [broad, narrow]
        )

        broad_peak = float(array(broad.pdf(broad.mode())).reshape(-1)[0])
        narrow_peak = float(array(narrow.pdf(narrow.mode())).reshape(-1)[0])
        self.assertGreater(0.4 * narrow_peak, 0.6 * broad_peak)

        npt.assert_allclose(distribution.mode(), array([1.25, 5.0]))


if __name__ == "__main__":
    unittest.main()
