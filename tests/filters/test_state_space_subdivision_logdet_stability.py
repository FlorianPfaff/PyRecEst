import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter


@unittest.skipUnless(
    pyrecest.backend.__backend_name__ == "numpy",  # pylint: disable=no-member
    "The regression uses float64 covariances below float32 range.",
)
class TestStateSpaceSubdivisionLogdetStability(unittest.TestCase):
    def test_single_likelihood_accepts_tiny_positive_definite_covariances(self):
        covariance = array([[1.0e-200, 0.0], [0.0, 1.0e-200]])
        mean = array([0.0, 0.0])
        grid = HypertoroidalGridDistribution.from_distribution(
            CircularUniformDistribution(), (4,)
        )
        linear_distributions = [
            GaussianDistribution(mean, covariance) for _ in range(4)
        ]
        filter_ = StateSpaceSubdivisionFilter(
            StateSpaceSubdivisionGaussianDistribution(grid, linear_distributions)
        )
        likelihood = GaussianDistribution(mean, covariance)

        filter_.update(likelihoods_linear=[likelihood])

        expected_covariance = 0.5 * covariance
        for distribution in filter_.filter_state.linear_distributions:
            npt.assert_allclose(distribution.mu, mean, rtol=0.0, atol=0.0)
            npt.assert_allclose(
                distribution.C,
                expected_covariance,
                rtol=1.0e-12,
                atol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
