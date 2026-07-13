import unittest
from math import pi

import numpy.testing as npt
from pyrecest.backend import __backend_name__ as backend_name
from pyrecest.backend import array
from pyrecest.distributions.cart_prod.hypercylindrical_state_space_subdivision_distribution import (
    HypercylindricalStateSpaceSubdivisionDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)


class HypercylindricalSubdivisionPeriodicityTest(unittest.TestCase):
    @unittest.skipIf(
        backend_name != "numpy",
        reason="Not supported on this backend",
    )
    def test_pdf_uses_periodic_conditional_selection_across_multiple_turns(self):
        grid_distribution = HypertoroidalGridDistribution(
            array([1.0, 1.0]),
            grid_type="custom",
            grid=array([[0.0], [pi]]),
        )
        conditional_distributions = [
            GaussianDistribution(array([0.0]), array([[0.25]])),
            GaussianDistribution(array([5.0]), array([[0.25]])),
        ]
        distribution = HypercylindricalStateSpaceSubdivisionDistribution(
            grid_distribution,
            conditional_distributions,
        )

        reference = distribution.pdf(array([[0.1, 0.0]]))
        equivalent = distribution.pdf(array([[0.1 + 4.0 * pi, 0.0]]))

        npt.assert_allclose(equivalent, reference, rtol=0.0, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
