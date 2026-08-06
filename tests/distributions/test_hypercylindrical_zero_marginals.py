import unittest
from math import pi

import numpy as np
from pyrecest.backend import __backend_name__ as backend_name
from pyrecest.backend import array
from pyrecest.distributions.cart_prod.hypercylindrical_state_space_subdivision_distribution import (
    HypercylindricalStateSpaceSubdivisionDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)


@unittest.skipIf(backend_name != "numpy", reason="Not supported on this backend")
class TestHypercylindricalZeroMarginals(unittest.TestCase):
    def test_from_function_handles_zero_marginal_grid_points(self):
        gaussian = GaussianDistribution(array([0.0]), array([[1.0]]))

        def joint_pdf(xs):
            angular_factor = np.sin(np.asarray(xs[:, 0])) ** 2
            return angular_factor * np.asarray(gaussian.pdf(xs[:, 1:]))

        dist = HypercylindricalStateSpaceSubdivisionDistribution.from_function(
            joint_pdf, 8, 1, 1
        )

        grid_values = np.asarray(dist.gd.grid_values)
        self.assertTrue(np.any(grid_values == 0.0))

        values = np.asarray(dist.pdf(array([[0.0, 0.0], [pi / 2.0, 0.0]])))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertEqual(float(values[0]), 0.0)
        self.assertGreater(float(values[1]), 0.0)

    def test_from_function_rejects_zero_total_grid_mass(self):
        def zero_pdf(xs):
            return np.zeros(xs.shape[0])

        with self.assertRaisesRegex(ValueError, "positive total mass"):
            HypercylindricalStateSpaceSubdivisionDistribution.from_function(
                zero_pdf, 8, 1, 1
            )


if __name__ == "__main__":
    unittest.main()
