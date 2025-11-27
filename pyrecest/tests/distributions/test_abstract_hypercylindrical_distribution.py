import unittest
from math import pi

import matplotlib
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, arange, array, column_stack, diff, ones, zeros
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)

class AbstractHypercylindricalDistributionTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on jax backend",
    )
    def test_mode_numerical_gaussian_2D(self):
        mu = array([5.0, 1.0])
        C = array([[2.0, 1.0], [1.0, 1.0]])
        g = PartiallyWrappedNormalDistribution(mu, C, 1)
        npt.assert_allclose(g.mode_numerical(), mu, atol=5e-5)

    def test_linear_mean_numerical(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        npt.assert_allclose(hwn.linear_mean_numerical(), hwn.mu[-1])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_condition_on_periodic(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        dist_cond1 = hwn.condition_on_periodic(array(1.5))
        # There is some normalization constant involved, therefore, test if ratio stays the same
        npt.assert_allclose(
            diff(
                hwn.pdf(column_stack([1.5 * ones(11), arange(-5, 6)]))
                / dist_cond1.pdf(arange(-5, 6))
            ),
            zeros(10),
            atol=1e-10,
        )
        dist_cond2 = hwn.condition_on_periodic(array(1.5) + 2.0 * pi)
        npt.assert_allclose(
            diff(
                hwn.pdf(column_stack([1.5 * ones(11), arange(-5, 6)]))
                / dist_cond2.pdf(arange(-5, 6))
            ),
            zeros(10),
            atol=1e-10,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_plot(self):
        matplotlib.pyplot.close("all")
        matplotlib.use("Agg")
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        hwn.plot()
        hwn.plot_cylinder()

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_condition_on_linear(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        dist_cond1 = hwn.condition_on_linear(array(1.5))
        npt.assert_allclose(
            diff(
                hwn.pdf(column_stack([arange(-5, 6), 1.5 * ones(11)]))
                / dist_cond1.pdf(arange(-5, 6))
            ),
            zeros(10),
            atol=1e-10,
        )
        dist_cond2 = hwn.condition_on_linear(array(1.5 + 2.0 * pi))
        self.assertFalse(
            (
                allclose(
                    diff(
                        hwn.pdf(column_stack([arange(-5, 6), 1.5 * ones(11)]))
                        / dist_cond2.pdf(arange(-5, 6))
                    ),
                    zeros(10),
                )
            )
        )
        
    def test_hybrid_moment_numerical_gaussian_2d(self):
        mu = np.array([5, 10])
        C = np.array([[2, 1], [1, 1]])
        g = PartiallyWrappedNormalDistribution(mu, C, 1)
        
        #integrand_cos = lambda x: np.cos(x) * (1/np.sqrt(2*np.pi*C[0, 0])) * np.exp(-((x-mu[0])**2)/(2*C[0, 0]))
        #integrand_sin = lambda x: np.sin(x) * (1/np.sqrt(2*np.pi*C[0, 0])) * np.exp(-((x-mu[0])**2)/(2*C[0, 0]))
        #expected_result = np.array([quad(integrand_cos, 0, 2*np.pi)[0],
        #                            quad(integrand_sin, 0, 2*np.pi)[0],
        #                            mu[-1]])
        
        np.testing.assert_allclose(g.hybrid_moment_numerical(), g.hybrid_moment())



if __name__ == "__main__":
    unittest.main()
