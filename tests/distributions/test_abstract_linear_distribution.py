import unittest

import matplotlib
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, isclose
from pyrecest.distributions import (
    AbstractLinearDistribution,
    CustomLinearDistribution,
    GaussianDistribution,
)

matplotlib.pyplot.close("all")
matplotlib.use("Agg")


class TestAbstractLinearDistribution(unittest.TestCase):
    def setUp(self):
        self.mu_2D = array([0.0, 0.0])
        self.C_2D = array([[2.0, 1.0], [1.0, 1.0]])
        self.mu_3D = array([1.0, 2.0, 3.0])
        self.C_3D = array([[1.1, 0.4, 0.0], [0.4, 0.9, 0.0], [0.0, 0.0, 1.0]])
        self.g_2D = GaussianDistribution(self.mu_2D, self.C_2D)
        self.g_3D = GaussianDistribution(self.mu_3D, self.C_3D)

    def test_integrate(self):
        """Test that the numerical integration of a Gaussian distribution equals 1."""
        dist = GaussianDistribution(array([1.0, 2.0]), diag(array([1.0, 2.0])))
        integration_result = dist.integrate_numerically()
        assert isclose(
            integration_result, 1.0, rtol=1e-5
        ), f"Expected 1.0, but got {integration_result}"

    def test_integrate_fun_over_domain(self):
        dist = GaussianDistribution(array([1.0, 2.0]), diag(array([1.0, 2.0])))

        def f(x):
            return 0.3 * dist.pdf(x)

        dim = 2
        left = [-float("inf"), -float("inf")]
        right = [float("inf"), float("inf")]

        integration_result = AbstractLinearDistribution.integrate_fun_over_domain(
            f, dim, left, right
        )
        assert isclose(
            integration_result, 0.3, rtol=1e-5
        ), f"Expected 0.3, but got {integration_result}"

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_mode_numerical_custom_1D(self):
        cd = CustomLinearDistribution(
            lambda x: array(
                ((x > -1.0) & (x <= 0.0)) * (1.0 + x)
                + ((x > 0.0) & (x <= 1.0)) * (1.0 - x)
            ).squeeze(),
            1,
        )
        cd = cd.shift(array(0.5))
        self.assertAlmostEqual(cd.mode_numerical(), 0.5, delta=1e-4)

    def test_mean_numerical_gaussian_2D(self):
        npt.assert_allclose(self.g_2D.mean_numerical(), self.mu_2D, atol=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_mode_numerical_gaussian_2D_mean_far_away(self):
        mu = array([5.0, 10.0])
        C = array([[2.0, 1.0], [1.0, 1.0]])
        g = GaussianDistribution(mu, C)
        npt.assert_allclose(g.mode_numerical(), mu, atol=2e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_mode_numerical_accepts_scalar_starting_point_for_1d(self):
        g = GaussianDistribution(array([1.0]), array([[0.7]]))

        npt.assert_allclose(g.mode_numerical(0.0), array([1.0]), atol=2e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_mode_numerical_accepts_list_starting_point(self):
        mu = array([5.0, 10.0])
        C = array([[2.0, 1.0], [1.0, 1.0]])
        g = GaussianDistribution(mu, C)

        npt.assert_allclose(g.mode_numerical([5.0, 10.0]), mu, atol=2e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_mode_numerical_rejects_non_vector_starting_point(self):
        g = GaussianDistribution(array([1.0, 2.0]), array([[1.0, 0.0], [0.0, 1.0]]))

        with self.assertRaisesRegex(ValueError, "1D array"):
            g.mode_numerical([[1.0, 2.0]])

    def test_sample_metropolis_hastings_rejects_wrong_start_point_dimension(self):
        g = GaussianDistribution(array([1.0, 2.0]), diag(array([1.0, 1.0])))

        with self.assertRaisesRegex(ValueError, "correct dimension"):
            g.sample_metropolis_hastings(1, start_point=[0.0])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_mode_numerical_gaussian_3D(self):
        npt.assert_allclose(self.g_3D.mode_numerical(), self.mu_3D, atol=5e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_covariance_numerical_gaussian_2D(self):
        npt.assert_allclose(self.g_2D.covariance_numerical(), self.C_2D, atol=1e-6)

    def test_plot_state_r2(self):
        gd = GaussianDistribution(array([1.0, 2.0]), array([[1.0, 0.5], [0.5, 1.0]]))
        gd.plot_state()

    def test_plot_state_r3(self):
        gd = GaussianDistribution(
            array([1.0, 2.0, 3.0]),
            array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        gd.plot_state()


if __name__ == "__main__":
    unittest.main()
