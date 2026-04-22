import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, linalg
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.hyperspherical_ukf import HypersphericalUKF


class HypersphericalUKFTest(unittest.TestCase):
    def setUp(self):
        if pyrecest.backend.__backend_name__ in ("jax",):
            return  # setUp deferred; individual tests are skipped
        self.filter_2d = HypersphericalUKF(dim=2)
        self.filter_3d = HypersphericalUKF(dim=3)
        self.gauss_2d = GaussianDistribution(
            array([1.0, 0.0]), array([[0.5, 0.0], [0.0, 0.5]])
        )
        self.gauss_3d = GaussianDistribution(
            array([1.0, 0.0, 0.0]),
            array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax",),
        reason="Not supported on this backend",
    )
    def test_initialization_2d(self):
        """Default initial state is on S^1."""
        est = self.filter_2d.get_point_estimate()
        npt.assert_allclose(float(linalg.norm(est)), 1.0, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax",),
        reason="Not supported on this backend",
    )
    def test_initialization_3d(self):
        """Default initial state is on S^2."""
        est = self.filter_3d.get_point_estimate()
        npt.assert_allclose(float(linalg.norm(est)), 1.0, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax",),
        reason="Not supported on this backend",
    )
    def test_filter_state_setter(self):
        """Setting filter_state stores it correctly."""
        self.filter_2d.filter_state = self.gauss_2d
        g = self.filter_2d.filter_state
        self.assertIsInstance(g, GaussianDistribution)
        npt.assert_array_equal(np.asarray(self.gauss_2d.mu), np.asarray(g.mu))
        npt.assert_array_equal(np.asarray(self.gauss_2d.C), np.asarray(g.C))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_predict_identity_preserves_mean(self):
        """Identity prediction with small noise keeps mean close to initial."""
        self.filter_3d.filter_state = self.gauss_3d
        small_noise = GaussianDistribution(
            array([0.0, 0.0, 0.0]),
            array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]),
        )
        self.filter_3d.predict_identity(small_noise)
        est = self.filter_3d.get_point_estimate()
        npt.assert_allclose(float(linalg.norm(est)), 1.0, atol=1e-10)
        npt.assert_allclose(est, self.gauss_3d.mu, atol=0.1)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_predict_identity_increases_uncertainty(self):
        """Identity prediction should increase the covariance trace."""
        self.filter_3d.filter_state = self.gauss_3d
        noise = GaussianDistribution(
            array([0.0, 0.0, 0.0]),
            array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]]),
        )
        trace_before = float(np.trace(np.asarray(self.gauss_3d.C, dtype=float)))
        self.filter_3d.predict_identity(noise)
        trace_after = float(
            np.trace(np.asarray(self.filter_3d.filter_state.C, dtype=float))
        )
        self.assertGreater(trace_after, trace_before)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_predict_nonlinear_identity_mean_on_sphere(self):
        """Nonlinear identity prediction keeps mean on the sphere."""
        self.filter_3d.filter_state = self.gauss_3d
        noise = GaussianDistribution(
            array([0.0, 0.0, 0.0]),
            array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]),
        )
        self.filter_3d.predict_nonlinear(lambda x: x, noise)
        est = self.filter_3d.get_point_estimate()
        npt.assert_allclose(float(linalg.norm(est)), 1.0, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_identity_reduces_uncertainty(self):
        """Identity update with a measurement equal to the mean reduces covariance."""
        self.filter_3d.filter_state = self.gauss_3d
        meas_noise = GaussianDistribution(
            array([0.0, 0.0, 0.0]),
            array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
        )
        trace_before = float(np.trace(np.asarray(self.gauss_3d.C, dtype=float)))
        self.filter_3d.update_identity(meas_noise, self.gauss_3d.mu)
        trace_after = float(
            np.trace(np.asarray(self.filter_3d.filter_state.C, dtype=float))
        )
        self.assertLess(trace_after, trace_before)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_identity_mean_stays_on_sphere(self):
        """Identity update keeps the mean on the sphere."""
        self.filter_3d.filter_state = self.gauss_3d
        meas_noise = GaussianDistribution(
            array([0.0, 0.0, 0.0]),
            array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
        )
        z = array([0.0, 1.0, 0.0])
        self.filter_3d.update_identity(meas_noise, z)
        est = self.filter_3d.get_point_estimate()
        npt.assert_allclose(float(linalg.norm(est)), 1.0, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_identity_shifts_mean_toward_measurement(self):
        """Update with off-axis measurement shifts the mean estimate."""
        self.filter_3d.filter_state = self.gauss_3d
        meas_noise = GaussianDistribution(
            array([0.0, 0.0, 0.0]),
            array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
        )
        z = array([0.0, 1.0, 0.0])
        self.filter_3d.update_identity(meas_noise, z)
        est = self.filter_3d.get_point_estimate()
        # Estimate should have moved away from [1,0,0] toward [0,1,0]
        init_mu = np.asarray(self.gauss_3d.mu, dtype=float)
        est_np = np.asarray(est, dtype=float)
        self.assertGreater(est_np[1], init_mu[1])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_nonlinear_identity_function(self):
        """Nonlinear update with identity function is equivalent to identity update."""
        import copy

        f3a = copy.deepcopy(self.filter_3d)
        f3b = copy.deepcopy(self.filter_3d)
        f3a.filter_state = self.gauss_3d
        f3b.filter_state = self.gauss_3d
        meas_noise = GaussianDistribution(
            array([0.0, 0.0, 0.0]),
            array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
        )
        z = array([0.0, 1.0, 0.0])
        f3a.update_identity(meas_noise, z)
        f3b.update_nonlinear(lambda x: x, meas_noise, z)
        npt.assert_allclose(
            np.asarray(f3a.get_point_estimate(), dtype=float),
            np.asarray(f3b.get_point_estimate(), dtype=float),
            atol=1e-10,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_predict_nonlinear_arbitrary_noise_mean_on_sphere(self):
        """predict_nonlinear_arbitrary_noise keeps mean on the sphere."""
        self.filter_3d.filter_state = self.gauss_3d
        noise_samples = np.random.default_rng(0).normal(
            size=(3, 10)
        )  # (noise_dim, n_noise)
        noise_weights = np.ones(10)

        def f(x, v):  # simple additive model, result will be normalized inside
            return array(np.asarray(x, dtype=float) + 0.01 * np.asarray(v, dtype=float))

        self.filter_3d.predict_nonlinear_arbitrary_noise(
            f, noise_samples, noise_weights
        )
        est = self.filter_3d.get_point_estimate()
        npt.assert_allclose(float(linalg.norm(est)), 1.0, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_get_point_estimate_returns_unit_vector(self):
        """get_point_estimate always returns a unit vector."""
        npt.assert_allclose(
            float(linalg.norm(self.filter_2d.get_point_estimate())), 1.0, atol=1e-10
        )
        npt.assert_allclose(
            float(linalg.norm(self.filter_3d.get_point_estimate())), 1.0, atol=1e-10
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_sequential_predict_update(self):
        """Full predict-update cycle produces a valid unit-vector estimate."""
        self.filter_3d.filter_state = self.gauss_3d
        sys_noise = GaussianDistribution(
            array([0.0, 0.0, 0.0]),
            array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]]),
        )
        meas_noise = GaussianDistribution(
            array([0.0, 0.0, 0.0]),
            array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]),
        )
        z = array([0.0, 0.0, 1.0])
        for _ in range(5):
            self.filter_3d.predict_identity(sys_noise)
            self.filter_3d.update_identity(meas_noise, z)
        est = self.filter_3d.get_point_estimate()
        npt.assert_allclose(float(linalg.norm(est)), 1.0, atol=1e-10)
        # After repeated updates toward [0,0,1], estimate should be close to it
        npt.assert_allclose(np.asarray(est, dtype=float), [0.0, 0.0, 1.0], atol=0.3)


if __name__ == "__main__":
    unittest.main()
