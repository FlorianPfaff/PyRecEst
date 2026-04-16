import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import all, array, ones
from pyrecest.distributions import SE2BinghamDistribution


class TestSE2BinghamDistribution(unittest.TestCase):
    def setUp(self):
        """Set up a test SE2BinghamDistribution instance."""
        # Build a valid parameter set: C3 negative definite, C1 symmetric
        self.C1 = array([[-3.0, 0.5], [0.5, -1.0]])
        self.C2 = array([[0.1, 0.2], [-0.1, 0.3]])
        self.C3 = array([[-2.0, 0.1], [0.1, -1.5]])
        self.dist = SE2BinghamDistribution(self.C1, self.C2, self.C3)

    def test_constructor_from_parts(self):
        """Distribution can be constructed from C1, C2, C3."""
        dist = SE2BinghamDistribution(self.C1, self.C2, self.C3)
        self.assertIsInstance(dist, SE2BinghamDistribution)

    def test_constructor_from_full_matrix(self):
        """Distribution can be constructed from the full 4x4 matrix."""
        C_full = self.dist.C
        dist2 = SE2BinghamDistribution(C_full)
        self.assertIsInstance(dist2, SE2BinghamDistribution)
        npt.assert_array_almost_equal(dist2.C1, self.dist.C1)
        npt.assert_array_almost_equal(dist2.C2, self.dist.C2)
        npt.assert_array_almost_equal(dist2.C3, self.dist.C3)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_nc_positive(self):
        """Normalization constant must be positive."""
        self.assertGreater(self.dist.nc, 0.0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_pdf_positive(self):
        """PDF values must be positive."""
        # A few dual-quaternion-like points (norm of first two not necessarily 1 here,
        # but pdf is evaluated at arbitrary 4D points)
        points = array(
            [
                [1.0, 0.0, 0.5, -0.3],
                [0.0, 1.0, 0.1, 0.2],
                [0.7071, 0.7071, -0.2, 0.4],
            ]
        )
        vals = self.dist.pdf(points)
        self.assertTrue(all(vals > 0))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_pdf_from_angle_pos(self):
        """PDF should accept angle-pos (N x 3) input and give consistent results."""
        # Create angle-pos samples
        angle_pos = array([[0.5, 1.0, -1.0], [1.0, 0.0, 0.5]])
        p_ap = self.dist.pdf(angle_pos)

        # Convert to dual quaternion manually and evaluate
        from pyrecest.distributions import AbstractSE2Distribution

        dq = AbstractSE2Distribution.angle_pos_to_dual_quaternion(angle_pos)
        p_dq = self.dist.pdf(dq)
        npt.assert_array_almost_equal(p_ap, p_dq)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_mode_shape(self):
        """Mode should be a 4-element array."""
        m = self.dist.mode()
        self.assertEqual(m.shape, (4,))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_mode_is_local_maximum(self):
        """PDF at mode should be >= PDF at nearby on-manifold perturbed points."""
        from pyrecest.distributions import AbstractSE2Distribution

        # Mode is in dual-quaternion (DQ) representation
        m_dq = self.dist.mode().reshape(1, -1)

        # Convert mode DQ → angle-pos
        angle_arr, pos_arr = AbstractSE2Distribution.dual_quaternion_to_angle_pos(m_dq)
        angle0 = float(angle_arr.reshape(-1)[0])
        pos0 = pos_arr.reshape(-1)

        p_mode = float(self.dist.pdf(m_dq).reshape(-1)[0])

        rng = np.random.default_rng(42)
        for _ in range(20):
            # Perturb angle and position (stays on S^1 x R^2 manifold)
            angle_p = angle0 + rng.normal(0, 0.15)
            pos_p = [
                float(pos0[0]) + rng.normal(0, 0.15),
                float(pos0[1]) + rng.normal(0, 0.15),
            ]
            ap = array([[angle_p, pos_p[0], pos_p[1]]])
            dq_p = AbstractSE2Distribution.angle_pos_to_dual_quaternion(ap)
            p_perturbed = float(self.dist.pdf(dq_p).reshape(-1)[0])
            self.assertGreaterEqual(p_mode, p_perturbed - 1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_sample_shape(self):
        """sample() must return an (n, 4) array."""
        n = 50
        s = self.dist.sample(n)
        self.assertEqual(s.shape, (n, 4))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_fit_returns_instance(self):
        """fit() should return a valid SE2BinghamDistribution."""
        samples = self.dist.sample(500)
        fitted = SE2BinghamDistribution.fit(samples)
        self.assertIsInstance(fitted, SE2BinghamDistribution)
        self.assertGreater(fitted.nc, 0.0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_fit_weighted(self):
        """fit() should accept explicit weights."""
        n = 200
        samples = self.dist.sample(n)
        weights = ones(n) / n
        fitted = SE2BinghamDistribution.fit(samples, weights)
        self.assertIsInstance(fitted, SE2BinghamDistribution)


if __name__ == "__main__":
    unittest.main()
