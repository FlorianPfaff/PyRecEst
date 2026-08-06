import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, pi, random, zeros
from pyrecest.distributions.cart_prod.hypercylindrical_dirac_distribution import (
    HypercylindricalDiracDistribution,
)
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)
from pyrecest.filters.hypercylindrical_particle_filter import (
    HypercylindricalParticleFilter,
)


class HypercylindricalParticleFilterTest(unittest.TestCase):
    def setUp(self):
        self.seed = 0
        self.bound_dim = 1
        self.lin_dim = 2
        self.C = array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1.0]])
        self.mu = array([1.0, 1.0, 1.0]) + pi / 2
        self.pwn = PartiallyWrappedNormalDistribution(self.mu, self.C, self.bound_dim)
        random.seed(self.seed)

    def test_initialization(self):
        hpf = HypercylindricalParticleFilter(10, self.bound_dim, self.lin_dim)
        self.assertIsNotNone(hpf.filter_state)
        self.assertEqual(hpf.filter_state.d.shape, (10, self.bound_dim + self.lin_dim))

    def test_initialization_accepts_scalar_numpy_integer_counts(self):
        hpf = HypercylindricalParticleFilter(
            np.int64(4), np.array(1, dtype=np.int64), np.int64(2)
        )

        self.assertEqual(hpf.filter_state.d.shape, (4, 3))
        self.assertEqual(hpf.filter_state.bound_dim, 1)
        self.assertEqual(hpf.filter_state.lin_dim, 2)

    def test_initialization_rejects_invalid_particle_counts(self):
        invalid_counts = (
            0,
            -1,
            1.5,
            True,
            np.bool_(True),
            np.array(True),
            np.array([1]),
        )

        for n_particles in invalid_counts:
            with self.subTest(n_particles=n_particles):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    HypercylindricalParticleFilter(
                        n_particles, self.bound_dim, self.lin_dim
                    )

    def test_initialization_rejects_invalid_dimension_counts(self):
        invalid_bound_dims = (True, np.bool_(True), np.array(True), -1, 1.5, [1])
        for bound_dim in invalid_bound_dims:
            with self.subTest(bound_dim=bound_dim):
                with self.assertRaisesRegex(
                    ValueError, "bound_dim must be a non-negative integer"
                ):
                    HypercylindricalParticleFilter(4, bound_dim, self.lin_dim)

        invalid_lin_dims = (False, np.bool_(False), np.array(False), -1, 1.5, [2])
        for lin_dim in invalid_lin_dims:
            with self.subTest(lin_dim=lin_dim):
                with self.assertRaisesRegex(
                    ValueError, "lin_dim must be a non-negative integer"
                ):
                    HypercylindricalParticleFilter(4, self.bound_dim, lin_dim)

        with self.assertRaisesRegex(ValueError, "total dimension must be positive"):
            HypercylindricalParticleFilter(4, 0, 0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax", reason="Backend not supported"
    )
    def test_set_state(self):
        hpf = HypercylindricalParticleFilter(500, self.bound_dim, self.lin_dim)
        hpf.filter_state = self.pwn
        npt.assert_allclose(hpf.get_point_estimate(), self.mu, atol=0.1)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax", reason="Backend not supported"
    )
    def test_set_state_from_dirac(self):
        hpf = HypercylindricalParticleFilter(500, self.bound_dim, self.lin_dim)
        samples = self.pwn.sample(500)
        dirac_dist = HypercylindricalDiracDistribution(self.bound_dim, samples)
        hpf.filter_state = dirac_dist
        npt.assert_allclose(hpf.get_point_estimate(), self.mu, atol=0.1)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax", reason="Backend not supported"
    )
    def test_predict_update_cycle_3d(self):
        hpf = HypercylindricalParticleFilter(500, self.bound_dim, self.lin_dim)
        hpf.filter_state = self.pwn
        forced_mean = array([1.0, 10.0, 20.0])
        noise_predict = PartiallyWrappedNormalDistribution(
            zeros(3), self.C, self.bound_dim
        )
        noise_update = PartiallyWrappedNormalDistribution(
            zeros(3), 0.5 * self.C, self.bound_dim
        )
        for _ in range(50):
            hpf.predict_identity(noise_predict)
            self.assertEqual(hpf.get_point_estimate().shape, (3,))
            for _ in range(3):
                hpf.update_identity(noise_update, forced_mean)
        self.assertEqual(hpf.get_point_estimate().shape, (3,))
        npt.assert_allclose(hpf.get_point_estimate(), forced_mean, atol=0.2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax", reason="Backend not supported"
    )
    def test_predict_identity_shape(self):
        hpf = HypercylindricalParticleFilter(100, self.bound_dim, self.lin_dim)
        hpf.filter_state = self.pwn
        noise = PartiallyWrappedNormalDistribution(
            zeros(3), diag(array([0.1, 0.1, 0.1])), self.bound_dim
        )
        hpf.predict_identity(noise)
        self.assertEqual(hpf.get_point_estimate().shape, (3,))
        # Periodic dimensions should remain in [0, 2*pi)
        self.assertTrue(
            (hpf.filter_state.d[:, : self.bound_dim] >= 0).all()
            and (hpf.filter_state.d[:, : self.bound_dim] < 2.0 * pi).all()
        )


if __name__ == "__main__":
    unittest.main()
