import unittest

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
