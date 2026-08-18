import unittest

# pylint: disable=redefined-builtin,no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import allclose, array, hstack, linalg, ones, random, zeros
from pyrecest.distributions import HyperhemisphericalWatsonDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_dirac_distribution import (
    HyperhemisphericalDiracDistribution,
)
from pyrecest.filters.hyperhemispherical_particle_filter import (
    HyperhemisphericalParticleFilter,
)


class HyperhemisphericalParticleFilterTest(unittest.TestCase):
    def setUp(self):
        random.seed(1)
        self.n_particles = 2000
        self.watson_hemi_init = HyperhemisphericalWatsonDistribution(
            array([1.0, 0.0, 0.0]), 10.0
        )
        self.watson_hemi_sys = HyperhemisphericalWatsonDistribution(
            array([0.0, 0.0, 1.0]), 10.0
        )
        self.watson_hemi_meas = HyperhemisphericalWatsonDistribution(
            array([0.0, 0.0, 1.0]), 3.0
        )

    def test_initialization(self):
        hpf = HyperhemisphericalParticleFilter(self.n_particles, 3)
        self.assertEqual(hpf.filter_state.w.shape, (self.n_particles,))
        self.assertEqual(hpf.filter_state.d.shape, (self.n_particles, 4))
        expected_particles = hstack(
            (
                zeros((self.n_particles, 3)),
                ones((self.n_particles, 1)),
            )
        )
        self.assertTrue(bool(allclose(hpf.filter_state.d, expected_particles)))

    def test_constructor_rejects_invalid_particle_count(self):
        for n_particles in (True, 0, -1, 1.5):
            with self.subTest(n_particles=n_particles):
                with self.assertRaisesRegex(ValueError, "n_particles"):
                    HyperhemisphericalParticleFilter(n_particles, 3)

    def test_constructor_rejects_invalid_dimension(self):
        for dim in (True, 0, -1, 1.5):
            with self.subTest(dim=dim):
                with self.assertRaisesRegex(ValueError, "dim"):
                    HyperhemisphericalParticleFilter(5, dim)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_set_state(self):
        hpf = HyperhemisphericalParticleFilter(self.n_particles, 2)
        hpf.set_state(self.watson_hemi_init)
        point_est = hpf.get_point_estimate()
        watson_mode = self.watson_hemi_init.mode()
        is_close_to_mode = linalg.norm(point_est - watson_mode) < 1e-1
        is_close_to_neg_mode = linalg.norm(point_est + watson_mode) < 1e-1
        self.assertTrue(is_close_to_mode or is_close_to_neg_mode)
        self.assertIsInstance(hpf.filter_state, HyperhemisphericalDiracDistribution)

    def test_set_state_rejects_invalid_distribution(self):
        hpf = HyperhemisphericalParticleFilter(5, 2)

        with self.assertRaisesRegex(TypeError, "AbstractHyperhemispherical"):
            hpf.set_state(object())


if __name__ == "__main__":
    unittest.main()
