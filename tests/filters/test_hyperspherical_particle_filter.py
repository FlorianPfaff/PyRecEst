import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, exp, linalg, ones, random
from pyrecest.distributions import (
    HypersphericalDiracDistribution,
    VonMisesFisherDistribution,
)
from pyrecest.filters.hyperspherical_particle_filter import HypersphericalParticleFilter


class HypersphericalParticleFilterTest(unittest.TestCase):
    def test_constructor_rejects_invalid_particle_count(self):
        for n_particles in (True, 0, -1, 1.5):
            with self.subTest(n_particles=n_particles):
                with self.assertRaisesRegex(ValueError, "n_particles"):
                    HypersphericalParticleFilter(n_particles, 3)

    def test_constructor_rejects_invalid_dimension(self):
        for dim in (True, 0, -1, 1.5):
            with self.subTest(dim=dim):
                with self.assertRaisesRegex(ValueError, "dim"):
                    HypersphericalParticleFilter(5, dim)

    def test_filter_state_assignment_does_not_alias_input(self):
        hpf = HypersphericalParticleFilter(2, 3)
        original_particles = array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        original_weights = array([0.5, 0.5])
        state = HypersphericalDiracDistribution(original_particles, original_weights)

        hpf.filter_state = state

        self.assertIsNot(hpf.filter_state, state)
        state.d = array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        state.w = array([0.9, 0.1])
        npt.assert_allclose(hpf.filter_state.d, original_particles)
        npt.assert_allclose(hpf.filter_state.w, original_weights)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test3D(self):
        random.seed(1)

        n_samples = 20000
        vmf_init_mean = array([1.0, 0.0, 0.0])
        vmf_init = VonMisesFisherDistribution(vmf_init_mean, 10)
        vmf_sys = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 10)
        vmf_meas = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 3)

        # Initialize filter
        hpf = HypersphericalParticleFilter(n_samples, 3)
        hpf.filter_state = HypersphericalDiracDistribution(vmf_init.sample(n_samples))
        npt.assert_allclose(hpf.get_point_estimate(), vmf_init_mean, atol=1e-2)

        # Prediction step
        hpf.predict_identity(vmf_sys)
        npt.assert_allclose(hpf.get_point_estimate(), vmf_init_mean, atol=1e-2)

        # Update steps
        enforced_mean = array([0.0, 0.0, 1.0])
        hpf.update_identity(vmf_meas, enforced_mean)
        hpf.update_identity(vmf_meas, enforced_mean)
        hpf.update_identity(vmf_meas, enforced_mean)
        hpf.predict_identity(vmf_sys)
        hpf.update_identity(vmf_meas, enforced_mean)
        hpf.update_identity(vmf_meas, enforced_mean)
        hpf.update_identity(vmf_meas, enforced_mean)

        hpf_est_mean = hpf.get_point_estimate()
        self.assertEqual(hpf_est_mean.shape, (3,))
        npt.assert_allclose(hpf_est_mean, enforced_mean, atol=0.25)

        # Reset state
        hpf.filter_state = HypersphericalDiracDistribution(vmf_init.sample(n_samples))

        # Nonlinear update with a function that returns ones
        def f1(_, x):
            return ones(x.shape[0])

        z = array(3.0)
        est = hpf.get_point_estimate()
        npt.assert_allclose(linalg.norm(est), 1, atol=1e-10)
        # Since the likelihood is constant, the estimate should not change.
        hpf.update_nonlinear(f1, z)
        npt.assert_allclose(est, hpf.get_point_estimate(), atol=1e-2)

        z = array([0.0, 1.0, 0.0])

        def f2(z, x):
            return exp(z @ x.T)

        hpf.update_nonlinear(f2, z)
        est = hpf.get_point_estimate()
        npt.assert_allclose(linalg.norm(est), 1, atol=1e-10)
        self.assertFalse(allclose(est, vmf_init_mean, atol=1e-2))

    def test_get_estimate_mean_rejects_zero_resultant(self):
        hpf = HypersphericalParticleFilter(2, 3)
        hpf.filter_state = HypersphericalDiracDistribution(
            array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            array([0.5, 0.5]),
        )

        with self.assertWarnsRegex(UserWarning, "Mean direction is undefined"):
            with self.assertRaisesRegex(ValueError, "Mean direction is undefined"):
                hpf.get_estimate_mean()
