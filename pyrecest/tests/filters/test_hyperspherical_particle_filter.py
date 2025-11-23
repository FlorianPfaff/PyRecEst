import unittest
import numpy as np
import warnings
from pyrecest.distributions import VonMisesFisherDistribution
# from pyrecest.distributions
from pyrecest.filters.hyperspherical_particle_filter import HypersphericalParticleFilter

class HypersphericalParticleFilterTest(unittest.TestCase):
    def test3D(self):
        np.random.seed(1)
        n_samples = 20000
        vmf_init = VonMisesFisherDistribution(np.array([1, 0, 0]), 10)
        vmf_sys = VonMisesFisherDistribution(np.array([0, 0, 1]), 10)
        vmf_meas = VonMisesFisherDistribution(np.array([0, 0, 1]), 3)

        hpf = HypersphericalParticleFilter(n_samples, 3)
        hpf.filter_state = HypersphericalDiracDistribution(vmf_init.sample(n_samples))

        est = hpf.get_estimate_mean()
        self.assertIsInstance(est, np.ndarray)
        vmf_init_mean = vmf_init.moment()
        hpf_mean = est
        self.assertTrue(np.allclose(vmf_init_mean, hpf_mean, atol=1e-2))

        # Prediction step
        hpf.predict_identity(vmf_sys)

        # Update steps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hpf.update_identity(vmf_meas, np.array([0, 0, 1]))
            hpf.update_identity(vmf_meas, np.array([0, 0, 1]))
            hpf.update_identity(vmf_meas, np.array([0, 0, 1]))

        hpf_est_mean = hpf.get_estimate_mean()
        self.assertEqual(hpf_est_mean.shape, (3,))
        expected_mean = np.array([0, 0, 1])
        self.assertTrue(np.allclose(hpf_est_mean, expected_mean, atol=0.05))

        # Reset state
        hpf.filter_state = HypersphericalDiracDistribution(vmf_init.sample(n_samples))

        # Nonlinear update with a function that returns ones
        def f(z, x):
            return np.ones(x.shape[1])

        z = 3
        est = hpf.get_estimate_mean()
        self.assertAlmostEqual(np.linalg.norm(est), 1, delta=1e-10)
        hpf.update_nonlinear(f, z)
        self.assertTrue(np.allclose(est, hpf.get_estimate_mean(), atol=1e-2))
